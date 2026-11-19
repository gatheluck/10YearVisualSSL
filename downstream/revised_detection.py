"""Source-aligned revised COCO bbox detection, distinct from classification."""
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from downstream.coco import SimpleFeaturePyramid
from downstream.native_detection import TransposedFeaturePyramid


class RevisedCoco(Dataset):
    def __init__(self, root, split):
        from pycocotools.coco import COCO
        if split not in ('train', 'val'):
            raise ValueError('split must be train or val')
        root = Path(root)
        self.train = split == 'train'
        self.img_dir = root / 'images' / f'{split}2017'
        self.annotation_file = root / 'annotations' / f'instances_{split}2017.json'
        self.coco = COCO(str(self.annotation_file))
        self.ids, self.cats = sorted(self.coco.getImgIds()), sorted(self.coco.getCatIds())
        self.category_to_label = {category: i + 1 for i, category in enumerate(self.cats)}
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        identifier = self.ids[index]
        info = self.coco.loadImgs(identifier)[0]
        with Image.open(self.img_dir / info['file_name']) as source:
            image = source.convert('RGB')
        boxes, labels = [], []
        for annotation in self.coco.loadAnns(self.coco.getAnnIds(imgIds=identifier)):
            x, y, width, height = annotation['bbox']
            if annotation.get('iscrowd', 0) or width <= 1 or height <= 1:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_to_label[annotation['category_id']])
        if self.train and random.random() < .5:
            image = cast(Any, TF.hflip)(image)
            boxes = [[image.width-x2, y1, image.width-x1, y2] for x1,y1,x2,y2 in boxes]
        return TF.to_tensor(image), {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
            'labels': torch.tensor(labels, dtype=torch.int64), 'image_id': torch.tensor([identifier])}


def collate(batch):
    return tuple(zip(*batch))


def _binding(body):
    family = getattr(body, 'family', None)
    style = getattr(body, 'native_pyramid_style', None)
    if family == 'map_pool':
        return [.5]*3, [.5]*3, 'forward_detection_features'
    if family == 'register_cls' or (family is None and style == 'bilinear'):
        return [.485,.456,.406], [.229,.224,.225], 'forward_features'
    if family is None and style == 'transposed':
        return [0.]*3, [1.]*3, 'forward_detection_features'
    raise ValueError('provider lacks a revised detection normalization binding')


def detection_features(backbone, probe):
    """Differentiable representative map from raw RGB (before detector resize)."""
    mean, std, method = _binding(backbone)
    return getattr(backbone, method)(TF.normalize(probe, mean, std)).float()


class DetectionBody(nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        _, _, self.readout = _binding(body)
        style = body.native_pyramid_style
        if style not in ('bilinear', 'transposed'):
            raise ValueError('unsupported revised feature pyramid')
        constructor = SimpleFeaturePyramid if style == 'bilinear' else TransposedFeaturePyramid
        self.fpn = constructor(body.out_channels)
        self.out_channels = 256

    def train(self, mode=True):
        super().train(mode)
        self.body.eval()
        return self

    def forward(self, images):
        return self.fpn(getattr(self.body, self.readout)(images).float())


def build_detector(backbone, diagnostic=None):
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    diagnostic = diagnostic or {}
    if diagnostic.keys() - {'size', 'proposals'} or any(type(v) is not int or v < 1 for v in diagnostic.values()):
        raise ValueError('invalid detector diagnostic overrides')
    mean, std, _ = _binding(backbone)
    options = {}
    if 'proposals' in diagnostic:
        for key in ('rpn_pre_nms_top_n_train','rpn_post_nms_top_n_train',
                    'rpn_pre_nms_top_n_test','rpn_post_nms_top_n_test'):
            options[key] = diagnostic['proposals']
    return FasterRCNN(DetectionBody(backbone), num_classes=91,
        rpn_anchor_generator=AnchorGenerator(((32,),(64,),(128,),(256,)), ((.5,1.,2.),)*4),
        box_roi_pool=MultiScaleRoIAlign(['0','1','2','3'],7,2),
        min_size=diagnostic.get('size',800), max_size=diagnostic.get('size',1333),
        image_mean=mean, image_std=std, **options)


def detector_parameter_groups(model, adapters, recipe):
    adapters = adapters.values() if isinstance(adapters, Mapping) else adapters
    expected = {id(p) for adapter in adapters for p in (adapter.lora_A, adapter.lora_B)}
    actual = {id(p) for p in model.backbone.body.parameters() if p.requires_grad}
    if not expected or expected != actual:
        raise ValueError('detector backbone trainables must be exactly the installed adapters')
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) not in expected and not name.startswith(('backbone.fpn.', 'rpn.', 'roi_heads.')):
            raise ValueError('unexpected trainable detector parameter')
        (no_decay if parameter.ndim < 2 or 'norm' in name.lower() else decay).append(parameter)
    return [{'params': decay, 'lr': recipe.head_lr, 'base_lr': recipe.head_lr, 'weight_decay': recipe.weight_decay},
            {'params': no_decay, 'lr': recipe.head_lr, 'base_lr': recipe.head_lr, 'weight_decay': 0.}]

def evaluate_detection(model, loader, device):
    from pycocotools.cocoeval import COCOeval
    from torch import distributed as dist

    from downstream.revised_engine import _autocast, _stage
    device = torch.device(device)
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    model.eval()

    def collect():
        rows, identifiers = [], []
        with torch.no_grad():
            for images, targets in loader:
                with _autocast(device):
                    predictions = model([image.to(device) for image in images])
                if len(predictions) != len(targets):
                    raise ValueError('detector prediction population differs')
                for prediction, target in zip(predictions, targets, strict=True):
                    identifier = int(target['image_id'].item())
                    identifiers.append(identifier)
                    boxes = prediction['boxes'].detach().float().cpu()
                    scores = prediction['scores'].detach().float().cpu()
                    labels = prediction['labels'].detach().cpu()
                    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                        raise ValueError('non-finite detector predictions')
                    for box, score, label in zip(boxes, scores, labels, strict=True):
                        label = int(label)
                        if label not in dataset.label_to_category or label > 80:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        rows.append({'image_id': identifier, 'category_id': dataset.label_to_category[label],
                            'bbox': [x1,y1,x2-x1,y2-y1], 'score': float(score)})
        return rows, identifiers

    local = _stage(collect, device)
    distributed = dist.is_available() and dist.is_initialized()
    parts: list[Any] = [local]
    if distributed:
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, local)
    rows = [row for part in parts for row in part[0]]
    identifiers = [identifier for part in parts for identifier in part[1]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('evaluation requires unique nonempty image population')

    def score():
        if distributed and dist.get_rank() != 0:
            return None
        names = ('bbox_ap','bbox_ap50','bbox_ap75','bbox_aps','bbox_apm','bbox_apl')
        values = [0.] * 6
        if rows:
            evaluator = COCOeval(dataset.coco, dataset.coco.loadRes(rows), 'bbox')
            evaluator.params.imgIds = sorted(identifiers)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            values = [float(value)*100 for value in evaluator.stats[:6]]
        return dict(zip(names, values, strict=True), n_images=len(identifiers), n_detections=len(rows))
    result = _stage(score, device)
    if distributed:
        objects = [result]
        dist.broadcast_object_list(objects, src=0)
        result = objects[0]
    return result
