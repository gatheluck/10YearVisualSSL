"""Explicit Extended image LP/AP components; never a paper-score certification."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from downstream import contract
from downstream.attention import QueryReader
from downstream import extended_execution as execution
from downstream.extended_resume import Continuation
from downstream import extended_distributed as distributed
from downstream.captured_readers import CrossSelfQueryReader, SINGLE_BLOCK, CROSS_SELF
from downstream.hf_vision import vision_no_decay
from downstream.spatial_backbones import build_frozen_backbone, discover_providers, _load_provider

TASK = 'extended_image_classification'
PROFILE = 'capture_extended_components'
PROTOCOLS = Path(__file__).resolve().parents[1] / 'docs' / 'submission_protocols'
METRICS = {'top1': 'extended_image_top1', 'top5': 'extended_image_top5',
           'images': None, 'epochs': 'epochs_completed'}


def recipe(dataset, adaptation):
    if adaptation not in ('frozen', 'attentive'):
        raise ValueError('Extended image execution supports LP/AP only')
    parent = json.loads((PROTOCOLS / 'EXTEND_LINEAR_v1.json').read_text())
    entry = parent['datasets'].get(dataset, {})
    rid = entry.get('recipe')
    if rid not in ('image_cls_100', 'image_cls_small_100'):
        raise ValueError('dataset requires an Extended image classification recipe')
    if entry.get('primary_metric') != 'top-1 accuracy':
        raise ValueError('dataset requires a different metric; pooled top-1 cannot substitute')
    result = dict(parent['recipes'][rid]['optimization'])
    if adaptation == 'attentive':
        ap = json.loads((PROTOCOLS / 'EXTEND_ATTENTIVE_v1.json').read_text())
        result = dict(ap['recipe_overrides'][rid]['optimization'])
    return result


def frozen_global_features(backbone, images):
    """Canonical normalized global readout, including token-only providers."""
    if callable(getattr(backbone, 'classification_features', None)):
        features = backbone.classification_features(images, adaptation='frozen').float()
    else:
        features = backbone.forward_features(images).float().mean((2, 3))
    return F.normalize(features, dim=-1)


class Classifier(nn.Module):
    """Frozen canonical global readout or explicitly selected spatial AP reader."""
    def __init__(self, backbone, classes, adaptation, reader_profile):
        super().__init__()
        if (adaptation not in ('frozen', 'attentive') or
            (adaptation == 'frozen' and reader_profile is not None) or
            (adaptation == 'attentive' and reader_profile not in (SINGLE_BLOCK, CROSS_SELF))):
            raise ValueError('explicit LP/AP reader selection required')
        if type(classes) is not int or classes < 2:
            raise ValueError('classification requires at least two classes')
        self.backbone = backbone.requires_grad_(False).eval()
        self.reader = ((CrossSelfQueryReader if reader_profile == CROSS_SELF else QueryReader)(backbone.out_channels)
                       if adaptation == 'attentive' else None)
        channels = 512 if self.reader else getattr(backbone, 'global_channels', backbone.out_channels)
        self.head = nn.Linear(channels, classes)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            if self.reader is not None:
                features = self.backbone.forward_features(images).float().flatten(2).transpose(1, 2)
            else:
                features = frozen_global_features(self.backbone, images)
        features = self.reader(features) if self.reader is not None else features
        return self.head(features)


class Images(Dataset):
    def __init__(self, root, rows, *, train, profile):
        self.root, self.rows = Path(root), rows
        if profile not in ('captured_rgb_rrc_v1', 'captured_small_crop_v1', 'captured_small_no_flip_v1'):
            raise ValueError('unknown explicit image transform profile')
        if not train:
            ops = [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)]
        elif profile == 'captured_rgb_rrc_v1':
            ops = [T.RandomResizedCrop(224, scale=(.08, 1.), interpolation=T.InterpolationMode.BICUBIC)]
        else:
            ops = [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.RandomCrop(224, padding=4)]
        if train and profile != 'captured_small_no_flip_v1':
            ops.append(T.RandomHorizontalFlip(.5))
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Broken samples must not become labelled black-image training examples.
        with Image.open(self.root / row['path']) as im:
            return self.transform(im.convert('RGB')), row['target']


def load_data(manifest, root, profile, *, dataset_factory=Images, allow_directories=False):
    raw = Path(manifest).read_bytes()
    data = json.loads(raw)
    if (set(data) != {'schema_version','classes','split_evidence','train','validation'} or
        type(data['schema_version']) is not int or data['schema_version'] != 1):
        raise ValueError('invalid sample manifest schema')
    classes = data['classes']
    if (not isinstance(classes, list) or len(classes) < 2 or
        any(not isinstance(x,str) or not x.strip() for x in classes) or len(set(classes)) != len(classes)):
        raise ValueError('classes must be a unique ordered label vocabulary')
    if not isinstance(data['split_evidence'], str) or not data['split_evidence'].strip():
        raise ValueError('split evidence must be recorded; no inferred random split')
    seen = set()
    root = Path(root).resolve()
    for split in ('train','validation'):
        rows = data[split]
        if not isinstance(rows,list) or not rows:
            raise ValueError('both splits must be nonempty')
        for row in rows:
            if not isinstance(row,dict) or set(row) != {'path','target'} or not isinstance(row['path'],str):
                raise ValueError('sample requires path and target only')
            path = Path(row['path'])
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('sample paths must be relative and remain under data_root')
            resolved = (root/path).resolve()
            if not resolved.is_relative_to(root) or resolved in seen:
                raise ValueError('duplicate, overlapping or escaping sample path')
            if not resolved.is_file() and not (allow_directories and resolved.is_dir()):
                raise ValueError('sample file is missing')
            if type(row['target']) is not int or not 0 <= row['target'] < len(classes):
                raise ValueError('sample target is outside the declared class vocabulary')
            seen.add(resolved)
    return (dataset_factory(root,data['train'],train=True,profile=profile),
            dataset_factory(root,data['validation'],train=False,profile=profile),
            dict(manifest_sha256=contract.sha256_bytes(raw),classes=classes,
                 split_evidence=data['split_evidence'], train_count=len(data['train']),
                 validation_count=len(data['validation']), official_membership_verified=False))


def score(logits, targets):
    if (logits.ndim != 2 or min(logits.shape) < 1 or targets.shape != (len(logits),) or
        targets.dtype != torch.long or not torch.isfinite(logits).all() or
        not ((targets >= 0) & (targets < logits.shape[1])).all()):
        raise ValueError('invalid or empty classification evaluation population')
    correct = logits.topk(min(5,logits.shape[1]),dim=1).indices.eq(targets[:,None])
    return dict(top1=100.*correct[:,0].double().mean().item(),
                top5=100.*correct.any(1).double().mean().item(), images=len(targets))


def optimizer(model, settings, *, batch_size, steps_per_epoch):
    if type(batch_size) is not int or batch_size < 1 or type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError('positive effective batch and microbatch steps required')
    lr = settings['base_lr'] * batch_size / settings['lr_reference_effective_batch']
    groups = []
    for no_decay in (False,True):
        params = [p for n,p in model.named_parameters() if p.requires_grad and vision_no_decay(n) == no_decay]
        if params:
            groups.append(dict(params=params,weight_decay=0. if no_decay else settings['weight_decay']))
    if settings['optimizer'] == 'SGD':
        opt = torch.optim.SGD(groups,lr=lr,momentum=settings['momentum'])
    elif settings['optimizer'] == 'AdamW':
        opt = torch.optim.AdamW(groups,lr=lr,betas=tuple(settings['betas']))
    else:
        raise ValueError('unsupported optimizer')
    warm_epochs = {'none':0, '1 epoch from 1e-6':1, '5 epochs from 1e-6':5}
    if settings['warmup'] not in warm_epochs:
        raise ValueError('unresolved warmup specification')
    warm = warm_epochs[settings['warmup']] * steps_per_epoch
    total = settings['epochs'] * steps_per_epoch
    floor = 1e-6 if settings['schedule'] == 'cosine to 1e-6' else 0.
    if settings['schedule'] not in ('cosine to 0','cosine to 1e-6'):
        raise ValueError('unresolved schedule specification')
    def factor(step):
        if step < warm:
            return (1e-6 + (lr-1e-6)*step/max(1,warm))/lr
        t = min(max((step-warm)/max(1,total-warm),0.),1.)
        return (floor + .5*(lr-floor)*(1+math.cos(math.pi*t)))/lr
    return opt, torch.optim.lr_scheduler.LambdaLR(opt,factor)


def validate_config(cfg, *, task=TASK, get_recipe=recipe, reader_attribute="EXTENDED_IMAGE_READER", max_epochs=100):
    required = {'task','profile','dataset','seed','device','data_root','samples','transform_profile',
                'adaptation','reader_profile','backbone','probe'}
    if set(cfg)-{'execution','resume'} != required or cfg['task'] != task or cfg['profile'] != PROFILE:
        raise ValueError('explicit Extended component config required')
    if cfg['seed'] != 0 or type(cfg['seed']) is not int:
        raise ValueError('Extended protocol uses seed 0')
    distributed.launch()
    get_recipe(cfg['dataset'],cfg['adaptation'])
    probe=cfg['probe']
    if set(probe) != {'epochs','batch_size','num_workers'}:
        raise ValueError('probe requires epochs, batch_size, num_workers')
    for name in probe:
        if type(probe[name]) is not int or probe[name] < (0 if name=='num_workers' else 1):
            raise ValueError('invalid probe integer')
    if probe['epochs'] > max_epochs:
        raise ValueError('component execution cannot exceed the recipe epoch horizon')
    path=discover_providers().get(cfg['backbone'].get('kind'))
    module=_load_provider(path) if path else None
    expected=getattr(module,reader_attribute,None)
    if expected not in (SINGLE_BLOCK,CROSS_SELF):
        raise ValueError('provider lacks a verified Extended image readout')
    if cfg['reader_profile'] != (expected if cfg['adaptation']=='attentive' else None):
        raise ValueError('reader_profile does not match the Extended provider recipe')
    return execution.resolve_execution(cfg,module)


def run(cfg,out):
    with distributed.session(cfg['device']) as context:
        return _run(cfg,out,context)


def _run(cfg,out,context, *, api=None):
    if api is None:
        import sys
        api=sys.modules[__name__]
    plan=context.call(lambda:api.validate_config(cfg))
    context.agree(cfg)
    context.seed(cfg['seed']); device=context.device
    execution.autocast_context(device,plan['precision'])
    train,val,membership=context.call(lambda:api.load_data(cfg['samples'],cfg['data_root'],cfg['transform_profile']))
    context.agree(membership)
    settings=cfg['probe']
    loader=context.call(lambda:context.loader(train,settings,cfg['seed']))
    validation=DataLoader(val,batch_size=settings['batch_size'],num_workers=settings['num_workers'])
    model=context.call(lambda:api.Classifier(api.build_frozen_backbone(cfg['backbone'],device),
        len(membership['classes']),cfg['adaptation'],cfg['reader_profile']).to(device))
    raw_model=model
    model=context.wrap(model)
    settings_recipe=api.recipe(cfg['dataset'],cfg['adaptation'])
    opt,scheduler=api.optimizer(raw_model,settings_recipe,batch_size=plan['effective_batch'],steps_per_epoch=len(loader))
    runtime=execution.execution_report(plan,settings_recipe,len(loader),scheduler.base_lrs[0])
    def forward(images):
        with execution.autocast_context(device,plan['precision']):
            return model(images.to(device))
    def loss_for_batch(batch):
        images,targets=batch
        return F.cross_entropy(forward(images).float(),targets.to(device))
    continuation=Continuation(cfg,membership,raw_model,opt,scheduler,runtime,loader,context,out)
    for epoch in range(continuation.start_epoch,settings['epochs']):
        if context.world>1: loader.sampler.set_epoch(epoch)
        statistics=execution.train_epoch(model,loader,loss_for_batch,opt,scheduler,
            accumulation_steps=plan['accumulation_steps'],tail_policy=plan['tail_policy'],adaptation=cfg['adaptation'])
        execution.record_epoch(runtime,statistics)
        continuation.save(epoch+1)
    model=raw_model
    out=Path(out)
    def evaluate_and_save():
        model.eval()
        logits,targets=[],[]
        with torch.no_grad():
            for images,y in validation:
                logits.append(forward(images).float().cpu()); targets.append(y)
        raw=api.score(torch.cat(logits),torch.cat(targets))
        raw['epochs']=settings['epochs']
        contract.write_metrics(out,raw,api.METRICS)
        torch.save({k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.')},out/'probe.pt')
        report=dict(task=api.TASK,profile=api.PROFILE,dataset=cfg['dataset'],adaptation=cfg['adaptation'],
                    backbone=cfg['backbone'],reader_profile=cfg['reader_profile'],transform_profile=cfg['transform_profile'],
                    membership=membership,recipe=settings_recipe,physical_batch=settings['batch_size'],
                    updates=scheduler.last_epoch,execution=runtime,continuation=continuation.report(),final=raw,canonical_eligible=False,record_value=False,
                    limitations=['legacy native checkpoint import not supported','official split membership unverified',
                                 'released-weight and full-score parity unverified'])
        (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        return raw

    return context.call(evaluate_and_save,leader=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    data=Path(args.config).read_bytes()
    return distributed.cli(data,args.out,run,TASK)


if __name__=='__main__': raise SystemExit(main())
