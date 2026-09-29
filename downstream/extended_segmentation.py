"""Extended semantic LP/AP components with explicit mask and metric identity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import traceback

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from downstream import contract
from downstream.attention import SpatialAdapter, clip_attentive_gradients
from downstream.captured_readers import SingleBlockSpatialAdapter, SINGLE_BLOCK, CROSS_SELF
from downstream.extended_classification import PROTOCOLS, optimizer
from downstream.spatial_backbones import build_frozen_backbone, discover_providers, _load_provider
from downstream.ssv2 import make_deterministic, resolve_device

TASK = 'extended_semantic_segmentation'
PROFILE = 'capture_extended_components'
TRANSFORMS = ('captured_fixed224_v1', 'captured_pair_crop224_v1', 'captured_ade_crop224_v1')
METRICS = {'miou':'extended_semantic_miou', 'pixel_accuracy':'extended_semantic_pixel_accuracy',
           'pixels':None, 'images':None, 'classes_with_union':None, 'epochs':'epochs_completed'}


def dataset_spec(dataset):
    entry = json.loads((PROTOCOLS/'EXTEND_LINEAR_v1.json').read_text())['datasets'].get(dataset,{})
    if (entry.get('recipe') != 'semantic_seg_20' or entry.get('primary_metric') != 'mIoU'
            or type(entry.get('head_outputs')) is not int):
        raise ValueError('semantic recipe, fixed ontology and mIoU must agree; unresolved dataset')
    return entry


def recipe(dataset, adaptation):
    dataset_spec(dataset)
    if adaptation not in ('frozen','attentive'):
        raise ValueError('Extended semantic execution supports LP/AP only')
    filename = 'EXTEND_ATTENTIVE_v1.json' if adaptation=='attentive' else 'EXTEND_LINEAR_v1.json'
    key = 'recipe_overrides' if adaptation=='attentive' else 'recipes'
    result = dict(json.loads((PROTOCOLS/filename).read_text())[key]['semantic_seg_20']['optimization'])
    result['betas'] = [.9,.999]  # Explicit in the captured trainer, absent from the catalog.
    return result


class DenseProbe(nn.Module):
    def __init__(self, backbone, classes, reader_profile):
        super().__init__()
        if type(classes) is not int or not 2 <= classes <= 255:
            raise ValueError('semantic classes must exclude the ignore ID 255')
        if reader_profile not in (None,SINGLE_BLOCK,CROSS_SELF):
            raise ValueError('unknown dense reader profile')
        self.backbone = backbone.requires_grad_(False).eval()
        # Reference construction creates the head before the optional adapter.
        self.head = nn.Conv2d(backbone.out_channels,classes,1)
        nn.init.normal_(self.head.weight,std=.01); nn.init.zeros_(self.head.bias)
        self.adapter = ((SingleBlockSpatialAdapter if reader_profile==SINGLE_BLOCK else SpatialAdapter)
                        (backbone.out_channels) if reader_profile is not None else None)

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self, images, out_hw):
        with torch.no_grad(): features = self.backbone.forward_features(images).float()
        if self.adapter is not None: features = self.adapter(features)
        return F.interpolate(self.head(features),size=out_hw,mode='bilinear',align_corners=False)


def _targets(targets, classes):
    if targets.dtype != torch.long or targets.ndim != 3:
        raise ValueError('semantic targets require integer [B,H,W] masks')
    if not (((targets>=0)&(targets<classes))|(targets==255)).all():
        raise ValueError('semantic target outside explicit ontology')
    return targets!=255


def pixel_loss(logits, targets):
    if (logits.ndim!=4 or logits.shape[0]!=targets.shape[0] or
        logits.shape[-2:]!=targets.shape[-2:] or not torch.isfinite(logits).all()):
        raise ValueError('invalid semantic logits')
    if not _targets(targets,logits.shape[1]).any():
        raise ValueError('training batch has no valid target pixels')
    return F.cross_entropy(logits,targets,ignore_index=255)


class SegmentationScore:
    def __init__(self, classes):
        if type(classes) is not int or not 2<=classes<=255:
            raise ValueError('invalid semantic class count')
        self.classes=classes; self.conf=torch.zeros(classes,classes,dtype=torch.int64); self.images=0

    def update(self, predictions, targets):
        valid=_targets(targets,self.classes)
        if (predictions.dtype!=torch.long or predictions.shape!=targets.shape or
            predictions.numel()==0 or not ((predictions>=0)&(predictions<self.classes)).all()):
            raise ValueError('invalid semantic predictions')
        ids=(targets[valid]*self.classes+predictions[valid]).cpu()
        self.conf += torch.bincount(ids,minlength=self.classes**2).reshape(self.classes,self.classes)
        self.images += len(targets)

    def result(self):
        pixels=int(self.conf.sum())
        if not pixels: raise ValueError('evaluation has no valid target pixels')
        inter=self.conf.diag().double(); union=self.conf.sum(0)+self.conf.sum(1)-self.conf.diag()
        present=union>0
        return dict(miou=float((inter[present]/union[present]).mean()*100),
                    pixel_accuracy=float(inter.sum()*100/pixels),pixels=pixels,images=self.images,
                    classes_with_union=int(present.sum()))


class MaskPairs(Dataset):
    def __init__(self, root, rows, mapping, *, train, profile):
        self.root,self.rows,self.mapping=Path(root),rows,mapping
        self.train,self.profile=train,profile

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(self.root/row['image']) as raw: image=raw.convert('RGB')
        with Image.open(self.root/row['mask']) as raw:
            if raw.mode not in ('L','P','I','I;16'): raise ValueError('color masks are not class IDs')
            ids=np.asarray(raw).copy()
        if ids.ndim!=2 or ids.dtype.kind not in 'iu' or ids.shape!=(image.height,image.width):
            raise ValueError('mask must contain integer IDs and match image geometry')
        observed=set(map(int,np.unique(ids)))
        if not observed<=set(self.mapping): raise ValueError('unknown native mask ID')
        mapped=np.full(ids.shape,255,dtype=np.uint8)
        for value in observed: mapped[ids==value]=self.mapping[value]
        mask=Image.fromarray(mapped)
        if self.profile in ('captured_pair_crop224_v1','captured_ade_crop224_v1'):
            if self.profile=='captured_pair_crop224_v1' and max(image.size)>512:
                factor=512/max(image.size)
                size=tuple(max(224,round(v*factor)) for v in image.size)
                image=image.resize(size,Image.Resampling.BICUBIC); mask=mask.resize(size,Image.Resampling.NEAREST)
            if self.train:
                scale=random.uniform(.5,2.)
                size=tuple(max(224,round(v*scale)) for v in image.size)
                image=image.resize(size,Image.Resampling.BICUBIC); mask=mask.resize(size,Image.Resampling.NEAREST)
                top=random.randint(0,size[1]-224); left=random.randint(0,size[0]-224)
                box=(left,top,left+224,top+224); image=image.crop(box); mask=mask.crop(box)
                if random.random()<.5: image=TF.hflip(image); mask=TF.hflip(mask)
        image=image.resize((224,224),Image.Resampling.BICUBIC)
        mask=mask.resize((224,224),Image.Resampling.NEAREST)
        return (TF.normalize(TF.to_tensor(image),(.485,.456,.406),(.229,.224,.225)),
                torch.from_numpy(np.asarray(mask,dtype=np.int64).copy()))


def load_data(manifest, root, profile):
    if profile not in TRANSFORMS: raise ValueError('explicit segmentation transform profile required')
    raw=Path(manifest).read_bytes(); data=json.loads(raw); root=Path(root).resolve()
    if (set(data)!={'schema_version','classes','label_map','split_evidence','train','validation'} or
        type(data['schema_version']) is not int or data['schema_version']!=1):
        raise ValueError('invalid semantic sample schema')
    classes=data['classes']
    if (not isinstance(classes,list) or not 2<=len(classes)<=255 or
        any(not isinstance(v,str) or not v.strip() for v in classes) or len(set(classes))!=len(classes)):
        raise ValueError('explicit unique ordered class vocabulary required')
    if not isinstance(data['split_evidence'],str) or not data['split_evidence'].strip():
        raise ValueError('split evidence required')
    labels=data['label_map']
    if (not isinstance(labels,dict) or not labels or any(not k.isdecimal() or str(int(k))!=k or
        type(v) is not int or (v!=255 and not 0<=v<len(classes)) for k,v in labels.items())):
        raise ValueError('canonical native IDs and valid training IDs required')
    mapping={int(k):v for k,v in labels.items()}; seen=set()
    for split in ('train','validation'):
        if not isinstance(data[split],list) or not data[split]: raise ValueError('nonempty splits required')
        for row in data[split]:
            if not isinstance(row,dict) or set(row)!={'image','mask'}: raise ValueError('image-mask pair required')
            for value in row.values():
                if not isinstance(value,str): raise ValueError('relative asset path required')
                path=Path(value); resolved=(root/path).resolve()
                if (path.is_absolute() or '..' in path.parts or not path.parts or
                    not resolved.is_relative_to(root) or not resolved.is_file() or resolved in seen):
                    raise ValueError('missing, escaping, duplicate or overlapping image/mask path')
                seen.add(resolved)
    meta=dict(manifest_sha256=contract.sha256_bytes(raw),classes=classes,label_map=labels,
              split_evidence=data['split_evidence'],train_count=len(data['train']),
              validation_count=len(data['validation']),official_membership_verified=False)
    return (MaskPairs(root,data['train'],mapping,train=True,profile=profile),
            MaskPairs(root,data['validation'],mapping,train=False,profile=profile),meta)


def validate_config(cfg):
    required={'task','profile','dataset','seed','device','data_root','samples','transform_profile',
              'adaptation','reader_profile','backbone','probe'}
    if set(cfg)!=required or cfg['task']!=TASK or cfg['profile']!=PROFILE:
        raise ValueError('explicit Extended semantic component config required')
    if type(cfg['seed']) is not int or cfg['seed']!=0: raise ValueError('Extended seed must be 0')
    if (os.environ.get('WORLD_SIZE','1')!='1' or
        (torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1)):
        raise ValueError('distributed Extended execution is not yet ported')
    settings=recipe(cfg['dataset'],cfg['adaptation'])
    if cfg['transform_profile'] not in TRANSFORMS: raise ValueError('explicit transform profile required')
    probe=cfg['probe']
    if not isinstance(probe,dict) or set(probe)!={'epochs','batch_size','num_workers'}:
        raise ValueError('probe requires epochs, batch_size and num_workers')
    if any(type(v) is not int or v<(0 if k=='num_workers' else 1) for k,v in probe.items()):
        raise ValueError('invalid probe integers')
    if probe['epochs']>settings['epochs']: raise ValueError('cannot exceed the scheduled epoch horizon')
    path=discover_providers().get(cfg['backbone'].get('kind'))
    expected=getattr(_load_provider(path),'EXTENDED_DENSE_READER',None) if path else None
    if expected not in (SINGLE_BLOCK,CROSS_SELF): raise ValueError('provider lacks verified Extended dense recipe')
    if cfg['reader_profile']!=(expected if cfg['adaptation']=='attentive' else None):
        raise ValueError('dense reader profile disagrees with provider')


def run(cfg,out):
    validate_config(cfg); make_deterministic(cfg['seed']); device=resolve_device(cfg['device'])
    train,val,membership=load_data(cfg['samples'],cfg['data_root'],cfg['transform_profile'])
    classes=dataset_spec(cfg['dataset'])['head_outputs']
    if len(membership['classes'])!=classes: raise ValueError('manifest ontology differs from dataset recipe')
    settings=cfg['probe']
    if len(train)<settings['batch_size']: raise ValueError('physical batch exceeds training population')
    loader=DataLoader(train,batch_size=settings['batch_size'],shuffle=True,drop_last=True,
        num_workers=settings['num_workers'],generator=torch.Generator().manual_seed(cfg['seed']))
    validation=DataLoader(val,batch_size=settings['batch_size'],num_workers=settings['num_workers'])
    model=DenseProbe(build_frozen_backbone(cfg['backbone'],device),classes,cfg['reader_profile']).to(device)
    spec=recipe(cfg['dataset'],cfg['adaptation'])
    opt,scheduler=optimizer(model,spec,batch_size=settings['batch_size'],steps_per_epoch=len(loader))
    for _ in range(settings['epochs']):
        model.train()
        for images,targets in loader:
            images,targets=images.to(device),targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss=pixel_loss(model(images,targets.shape[-2:]),targets)
            if not torch.isfinite(loss): raise ValueError('nonfinite semantic training loss')
            loss.backward(); clip_attentive_gradients(model,cfg['adaptation']); opt.step(); scheduler.step()
    model.eval(); score=SegmentationScore(classes)
    with torch.no_grad():
        for images,targets in validation:
            logits=model(images.to(device),targets.shape[-2:])
            if not torch.isfinite(logits).all(): raise ValueError('nonfinite semantic evaluation logits')
            score.update(logits.argmax(1).cpu(),targets)
    result=score.result(); result['epochs']=settings['epochs']; out=Path(out)
    contract.write_metrics(out,result,METRICS)
    torch.save({k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('backbone.')},out/'probe.pt')
    report=dict(task=TASK,profile=PROFILE,dataset=cfg['dataset'],adaptation=cfg['adaptation'],
        reader_profile=cfg['reader_profile'],transform_profile=cfg['transform_profile'],backbone=cfg['backbone'],
        evaluation_grid=[224,224],membership=membership,recipe=spec,physical_batch=settings['batch_size'],
        updates=scheduler.last_epoch,final=result,canonical_eligible=False,record_value=False,
        limitations=['single-process FP32 component','224-grid evaluation, not native-resolution benchmark',
                     'official membership and released-weight/full-score parity unverified'])
    (out/'results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv); raw=Path(args.config).read_bytes(); cfg=json.loads(raw)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    now=lambda:datetime.now(timezone.utc).isoformat()
    started=now(); error=None
    try: run(cfg,out); status='ok'
    except Exception: status='failed'; error=traceback.format_exc(limit=8)
    contract.write_manifest(out,task=TASK,method_ref=str(cfg.get('backbone',{}).get('encoder') or 'fixture'),
        status=status,config_sha256=contract.sha256_bytes(raw),started_at=started,finished_at=now(),
        seed=cfg.get('seed',0),backbone=cfg.get('backbone',{}),error=error)
    return 0 if status=='ok' else 1


if __name__=='__main__': raise SystemExit(main())
