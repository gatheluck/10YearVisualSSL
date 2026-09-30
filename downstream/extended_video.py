"""Explicit Extended HMDB51/UCF101 video LP/AP components, not score certification."""
import argparse
import json
from pathlib import Path
import random
import sys

from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms as T

from downstream import extended_classification as image
from downstream import extended_distributed as distributed
from downstream.spatial_backbones import build_frozen_backbone

TASK='extended_video_classification'
PROFILE=image.PROFILE
METRICS={'top1':'extended_video_top1','top5':'extended_video_top5','videos':None,'epochs':'epochs_completed'}


def recipe(dataset,adaptation):
    if dataset not in ('hmdb51','ucf101') or adaptation not in ('frozen','attentive'):
        raise ValueError('verified Extended video classification requires HMDB51/UCF101 LP/AP')
    parent=json.loads((image.PROTOCOLS/'EXTEND_LINEAR_v1.json').read_text())
    entry=parent['datasets'][dataset]
    if entry['recipe']!='video_cls_50' or entry['primary_metric']!='top-1 accuracy':
        raise ValueError('video dataset recipe/metric identity changed')
    if adaptation=='attentive':
        parent=json.loads((image.PROTOCOLS/'EXTEND_ATTENTIVE_v1.json').read_text())
        result=dict(parent['recipe_overrides']['video_cls_50']['optimization'])
        # Captured AdamW constructor defaults, absent from the protocol JSON.
        result['betas']=[.9,.999]
    else:
        result=dict(parent['recipes']['video_cls_50']['optimization'])
        if result['warmup']!='5 epochs linear from 1e-6':
            raise ValueError('unresolved video warmup')
        result['warmup']='5 epochs from 1e-6'
    return result


optimizer=image.optimizer


class Classifier(image.Classifier):
    """Preserve native clips; image encoders retain ordered frame means for AP."""
    def forward(self,clips):
        if clips.ndim!=5 or clips.shape[1:3]!=(16,3) or min(clips.shape[-2:])<1:
            raise ValueError('Extended video requires [B,16,3,H,W]')
        with torch.no_grad():
            if self.reader is None and callable(getattr(self.backbone,'classification_features',None)):
                features=self.backbone.classification_features(clips,adaptation='frozen',video=True).float()
            elif callable(getattr(self.backbone,'video_tokens',None)):
                tokens=self.backbone.video_tokens(clips).float()
                features=tokens if self.reader is not None else F.normalize(tokens.mean(1),dim=-1)
            elif self.reader is not None:
                b,t=clips.shape[:2]
                features=self.backbone.forward_features(clips.flatten(0,1)).float().mean((2,3)).reshape(b,t,-1)
            else:
                raise ValueError('provider lacks a verified video global readout')
        return self.head(self.reader(features) if self.reader is not None else features)


def segment_indices(count,*,train):
    if type(count) is not int or count<1:
        raise ValueError('video must contain frames')
    bounds=[round(i*count/16) for i in range(17)]
    result=[]
    for i in range(16):
        a=min(bounds[i],count-1); b=min(max(bounds[i+1],bounds[i]+1),count)
        result.append(random.randint(a,max(a,b-1)) if train else (a+b-1)//2)
    return result


def transform_clip(frames,*,train):
    if len(frames)!=16 or len({f.size for f in frames})!=1:
        raise ValueError('clip needs 16 equally sized RGB frames')
    if train:
        top,left,h,w=T.RandomResizedCrop.get_params(frames[0],scale=(.08,1.),ratio=(.75,4/3))
        flip=random.random()<.5
    result=[]
    for frame in frames:
        frame=frame.convert('RGB')
        if train:
            frame=TF.resized_crop(frame,top,left,h,w,(224,224),interpolation=TF.InterpolationMode.BICUBIC)
            if flip: frame=TF.hflip(frame)
        else:
            frame=TF.center_crop(TF.resize(frame,256,interpolation=TF.InterpolationMode.BICUBIC),224)
        result.append(TF.normalize(TF.to_tensor(frame),(.485,.456,.406),(.229,.224,.225)))
    return torch.stack(result)


class Videos(Dataset):
    def __init__(self,root,rows,*,train,profile):
        if profile!='captured_video16_v1': raise ValueError('explicit captured video input profile required')
        self.root,self.rows,self.train=Path(root).resolve(),rows,train
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]; path=self.root/row['path']
        if path.is_dir():
            files=sorted(p for p in path.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
            if any(not p.resolve().is_relative_to(self.root) or not p.is_file() for p in files):
                raise ValueError('frame path escapes data root or is not a file')
            frames=[]
            for i in segment_indices(len(files),train=self.train):
                with Image.open(files[i]) as im: frames.append(im.convert('RGB'))
        else:
            import decord
            reader=decord.VideoReader(str(path),num_threads=1)
            frames=[Image.fromarray(a) for a in reader.get_batch(segment_indices(len(reader),train=self.train)).asnumpy()]
        return transform_clip(frames,train=self.train),row['target']


def load_data(manifest,root,profile):
    train,val,metadata=image.load_data(manifest,root,profile,dataset_factory=Videos,allow_directories=True)
    metadata['media_policy']='decord files or explicitly prepared ordered frame directories; no decode fallback'
    return train,val,metadata


def score(logits,targets):
    result=image.score(logits,targets);result['videos']=result.pop('images');return result


def validate_config(cfg):
    if cfg.get('transform_profile')!='captured_video16_v1':
        raise ValueError('explicit captured video input profile required')
    return image.validate_config(cfg,task=TASK,get_recipe=recipe,reader_attribute='EXTENDED_VIDEO_READER',max_epochs=50)


def run(cfg,out):
    with distributed.session(cfg['device']) as context:
        return image._run(cfg,out,context,api=sys.modules[__name__])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    return distributed.cli(Path(args.config).read_bytes(),args.out,run,TASK)


if __name__=='__main__': raise SystemExit(main())
