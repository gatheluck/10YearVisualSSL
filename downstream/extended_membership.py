"""Convert explicit official classification annotations to portable sample lists.

No downloads, folder-order splits, or authenticity certification. Release counts
are required by the CLI; reduced fixture counts are an explicit Python test API.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

COUNTS = {'cub200': (5994,5794,200), 'dtd': (1880,1880,47),
          'fgvc_aircraft': (6667,3333,100), 'food101': (75750,25250,101),
          'oxford_pets': (3680,3669,37), 'ip102': (45095,22619,102),
          'mit_indoor': (5360,1340,67), 'cars': (8144,8041,196)}


def build(dataset, root, *, fixture_counts=None):
    if dataset not in COUNTS:
        raise ValueError('no verified membership parser for this dataset')
    root=Path(root).resolve()
    counts=fixture_counts if fixture_counts is not None else COUNTS[dataset]
    if len(counts)!=3 or any(type(v) is not int or v<1 for v in counts):
        raise ValueError('counts require train, evaluation and classes')
    hashes={}

    def safe(path):
        p=Path(path)
        if p.is_absolute() or '..' in p.parts or not p.parts:
            raise ValueError('unsafe annotation path')
        resolved=(root/p).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError('missing or escaping annotation/image')
        return resolved

    def lines(path):
        raw=safe(path).read_bytes(); hashes[path]=hashlib.sha256(raw).hexdigest()
        result=[x.strip() for x in raw.decode('utf-8-sig').splitlines() if x.strip() and not x.lstrip().startswith('#')]
        if not result or len(result)!=len(set(result)):
            raise ValueError('empty or duplicate annotation rows')
        return result

    def table(path):
        result={}
        for row in lines(path):
            key,value=row.split(maxsplit=1)
            if not key.isdecimal() or int(key)<1 or int(key) in result:
                raise ValueError('invalid or duplicate annotation ID')
            result[int(key)]=value
        return result

    def mat_vector(path, key):
        from scipy.io import loadmat
        raw=safe(path).read_bytes(); hashes[path]=hashlib.sha256(raw).hexdigest()
        try:
            data=loadmat(io.BytesIO(raw),simplify_cells=True)
        except Exception as exc:
            raise ValueError('unreadable MAT annotation') from exc
        if key not in data: raise ValueError('missing MAT annotation field')
        value=data[key]
        if hasattr(value,'reshape'): value=value.reshape(-1).tolist()
        return value if isinstance(value,(list,tuple)) else [value]

    rows={'train':[],'validation':[]}; unused=[]; extra={}
    if dataset=='cub200':
        classes_by_id=table('classes.txt')
        if set(classes_by_id)!=set(range(1,counts[2]+1)):
            raise ValueError('CUB class IDs do not match the release')
        classes=[classes_by_id[i] for i in range(1,counts[2]+1)]
        images=table('images.txt'); splits=table('train_test_split.txt'); labels=table('image_class_labels.txt')
        if set(images)!=set(splits) or set(images)!=set(labels):
            raise ValueError('CUB annotation image-ID join is incomplete')
        for key,path in images.items():
            label=int(labels[key])
            if (label not in classes_by_id or len(Path(path).parts)!=2 or
                Path(path).parts[0]!=classes_by_id[label] or not path.endswith('.jpg') or splits[key] not in ('0','1')):
                raise ValueError('CUB label, path or split disagreement')
            rows['train' if splits[key]=='1' else 'validation'].append(dict(path='images/'+path,target=label-1))
        inventory_root='images'
    elif dataset=='fgvc_aircraft':
        classes=lines('data/variants.txt')
        for split,filename in [('train','trainval'),('validation','test')]:
            for row in lines(f'data/images_variant_{filename}.txt'):
                key,label=row.split(maxsplit=1)
                if not key.isdecimal() or label not in classes:
                    raise ValueError('aircraft image ID or variant is invalid')
                rows[split].append(dict(path='data/images/'+key+'.jpg',target=classes.index(label)))
        inventory_root='data/images'
    elif dataset=='cars':
        classes=mat_vector('devkit/cars_meta.mat','class_names')
        if any(not isinstance(v,str) or not v.strip() for v in classes):
            raise ValueError('invalid Cars class vocabulary')
        for sp,folder,annotation in [('train','train','cars_train_annos.mat'),
                                      ('validation','val','cars_test_annos_withlabels.mat')]:
            for entry in mat_vector('devkit/'+annotation,'annotations'):
                if not isinstance(entry,dict) or not {'fname','class'}<=set(entry):
                    raise ValueError('invalid Cars annotation schema')
                name=entry['fname']; label=entry['class']
                if (not isinstance(name,str) or '/' in name or '\\' in name or
                    Path(name).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.ppm'}):
                    raise ValueError('invalid Cars image name')
                if type(label) is not int or not 1<=label<=counts[2]:
                    raise ValueError('invalid Cars class ID')
                rows[sp].append(dict(path=folder+'_original/original/'+name,target=int(label)-1))
        inventory_root=None
        extra['image_readout']='full original image; annotation bounding boxes are not applied'
    elif dataset=='food101':
        parts={sp:lines('meta/'+name+'.txt') for sp,name in [('train','train'),('validation','test')]}
        classes=sorted({p.split('/')[0] for p in parts['train']})
        for sp,paths in parts.items():
            if fixture_counts is None and set(Counter(p.split('/')[0] for p in paths).values())!={750 if sp=='train' else 250}:
                raise ValueError('Food-101 per-class split quota differs')
            for path in paths:
                bits=path.split('/')
                if len(bits)!=2 or bits[0] not in classes or '\\' in path or any(c.isspace() for c in path) or Path(path).suffix:
                    raise ValueError('Food-101 annotation path/class disagreement')
                rows[sp].append(dict(path='images/'+path+'.jpg',target=classes.index(bits[0])))
        inventory_root=None
    elif dataset=='oxford_pets':
        names={}
        for sp,filename in [('train','train'),('validation','test')]:
            for row in lines('annotations/'+filename+'.txt'):
                fields=row.split()
                if len(fields)!=4 or re.fullmatch(r'[A-Za-z_]+_[0-9]+',fields[0]) is None or not all(re.fullmatch(r'[0-9]+',v) for v in fields[1:]):
                    raise ValueError('invalid pet annotation schema')
                ident,raw,species,breed=fields; label=int(raw)-1; cls=ident.rsplit('_',1)[0]
                if not 0<=label<counts[2] or int(species) not in (1,2) or int(breed)<1:
                    raise ValueError('invalid pet label/species/breed')
                if sp=='train':
                    if names.get(label,cls)!=cls: raise ValueError('pet training label mapping conflict')
                    names[label]=cls
                if names.get(label)!=cls: raise ValueError('pet evaluation label mapping conflict')
                rows[sp].append(dict(path='images/'+ident+'.jpg',target=label))
        if set(names)!=set(range(counts[2])): raise ValueError('pet class IDs incomplete')
        classes=[names[i] for i in range(counts[2])]; inventory_root=None
        extra['training_annotation']='annotations/train.txt (captured renamed trainval list)'
    elif dataset=='ip102':
        names=table('classes.txt')
        if set(names)!=set(range(1,counts[2]+1)): raise ValueError('IP102 class IDs incomplete')
        classes=[names[i] for i in range(1,counts[2]+1)]
        parts={}
        for sp in ('train','val','test'):
            parts[sp]=[]
            for row in lines(sp+'.txt'):
                fields=row.split()
                if len(fields)!=2 or '/' in fields[0] or '\\' in fields[0] or Path(fields[0]).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff','.ppm'} or re.fullmatch(r'[0-9]+',fields[1]) is None:
                    raise ValueError('invalid IP102 list row')
                parts[sp].append((fields[0],int(fields[1])))
        labels={label for _,label in parts['train']}
        bases=[b for b in (0,1) if labels==set(range(b,counts[2]+b))]
        if len(bases)!=1: raise ValueError('ambiguous IP102 label encoding')
        base=bases[0]; extra['list_label_base']=base
        if len(parts['val'])!=(counts[1] if fixture_counts is not None else 7508):
            raise ValueError('IP102 held-out validation count differs')
        for sp,items in parts.items():
            if {label-base for _,label in items}!=set(range(counts[2])):
                raise ValueError('IP102 split class coverage differs')
            for path,label in items:
                (unused if sp=='val' else rows['train' if sp=='train' else 'validation']).append(dict(path='images/'+path,target=label-base))
        inventory_root='images'
        extra['held_out_validation']='val.txt checked and excluded from training/evaluation'
    elif dataset=='mit_indoor':
        index={}
        for folder in ('train','val'):
            directory=root/folder
            if not directory.is_dir(): raise ValueError('missing MIT storage directory')
            for p in directory.rglob('*'):
                if not p.is_file() or p.name=='.DS_Store': continue
                rel=p.relative_to(directory).as_posix()
                if len(Path(rel).parts)!=2 or p.suffix.lower()!='.jpg' or rel in index:
                    raise ValueError('ambiguous or invalid MIT storage image')
                safe(folder+'/'+rel); index[rel]=folder+'/'+rel
        classes=sorted({p.split('/')[0] for p in index}); listed=[]
        for sp,filename in [('train','TrainImages.txt'),('validation','TestImages.txt')]:
            for row in lines(filename):
                path=row.replace('\\','/')
                if path not in index: raise ValueError('missing MIT official image')
                listed.append(path)
                rows[sp].append(dict(path=index[path],target=classes.index(path.split('/')[0])))
        if set(listed)!=set(index): raise ValueError('MIT list union differs from image inventory')
        inventory_root=None
        extra['storage_membership']='official lists override physical train/val directory placement'
    else:
        parts={split:lines('labels/'+split+'1.txt') for split in ('train','val','test')}
        classes=sorted({p.split('/')[0] for p in parts['train']})
        if len(parts['val'])!=counts[1]:
            raise ValueError('DTD validation list count differs')
        for split,paths in parts.items():
            if fixture_counts is None and set(Counter(p.split('/')[0] for p in paths).values()) != {40}:
                raise ValueError('DTD partition 1 requires 40 images per class in every split')
            for path in paths:
                if len(Path(path).parts)!=2 or Path(path).parts[0] not in classes:
                    raise ValueError('DTD path/class disagreement')
                row=dict(path='images/'+path,target=classes.index(Path(path).parts[0]))
                (unused if split=='val' else rows['train' if split=='train' else 'validation']).append(row)
        inventory_root=None
    if len(classes)!=counts[2] or len(set(classes))!=len(classes):
        raise ValueError('class vocabulary differs from expected release')
    all_rows=rows['train']+rows['validation']+unused
    paths=[safe(row['path']) for row in all_rows]
    if len(paths)!=len(set(paths)):
        raise ValueError('duplicate or overlapping official split membership')
    for split,expected in zip(('train','validation'),counts[:2]):
        if len(rows[split])!=expected or {r['target'] for r in rows[split]}!=set(range(len(classes))):
            raise ValueError('split count or class coverage differs')
        rows[split].sort(key=lambda r:r['path'])
    if unused and {r['target'] for r in unused}!=set(range(len(classes))):
        raise ValueError('held-out validation class coverage differs')
    if inventory_root:
        actual={p.resolve() for p in (root/inventory_root).rglob('*') if p.is_file() and p.name!='.DS_Store'}
        if actual!=set(paths):
            raise ValueError('official membership does not match complete image inventory')
    evidence=dict(dataset=dataset,annotation_sha256=hashes,fixture=fixture_counts is not None,
                  split_counts=list(counts),authenticity_verified=False,
                  source='explicit annotation join; no prepared-folder membership inference',**extra)
    return dict(schema_version=1,classes=classes,split_evidence=json.dumps(evidence,sort_keys=True),**rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    cfg=json.loads(Path(args.config).read_text())
    if set(cfg)!={'dataset','data_root'}:
        raise ValueError('config requires dataset and data_root only')
    data=build(cfg['dataset'],cfg['data_root'])
    with Path(args.out).open('x') as handle:
        handle.write(json.dumps(data,indent=2,sort_keys=True)+'\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
