"""Convert explicit official classification annotations to portable sample lists.

No downloads, folder-order splits, or authenticity certification. Release counts
are required by the CLI; reduced fixture counts are an explicit Python test API.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

COUNTS = {'cub200': (5994,5794,200), 'dtd': (1880,1880,47),
          'fgvc_aircraft': (6667,3333,100)}


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

    rows={'train':[],'validation':[]}; unused=[]
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
                  source='explicit annotation join; no prepared-folder membership inference')
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
