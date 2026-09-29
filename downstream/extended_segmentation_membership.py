"""Explicit native semantic membership conversion; no downloads or inferred splits."""
import argparse
import hashlib
import json
from pathlib import Path

COUNTS={'ade20k':(20210,2000,150),'pascal_voc2012':(1464,1449,21),'bdd100k':(7000,1000,19)}


def build(dataset,root,*,fixture_counts=None):
    if dataset not in COUNTS: raise ValueError('native semantic membership not ported for this dataset')
    root=Path(root).resolve(); ntrain,nval,classes=COUNTS[dataset]
    counts=fixture_counts if fixture_counts is not None else (ntrain,nval)
    if len(counts)!=2 or any(type(v) is not int or v<1 for v in counts): raise ValueError('invalid split counts')
    rows={}; hashes={}; seen=set()

    def asset(path):
        resolved=path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file(): raise ValueError('missing or escaping asset')
        if resolved in seen: raise ValueError('duplicate or overlapping image/mask asset')
        seen.add(resolved)
        return path.relative_to(root).as_posix()

    def index(folder,suffix,*,strip_mask=False):
        found={}
        for path in sorted(folder.glob('*'+suffix)):
            name=path.stem
            if strip_mask:
                if 'color' in name.lower(): raise ValueError('color labels cannot supply train IDs')
                name=name.removesuffix('_train_id')
            if name in found: raise ValueError('duplicate native sample ID')
            found[name]=path
        return found

    for split,expected in zip(('train','validation'),counts):
        if dataset=='pascal_voc2012':
            lists=root/'ImageSets'/'Segmentation'
            if (lists/'trainaug.txt').exists():
                raise ValueError('trainaug exists: augmented membership needs an explicit separately verified manifest')
            path=lists/('train.txt' if split=='train' else 'val.txt')
            if not path.resolve().is_relative_to(root): raise ValueError('escaping split annotation')
            raw=path.read_bytes(); hashes[path.relative_to(root).as_posix()]=hashlib.sha256(raw).hexdigest()
            ids=[line.strip() for line in raw.decode().splitlines() if line.strip()]
            if len(set(ids))!=len(ids) or any('/' in v or '\\' in v or v in ('.','..') or len(v.split())!=1 for v in ids):
                raise ValueError('duplicate or invalid native split ID')
            images={v:root/'JPEGImages'/(v+'.jpg') for v in ids}
            masks={v:root/'SegmentationClass'/(v+'.png') for v in ids}
        else:
            tag=('training' if split=='train' else 'validation') if dataset=='ade20k' else ('train' if split=='train' else 'val')
            images=index(root/'images'/tag,'.jpg')
            masks=index(root/('annotations' if dataset=='ade20k' else 'labels')/tag,'.png',strip_mask=dataset=='bdd100k')
        if set(images)!=set(masks): raise ValueError('orphan image/mask: native pairing must be bijective')
        if len(images)!=expected: raise ValueError('native split count differs from release')
        rows[split]=[dict(image=asset(images[key]),mask=asset(masks[key])) for key in sorted(images)]
    label_map={str(i):i for i in range(classes)}
    if dataset=='ade20k': label_map={'0':255,**{str(i+1):i for i in range(classes)}}
    else: label_map['255']=255
    evidence=dict(dataset=dataset,split_counts=list(counts),annotation_sha256=hashes,
        paired_inventory_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        fixture=fixture_counts is not None,authenticity_verified=False,mask_values_verified=False,
        source='explicit native split lists or paired release directories; no random split')
    return dict(schema_version=1,classes=[str(i) for i in range(classes)],label_map=label_map,
                split_evidence=json.dumps(evidence,sort_keys=True),**rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv); cfg=json.loads(Path(args.config).read_text())
    if set(cfg)!={'dataset','data_root'}: raise ValueError('config requires dataset and data_root only')
    data=build(cfg['dataset'],cfg['data_root'])
    with Path(args.out).open('x') as handle: handle.write(json.dumps(data,indent=2,sort_keys=True)+'\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
