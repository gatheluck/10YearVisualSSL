"""Read HMDB51/UCF101 split-1 annotations without folder fallback or downloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

COUNTS={'hmdb51':(3570,1530,51),'ucf101':(9537,3783,101)}


def build(dataset,root,*,split_index=1,fixture_counts=None):
    if dataset not in COUNTS or type(split_index) is not int or split_index!=1:
        raise ValueError('paper video membership requires HMDB51/UCF101 split 1')
    root=Path(root).resolve();counts=COUNTS[dataset] if fixture_counts is None else fixture_counts
    if len(counts)!=3 or any(type(v) is not int or v<1 for v in counts):
        raise ValueError('positive train, test and class counts required')
    hashes={};rows={'train':[],'validation':[]};seen=set();groups={};unused=0
    def safe(name):
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not p.parts:
            raise ValueError('unsafe relative annotation path')
        source=root/p
        if any(part.is_symlink() for part in (source,*source.parents) if part!=root and part.is_relative_to(root)):
            raise ValueError('symlink annotation/video is not an immutable source')
        p=(root/p).resolve()
        if not p.is_relative_to(root) or not p.is_file():raise ValueError('missing or escaping video/annotation')
        return p
    def lines(name):
        raw=safe(name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
        return [s.strip() for s in raw.decode().splitlines() if s.strip()]
    def add(identity,target,split,group):
        asset='videos/'+identity;real=safe(asset)
        if real in seen:raise ValueError('duplicate or overlapping clip membership')
        seen.add(real)
        if group in groups and groups[group]!=split:raise ValueError('video group crosses splits')
        groups[group]=split;rows[split].append(dict(path=asset,target=target))
    if dataset=='hmdb51':
        suffix='_test_split1.txt';names=sorted(p.name for p in (root/'hmdb51_splits').glob('*'+suffix))
        classes=[n[:-len(suffix)] for n in names];ids=set();tags=Counter()
        for i,(cls,name) in enumerate(zip(classes,names)):
            for line in lines('hmdb51_splits/'+name):
                fields=line.rsplit(None,1)
                if len(fields)!=2:raise ValueError('invalid HMDB split row')
                clip,tag=fields
                if tag not in ('0','1','2') or not clip.endswith('.avi') or '/' in clip or '\\' in clip:
                    raise ValueError('invalid HMDB clip or split tag')
                identity=cls+'/'+clip
                if identity in ids:raise ValueError('duplicate HMDB clip')
                ids.add(identity);tags[tag]+=1
                if tag=='0':unused+=1;continue
                add(identity,i,'train' if tag=='1' else 'validation',identity)
        if fixture_counts is None and unused!=1666:raise ValueError('HMDB unused split population differs')
    else:
        mapping={}
        for line in lines('ucfTrainTestlist/classInd.txt'):
            parts=line.split()
            if len(parts)!=2:raise ValueError('invalid UCF vocabulary row')
            number,name=parts;index=int(number)-1
            if name in mapping or index in mapping.values() or '/' in name or '\\' in name:
                raise ValueError('duplicate or invalid UCF class')
            mapping[name]=index
        if set(mapping.values())!=set(range(counts[2])):raise ValueError('UCF class indices must be complete')
        classes=[k for k,v in sorted(mapping.items(),key=lambda p:p[1])]
        for split,name in [('train','trainlist01.txt'),('validation','testlist01.txt')]:
            for line in lines('ucfTrainTestlist/'+name):
                parts=line.split()
                if len(parts)!=(2 if split=='train' else 1):raise ValueError('invalid UCF split row')
                identity=parts[0];pieces=identity.split('/')
                if len(pieces)!=2 or pieces[0] not in mapping:raise ValueError('unknown UCF class/path')
                target=mapping[pieces[0]]
                if split=='train' and int(parts[1])-1!=target:raise ValueError('UCF label mapping conflict')
                match=re.fullmatch(r'(.+_g\d+)_c\d+\.avi',identity)
                if match is None:raise ValueError('invalid UCF video group')
                add(identity,target,split,match[1])
    if len(classes)!=counts[2] or len(set(classes))!=len(classes):raise ValueError('class count differs from release')
    for split,n in zip(('train','validation'),counts[:2]):
        if len(rows[split])!=n or {r['target'] for r in rows[split]}!=set(range(len(classes))):
            raise ValueError('video split count or class coverage differs')
        rows[split].sort(key=lambda r:r['path'])
    evidence=dict(dataset=dataset,split_index=1,unused_clips=unused,annotation_sha256=hashes,
                  fixture=fixture_counts is not None,authenticity_verified=False,decoder_verified=False,
                  metric='single-split top1; not three-split average')
    return dict(schema_version=1,classes=classes,split_evidence=json.dumps(evidence,sort_keys=True),**rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);cfg=json.loads(Path(args.config).read_text())
    if set(cfg)!={'dataset','data_root'}:raise ValueError('config requires dataset and data_root only')
    result=build(cfg['dataset'],cfg['data_root'])
    with Path(args.out).open('x') as handle:handle.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
