"""Stage native MNIST IDX as lossless images for the Extended LP/AP sample contract.

The four raw files are read only. Output must be a new directory outside the
source tree. A completed samples.json is written last; failed exports are never
resumed or overwritten. No downloads or membership/transform inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

COUNTS = (60000,10000,10)


def convert(root, out, *, fixture_counts=None):
    from PIL import Image
    root=Path(root).resolve(); out=Path(out).absolute()
    if '..' in out.parts:
        raise ValueError('output must not contain parent traversal')
    if out.resolve().is_relative_to(root):
        raise ValueError('output must be outside the read-only source tree')
    counts=COUNTS if fixture_counts is None else fixture_counts
    if len(counts)!=3 or any(type(v) is not int or v<1 for v in counts) or counts[2]>10:
        raise ValueError('counts require positive train/evaluation sizes and at most ten classes')
    hashes={}; payloads={}; rows={}
    def read(name):
        path=(root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('escaping IDX input link')
        raw=path.read_bytes(); hashes[name]=hashlib.sha256(raw).hexdigest()
        return raw
    for split,prefix,count in [('train','train',counts[0]),('validation','t10k',counts[1])]:
        images=read(f'MNIST/raw/{prefix}-images-idx3-ubyte')
        labels=read(f'MNIST/raw/{prefix}-labels-idx1-ubyte')
        if len(images)<16 or len(labels)<8: raise ValueError('truncated IDX header')
        magic,n,height,width=struct.unpack('>IIII',images[:16])
        label_magic,nlabels=struct.unpack('>II',labels[:8])
        if magic!=2051 or label_magic!=2049: raise ValueError('invalid IDX magic')
        if (height,width)!=(28,28): raise ValueError('MNIST requires 28 by 28 pixels')
        if n!=nlabels or n!=count: raise ValueError('IDX split count or image/label join differs')
        if len(images)!=16+n*784 or len(labels)!=8+n: raise ValueError('IDX truncated or trailing payload')
        if set(labels[8:])!=set(range(counts[2])): raise ValueError('IDX targets or class coverage differ')
        payloads[split]=images[16:]
        rows[split]=[dict(path=f'images/{prefix}/{i:05d}.png',target=label) for i,label in enumerate(labels[8:])]
    evidence=dict(dataset='mnist',annotation_sha256=hashes,fixture=fixture_counts is not None,
                  split_counts=list(counts),authenticity_verified=False,
                  source='official IDX split and row index; lossless grayscale PNG staging')
    data=dict(schema_version=1,classes=list(map(str,range(counts[2]))),
              split_evidence=json.dumps(evidence,sort_keys=True),**rows)
    # Validate both populations before creating any output. Existing files/links
    # or partial directories are deliberately refused even on an identical rerun.
    out.mkdir(parents=True,exist_ok=False)
    for split,items in rows.items():
        for i,row in enumerate(items):
            target=out/row['path']; target.parent.mkdir(parents=True,exist_ok=True)
            with Image.frombytes('L',(28,28),payloads[split][i*784:(i+1)*784]) as image:
                with target.open('xb') as handle: image.save(handle,format='PNG')
    pending=out/'.samples.json.tmp'
    with pending.open('x') as handle:
        handle.write(json.dumps(data,indent=2,sort_keys=True)+'\n')
    pending.replace(out/'samples.json')
    return data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    cfg=json.loads(Path(args.config).read_text())
    if set(cfg)!={'data_root'}: raise ValueError('config requires data_root only')
    convert(cfg['data_root'],args.out)
    return 0


if __name__=='__main__': raise SystemExit(main())
