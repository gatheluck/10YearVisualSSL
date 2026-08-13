# 09_jigsaw_puzzle_pp — VGG16 Jigsaw++ pretext (+ unified ViT-B/16 Step 2) + knowledge transfer + linear evaluation

Noroozi, Vinjimoor, Favaro & Pirsiavash, *Boosting Self-Supervised Learning via
Knowledge Transfer*, CVPR 2018
([arXiv:1805.00385](https://arxiv.org/abs/1805.00385)).

An image is cut into a 3×3 grid of tiles; up to two tiles may be replaced with
tiles from another image (**occlusions**), 70% of images are converted to
grayscale, the tiles are permuted by one of a fixed high-Hamming-distance
permutation set (701 in the paper), and a **shared VGG16 encoder** predicts which
permutation was applied. Step 1 is that pretext.

## Scope — the pretext, the knowledge transfer, and the probe

This port covers the paper's two headline stages:

- **pretrain** — the VGG16 Jigsaw++ **pretext** (stage a): the shared VGG16 encoder
  predicts which permutation reordered the tiles.
- **knowledge_transfer** — the paper's namesake (capture stages b + d): extract
  the VGG16 **conv4** features for every image, **k-means** cluster them into
  pseudo-labels, then train a **standard AlexNet** to classify the pseudo-labels.
  The clustering uses **faiss** — the paper-target backend, as in the capture,
  whose `cluster_and_pseudolabels.py` *explicitly refuses* a CPU/scikit-learn
  fallback ("fallback cluster assignments are inconsistent"). faiss-gpu ships only
  a linux-x86_64 wheel, so **this stage is GPU / x86_64-linux only** (faiss lives
  in the CUDA lock, marked `# gpu-only`; see `07_deepcluster`, which shares the
  mechanism).

The capture's **Step 2** (unified from-scratch **ViT-B/16**) is ported here as part
of the Step-2 fan-out (`configs/pretrain_vit.yaml`, `models/vit_jigsaw_pp.py`,
`train_pretrain_vit_jigsaw_pp.py`, `data/jigsaw_pp_vit_dataset.py`): the 9 processed
tiles are reassembled into one 224x224 image and the ViT's CLS token feeds the
701-way permutation head. Selected by `arch: vit`; the native VGG16 pretrain and the
knowledge_transfer stage are unchanged (only the ViT path needs `timm`). See
docs/STEP2_VIT_PORTING.md.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/9_jigsaw_puzzle_pp` (the lab's own VGG16 model + jigsaw++ dataset, and
the standard AlexNet for the knowledge transfer, torch/torchvision + faiss only)
— no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP and logs to
TensorBoard, and runs the clustering and the AlexNet training as two separate DDP
scripts; none of that is needed for a single-process run. So
`train_pretrain_jigsaw_pp.py` and `train_pretrain_cluster_cls.py` own thin fp32 loops,
the clustering and AlexNet training happen in one stage, the device is
**resolved** rather than assumed CUDA, and TensorBoard is dropped.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is:

- for **pretrain**, the shared **VGG16 encoder** (`encoder.*`) — four VGG conv
  blocks, an adaptive max-pool to 4×4×512, and an FC layer, giving one 1024-d
  feature per image;
- for **knowledge_transfer**, the **AlexNet conv trunk** (`features.*`), whose
  `get_encoder()` (features + avgpool) gives one 9216-d feature per image.

The classification head (the permutation classifier, or the pseudo-label head) is
training machinery and is excluded; the round trip (write it, load it back into a
rebuilt model, compare the weights) is tested for both.

`linear_eval` reads an `encoder.pt` and probes it: `arch=vgg16` (the default)
probes the VGG16 pretext encoder, and `arch=alexnet_cluster_cls` probes the
knowledge-transfer AlexNet — the paper's own downstream target. Both are models
this port trains, so either probe number is a genuine, comparable linear probe.
Images are resized to the encoder's native size (the VGG16 tile size, or the
AlexNet image size); the probe follows the lab's ARSSL protocol (features cached
once, mean-centred and L2-normalised, a single linear layer trained with SGD under
a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small VGG16, a few fabricated
  images, a 4-permutation puzzle with grayscale and occlusions — runs through
  `python -m adapter` on a CPU, passes `contract-test`, and the encoder
  round-trip and a determinism check pass.
- **Exercised (knowledge_transfer):** a hermetic smoke clusters a pretrain VGG16's
  conv4 features (faiss k-means, k=4) into pseudo-labels and trains a small
  AlexNet through `python -m adapter`, passes `contract-test`, and the AlexNet
  `encoder.pt` round-trips. This stage needs faiss, so the smoke is skipped where
  faiss is absent (non-x86_64-linux / no GPU wheel).
- **Exercised (linear_eval):** hermetic smokes fit the probe on a pretrain VGG16
  encoder and on a knowledge-transfer AlexNet over a two-class ImageFolder, pass
  `contract-test`, write the comparable `linear_probe` accuracies, and write
  **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` (701 permutations, 90 epochs) and
  `configs/knowledge_transfer.yaml` (k=2000, 90 epochs) are recipes, not
  completed runs.
- **GPU:** the device resolution and the knowledge-transfer guards are verified on
  real hardware; see the mutation specs
  (`mutations/09_jigsaw_puzzle_pp-pretrain-device.json`,
  `mutations/09_jigsaw_puzzle_pp-knowledge-transfer.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML — the self-contained methods' stack
— plus **faiss** for the knowledge-transfer clustering. Because faiss-gpu is
linux-x86_64-only, it is **not** in the cross-platform `requirements.lock.txt`
(CPU); it lives only in `requirements.lock.cu130.txt` (CUDA 13.0), via the
`# gpu-only` marker in `requirements.txt`. step 1 and the default `arch=vgg16`
probe run from the CPU lock; the `knowledge_transfer` stage needs the CUDA lock.

    # CPU (step 1 + arch=vgg16 probe; no faiss)
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/09_jigsaw_puzzle_pp/requirements.lock.txt -r requirements-tools.lock.txt

    # CUDA 13.0 (adds faiss for the knowledge_transfer stage)
    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        -r methods/09_jigsaw_puzzle_pp/requirements.lock.cu130.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/09_jigsaw_puzzle_pp/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/09_jigsaw_puzzle_pp && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # knowledge transfer (faiss / GPU / x86_64-linux): ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/09_jigsaw_puzzle_pp/configs/knowledge_transfer.yaml \
        --set DATA_ROOT=/path/to/images \
        --set ENCODER=/path/to/s1/encoder.pt --out kt.json
    cd methods/09_jigsaw_puzzle_pp && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/kt.json --out /path/to/kt

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is a step 1 or KT encoder.pt.
    # configs/linear_eval.yaml probes the VGG16 (arch=vgg16);
    # configs/linear_eval_cluster_cls.yaml probes the AlexNet (arch=alexnet_cluster_cls).
    python bin/resolve-config.py --config methods/09_jigsaw_puzzle_pp/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/09_jigsaw_puzzle_pp && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
