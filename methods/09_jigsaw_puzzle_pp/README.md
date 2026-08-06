# 09_jigsaw_puzzle_pp — step 1 (VGG16 Jigsaw++ pretext) + linear evaluation

Noroozi, Vinjimoor, Favaro & Pirsiavash, *Boosting Self-Supervised Learning via
Knowledge Transfer*, CVPR 2018
([arXiv:1805.00385](https://arxiv.org/abs/1805.00385)).

An image is cut into a 3×3 grid of tiles; up to two tiles may be replaced with
tiles from another image (**occlusions**), 70% of images are converted to
grayscale, the tiles are permuted by one of a fixed high-Hamming-distance
permutation set (701 in the paper), and a **shared VGG16 encoder** predicts which
permutation was applied. Step 1 is that pretext.

## Scope — the VGG16 pretext only

This port covers **stage (a)** of the paper: the VGG16 Jigsaw++ pretext. The
paper's headline **knowledge-transfer** stages — extract the VGG conv4 features,
k-means cluster them (k=2000) into pseudo-labels, then train an AlexNet to
classify those pseudo-labels — are a **faiss-GPU pipeline** (the capture's
`cluster_and_pseudolabels.py` *explicitly refuses* the CPU/scikit-learn fallback,
so faiss-GPU is mandatory). That is Group-3 / faiss work and is **deferred**
alongside deepcluster; see `docs/PORTING_ROADMAP.md`. The capture's step 2 (a ViT
variant) is excluded, as in every port, which also drops its `timm` dependency.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/9_jigsaw_puzzle_pp` (the lab's own VGG16 model + jigsaw++ dataset,
torch/torchvision only) — no `third_party/` submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP and logs to
TensorBoard; none is needed for a single-process run, so
`train_step1_jigsaw_pp.py` owns a thin fp32 loop, the device is **resolved**
rather than assumed CUDA, and TensorBoard is dropped.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the shared **VGG16 encoder** (`encoder.*`) — four VGG conv
blocks, an adaptive max-pool to 4×4×512, and an FC layer, giving one 1024-d
feature per image. The permutation classifier is pretext machinery and is
excluded, and the round trip (write it, load it back into a rebuilt model,
compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. (The paper's
own downstream eval probes the AlexNet cluster-classification network from the
deferred faiss stages; this port probes the VGG16 pretext encoder instead.)
Images are resized to the encoder's tile size; the probe follows the lab's ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small VGG16, a few fabricated
  images, a 4-permutation puzzle with grayscale and occlusions — runs through
  `python -m adapter` on a CPU, passes `contract-test`, and the encoder
  round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/step1.yaml` is the VGG16 recipe (701
  permutations, 90 epochs), a recipe, not a completed run.
- **Not ported:** the faiss-GPU knowledge-transfer stages (deferred, Group 3).
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/09_jigsaw_puzzle_pp-step1-device.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML — the self-contained methods'
stack, no submodule and no extra. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `05_jigsaw_puzzle`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/09_jigsaw_puzzle_pp/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py methods/09_jigsaw_puzzle_pp/configs/step1.yaml \
        --set DATA_ROOT=/path/to/images > resolved.json
    cd methods/09_jigsaw_puzzle_pp && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py methods/09_jigsaw_puzzle_pp/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt > eval.json
    cd methods/09_jigsaw_puzzle_pp && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
