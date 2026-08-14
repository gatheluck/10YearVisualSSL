# 07_deepcluster — step 1 (DeepCluster pretext) + linear evaluation

Caron, Bojanowski, Joulin & Douze, *Deep Clustering for Unsupervised Learning of
Visual Features*, ECCV 2018 ([arXiv:1807.05520](https://arxiv.org/abs/1807.05520)).

Each epoch, the whole training set's **fc7** features are extracted, reduced by
**PCA + whitening** and clustered by **k-means**; the cluster assignments become
**pseudo-labels** that a reset classification head is trained to predict. The
backbone is an **AlexNet-BN** with a fixed 2-channel **Sobel** front-end. Step 1
is that pretext.

## Scope — the AlexNet-BN path and the unified ViT-B/16 Step 2

This port covers the paper-faithful **AlexNet-BN** step 1 (`configs/pretrain.yaml`,
Sobel + SGD) **and** the capture's unified **ViT-B/16 Step 2**
(`configs/pretrain_vit.yaml`, `arch: vit`): the same ViT-B/16 backbone every method
shares plus a reset-each-epoch linear head, self-labelled by the **same per-epoch
faiss k-means** (CLS features, PCA-whitened, k clusters), cross-entropy on the
pseudo-labels. Optimiser AdamW (betas 0.9, 0.999) with warmup + cosine and AMP on
CUDA; checkpoints at 100/200/300 epochs, each probed by the same frozen-backbone
`linear_eval` (the CLS embedding, `embed_dim`-d). The ViT path needs `timm`
(imported lazily); because it reuses the faiss clustering it is **GPU /
x86_64-linux only**, the same as the native path. The native AlexNet-BN path is
byte-for-byte unchanged.

## Dependency decision — faiss (GPU / x86_64-linux only)

The clustering uses **faiss**, the paper-target backend the capture and the
original DeepCluster repo use. The capture ships faiss as *required*
(`require_faiss: true`; "Paper-target runs must not use the slow sklearn
fallback") and marks its sklearn fallback "not the official DeepCluster
protocol". For faithful reproducibility this port commits to **faiss** (dropping
the fallback) rather than reimplementing k-means.

`faiss-gpu==1.14.3` (the capture's pinned version) ships **only a linux-x86_64
wheel**, so — unlike every other ported method — **07_deepcluster is GPU /
x86_64-linux only**. faiss lives in `requirements.lock.cu130.txt`; the
cross-platform `requirements.lock.txt` carries only the torch (and timm) parts and
is **not, by itself, a runnable pretrain environment**. (Verified: `faiss-gpu==1.14.3`
installs on py3.12 and coexists with torch 2.13.0+cu130 on an NVIDIA Tesla T4,
both using the GPU in one process.) The unified ViT-B/16 Step 2 (`arch: vit`) reuses
this same faiss clustering, so it is GPU / x86_64-linux only too.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **AlexNet-BN backbone** (`features.*` + `classifier.*`, i.e.
conv1–5 + fc6/fc7). The reset-each-epoch `top_layer` (the k-way pseudo-label head)
and the fixed Sobel front-end are excluded (the Sobel filter is deterministic and
rebuilt on load). The round trip (write it, load it back into a rebuilt model,
compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
backbone's **4096-d fc7** feature. The probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule) — the same probe the other ported
methods use, so the number is comparable across them.

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — k=4 clusters, pca_dim=4, a 64px
  crop, one epoch, a few fabricated images — runs the full extract→cluster→train
  loop through `python -m adapter` (with faiss) on a CPU, passes `contract-test`,
  and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper-target recipe (k=10000,
  pca_dim=256, 224px, 500 epochs), a recipe, not a completed run.
- **Exercised (ViT Step 2):** a hermetic smoke — a tiny ViT (16-d embed, one
  block), k=4, pca_dim=4, two epochs with `save_at_epochs: [1, 2]` — runs the
  same extract→faiss-cluster→train loop through `python -m adapter` (with faiss
  and timm) on a CPU, writes `encoder.pt` and both `encoder_epoch{1,2}.pt`
  milestones, and a milestone probe passes `contract-test`. The full 300-epoch
  ViT-B/16 recipe (k=1000, pca_dim=128) has not been run here.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/07_deepcluster-pretrain-device.json`).

## Environment (GPU / x86_64-linux)

torch / torchvision / numpy / PyYAML / **timm** (the ViT Step-2 backbone, lazy)
**+ faiss-gpu** (CUDA lock only). `requirements.lock.cu130.txt` is the runnable GPU
closure (adds `faiss-gpu` and the nvidia CUDA wheels). `requirements.lock.txt` is
the torch + timm CPU closure (no faiss — faiss-gpu has no cross-platform wheel).

    uv pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        -r methods/07_deepcluster/requirements.lock.cu130.txt \
        -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images (needs a GPU + faiss)
    python bin/resolve-config.py --config methods/07_deepcluster/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/07_deepcluster && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/07_deepcluster/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/07_deepcluster && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
