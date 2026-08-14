# 10_inst_disc — step 1 (ResNet-50 + NCE memory bank) + linear evaluation

Wu, Xiong, Yu & Lin, *Unsupervised Feature Learning via Non-Parametric
Instance-level Discrimination*, CVPR 2018
([arXiv:1805.01978](https://arxiv.org/abs/1805.01978)).

Every image is treated as its own class. A ResNet-50 maps each image to a 128-d
L2-normalised embedding, and a **non-parametric NCE loss** over a momentum
**memory bank** (one row per training instance) pulls an image's embedding toward
its own bank row and away from `m` random negatives. Step 1 is that pretext.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/10_inst_disc` (the lab's own ResNet-50 model, NCE memory-bank loss and
dataset, torch/torchvision only) — no `third_party/` submodule. The capture's
unified Step 2 (`configs/pretrain_vit.yaml`, `arch: vit`) is also ported
additively: the same ViT-B/16 backbone every method shares, its CLS token through
`Linear(768, 128)`, trained from scratch under the same NCE memory-bank objective
(AdamW + warmup/cosine, no AMP/clip; checkpoints at 100/200/300, each probed by
the same frozen-backbone `linear_eval`, whose head sizes to the CLS feature). The
ViT path needs `timm` (imported lazily); the native ResNet-50 path is
byte-for-byte unchanged.

The lab wrapper trains under `DistributedDataParallel` and logs to TensorBoard;
neither is needed for a single-process run, so `train_pretrain_instdisc.py` owns a
thin loop, the device is **resolved** rather than assumed CUDA, and TensorBoard
is dropped. The NCE loss keeps its memory bank and momentum update; only the
multi-GPU all-reduce / all-gather branches are dropped.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **ResNet-50 backbone** (`encoder.*`) — up to the global
average pool, giving one 2048-d feature per image. The 128-d projection head
(`fc.*`) and the memory bank are instance-discrimination machinery and are
excluded (the memory bank lives in the NCE loss, not the model, so it is never in
the model's `state_dict`). The round trip (write it, load it back into a rebuilt
model, compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. It probes the
**backbone** (2048-d), the standard SSL convention, not the projection head.
Images are ImageNet-normalised and resized to the training size; the probe
follows the lab's ARSSL protocol (features cached once, mean-centred and
L2-normalised, a single linear layer trained with SGD under a cosine schedule).

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a ResNet-50, a few fabricated
  images, a 4-negative NCE memory bank — runs through `python -m adapter` on a
  CPU, passes `contract-test`, and the encoder round-trip and a determinism check
  pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the paper recipe (128-d embedding,
  4096 negatives, 200 epochs), a recipe, not a completed run.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/10_inst_disc-pretrain-device.json`).

## Environment

torch / torchvision / numpy / PyYAML — the self-contained methods' stack, no
submodule — plus `timm` for the unified ViT-B/16 Step-2 path (imported lazily, so
the native ResNet-50 path never needs it). `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
closure as `14_simclrv1`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/10_inst_disc/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is an ImageFolder of training images
    python bin/resolve-config.py --config methods/10_inst_disc/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/10_inst_disc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/10_inst_disc/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/10_inst_disc && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
