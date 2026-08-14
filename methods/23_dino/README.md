# 23_dino — step 1 (DINO ViT-S/16 pretext) + linear evaluation

Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO),
2021 ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)).

DINO is **self-distillation with no labels**. A **student** (a Vision Transformer
backbone + a 3-layer MLP head) sees all crops; a **teacher** (an
exponential-moving-average copy of the student, no gradient) sees only the two
global crops. The loss is the **cross-entropy** between the teacher's *centred +
sharpened* output and the student's *sharpened* output, averaged over the crop
pairs where the views differ. An online **centre** (EMA) prevents collapse; the
teacher momentum follows a cosine schedule 0.996 → 1.0. Step 1 is that pretext.

## Scope — the ViT-S/16 step 1 and the unified ViT-B/16 Step 2

This port covers DINO's **ViT-S/16** step 1 (`configs/pretrain.yaml`, the paper
recipe) **and** the capture's unified **ViT-B/16 Step 2** (`configs/pretrain_vit.yaml`,
`recipe: unified`): the *same* DINO objective, head and multi-crop, but on
`arch: vit_base` under the unified recipe — AdamW lr 6e-4, a **fixed** weight
decay (step 1 cosine-schedules it), betas (0.9, 0.95), per-iteration cosine LR
with a 10-epoch warmup, 300 epochs, milestone checkpoints at 100/200/300, each
probed by the same frozen-teacher `linear_eval`. Because these ViT-native methods
already use `arch` for the model *size*, the Step-2 path is selected by an
explicit `recipe: unified` key (absent = the native ViT-S/16 recipe, byte-for-byte
unchanged). DINO ships **its own** Vision Transformer
(`models/vision_transformer.py` — measured: it imports only `torch`, **not
`timm`**, and `vit_base` is a build option), so **both** paths are torch-only,
unlike the timm-based MoCo v3 — no new dependency.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/23_dino` (the lab's own DINO, following
[facebookresearch/dino](https://github.com/facebookresearch/dino): the ViT
backbone, the DINO head, the student/teacher model with its centring + EMA
teacher, the multi-crop dataset, the trainer and the probe) — no `third_party/`
submodule.

The lab wrapper trains under `DistributedDataParallel` with AMP autocast and logs
to TensorBoard; none is needed for a single-process run, so `train_pretrain_dino.py`
owns a thin fp32 loop, the device is **resolved** rather than assumed CUDA, and
AMP / TensorBoard / tqdm are dropped. The `MultiCropWrapper`, the centred+sharpened
DINO loss, the online centre EMA, the teacher EMA update, the per-parameter grad
clipping and the freeze-last-layer are kept faithfully; `img_size` is **threaded**
through the model and dataset (the capture hard-coded 224) so a small hermetic CPU
smoke runs at a lower resolution.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **teacher** ViT backbone (`teacher.backbone.*`, the prefix
stripped so it loads straight into a plain `VisionTransformer`): the class token,
the position embedding, the patch-embed conv, the transformer blocks and the final
norm — one embed_dim CLS feature per image (384 for `vit_small`). The DINO head,
the centre buffer and the whole student are training machinery and are excluded,
and the round trip (write it, load it back into a rebuilt ViT, compare the
weights) is tested. **The teacher, not the student, is shipped** — it is the
representation DINO is known for, and the capture's own linear eval defaults to it.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation — the
same normalisation DINO trains with); the probe follows the lab's shared ARSSL
protocol (features cached once, mean-centred and L2-normalised, a single linear
layer trained with SGD under a cosine schedule), which makes the number comparable
across the ported methods. (The capture's own DINO eval concatenates the last-4
blocks' CLS tokens and trains a distributed head; using the shared single-feature
probe instead is a documented deviation, the same as every other port.)

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a small `vit_small` at 32px global /
  16px local (a 2×2 / 1×1 token grid), a narrow head, 2 local crops, a few
  fabricated images — runs through `python -m adapter` on a CPU (exercising the
  EMA teacher update and the centring), passes `contract-test`, and the encoder
  round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a pretrain
  encoder over a two-class ImageFolder, passes `contract-test`, writes the
  comparable `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the DINO recipe (`vit_small`,
  out_dim 65536, 100 epochs, batch 1024, AdamW, warmup 10, multi-crop 2+8), a
  recipe, not a completed run.
- **Exercised (unified ViT-B/16 Step 2):** a hermetic smoke — `recipe: unified`,
  `arch: vit_base` at 32px, a narrow head, 2 local crops, two epochs with
  `save_at_epochs: [1, 2]` — runs through `python -m adapter` on a CPU, writes
  `encoder.pt` and both `encoder_epoch{1,2}.pt` milestones, and a milestone probe
  passes `contract-test`. The full 300-epoch ViT-B/16 recipe has not been run here.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/23_dino-pretrain-device.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML — the self-contained torch-only
stack (no submodule, no `timm`; DINO's ViT is its own). `requirements.lock.txt`
(CPU) and `requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the
same closure as `05_jigsaw_puzzle`: identical floors, identical resolution).

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/23_dino/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT is a folder of training images (searched recursively)
    python bin/resolve-config.py --config methods/23_dino/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/images --out resolved.json
    cd methods/23_dino && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/23_dino/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/23_dino && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
