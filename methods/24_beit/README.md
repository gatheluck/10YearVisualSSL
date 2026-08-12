# 24_beit — step 1 (BEiT MIM pretext) + linear evaluation

Bao et al., *BEiT: BERT Pre-Training of Image Transformers*, 2021
([arXiv:2106.08254](https://arxiv.org/abs/2106.08254)).

BEiT is **masked image modeling** with discrete targets. A **dVAE tokenizer**
turns each image into a grid of discrete **visual tokens**; a random block of
image patches is replaced by a shared learned **mask token** in the ViT input;
the ViT predicts the visual tokens at the masked positions — a **cross-entropy
over the dVAE vocabulary** (the image analogue of BERT's masked-token prediction).
Step 1 is that pretext.

## Scope — the ViT-Base step 1 only

This port covers BEiT's **ViT-Base/16** step 1. The capture's step 2 (ViT
fine-tuning) is excluded, as in every port. The ViT is the lab's **own**
(LayerScale blocks) — **no `timm`**. The MIM targets come from the frozen **OpenAI
DALL-E dVAE** tokenizer: for a **real run** it is a hash-pinned download of
`encoder.pkl` (`provenance.json` → `tokenizer_artifact`), a pickled `nn.Module`
unpickled by the `dall_e` code — the pinned **`third_party/dall_e` submodule**
(imported lazily through PYTHONPATH, never copied or installed, like the repo's
other upstream pins). The **hermetic smoke** uses a random torch-only tokenizer,
so CI downloads nothing and imports no submodule.

## Why this method, and what is new here

**A self-contained re-implementation** ported from the capture's own
`methods/24_beit` (the lab's own BEiT, following
[microsoft/unilm/beit](https://github.com/microsoft/unilm/tree/master/beit): the
BEiT ViT with its MIM head, the DALL-E dVAE tokenizer wrapper, the blockwise mask
generator + dual-view dataset, the trainer and the probe) — no `third_party/`
submodule for the model.

The lab wrapper trains under `DistributedDataParallel` with an AMP autocast and
logs to TensorBoard; none is needed for a single-process run, so
`train_step1_beit.py` owns a thin fp32 loop, the device is **resolved** rather
than assumed CUDA, and AMP / TensorBoard / tqdm are dropped. The blockwise mask
generator, the mask-token replacement and the masked-position cross-entropy are
kept faithfully; `build_beit` is the single construction path shared by the
trainer and the linear-eval loader; `img_size` and the ViT dims are **threaded**
so a small hermetic CPU smoke runs a tiny ViT at a lower resolution.

The tokenizer only produces the MIM **targets**; the representation this port
ships is the trained BEiT backbone, not the tokenizer.

## `encoder.pt`, and a linear evaluation that reads it

`encoder.pt` is the **BEiT backbone trunk** (`patch_embed`, `cls_token`,
`pos_embed`, `blocks`, `norm`) — one embed_dim mean-pooled patch-token feature per
image (768 for ViT-Base, the CLS token excluded). The shared mask token
(`mask_token`) and the MIM prediction head (`head.*`) are training machinery and
are excluded, and the round trip (write it, load it back into a rebuilt BEiT,
compare the weights) is tested.

`linear_eval` reads this `encoder.pt`: the representation is the model this port
trains, so the probe number is a genuine, comparable linear probe. Images use the
deterministic val pipeline (resize + centre crop, ImageNet normalisation); the
probe follows the lab's shared ARSSL protocol (features cached once, mean-centred
and L2-normalised, a single linear layer trained with SGD under a cosine
schedule), which makes the number comparable across the ported methods. (The
capture's own BEiT eval fine-tunes the ViT end to end; using the shared
single-feature probe instead is a documented deviation, the same as every other
port.)

## What has and has not been exercised

- **Exercised (step 1):** a hermetic smoke — a tiny ViT at 32px (a 2×2 patch grid,
  embed_dim 32), a random torch-only tokenizer, a few fabricated images — runs
  through `python -m adapter` on a CPU (exercising the block masking, the
  mask-token replacement and the masked-position cross-entropy), passes
  `contract-test`, and the encoder round-trip and a determinism check pass.
- **Exercised (linear_eval):** a hermetic smoke fits the probe on a step-1 encoder
  over a two-class ImageFolder, passes `contract-test`, writes the comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt`.
- **Not a full run:** `configs/pretrain.yaml` is the BEiT recipe (ViT-Base/16, 224px,
  8192-token DALL-E dVAE, 800 epochs, batch 2048, AdamW, warmup 10 → cosine), a
  recipe, not a completed run. A real run needs the DALL-E `encoder.pkl` (fetch
  and hash-verify it with `bin/fetch-weights.py --artifact tokenizer_artifact`)
  passed as `--set TOKENIZER_CKPT=<path>`, plus the `third_party/dall_e` submodule
  checked out (`git submodule update --init third_party/dall_e`) and its own deps
  (`requests`, `attr`, …) installed.
- **GPU:** the device resolution is verified on real hardware; see the device
  mutation spec (`mutations/24_beit-step1-device.json`).

## Environment

torch / torchvision / numpy / Pillow / PyYAML. `requirements.lock.txt` (CPU) and
`requirements.lock.cu130.txt` (CUDA 13.0) are the hashed closures (the same
torch+Pillow closure as `05_jigsaw_puzzle`). The `dall_e` code is **not** in the
lock: it is the pinned `third_party/dall_e` submodule, imported lazily for a real
run only, and it carries its own deps (`requests`, `attr`, …); the smoke does not
use it.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/24_beit/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # step 1: DATA_ROOT has a train/ subdirectory; TOKENIZER_CKPT is the DALL-E
    # encoder.pkl (fetch it with bin/fetch-weights.py --artifact tokenizer_artifact)
    python bin/resolve-config.py --config methods/24_beit/configs/pretrain.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set TOKENIZER_CKPT=/path/to/encoder.pkl --out resolved.json
    cd methods/24_beit && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/s1

    # linear eval: DATA_ROOT has train/ and val/; ENCODER is step 1's encoder.pt
    python bin/resolve-config.py --config methods/24_beit/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set ENCODER=/path/to/s1/encoder.pt --out eval.json
    cd methods/24_beit && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/eval.json --out /path/to/eval

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. The
linear_eval stage writes `metrics.json` and **no** `encoder.pt`; the manifest
carries `encoder_absent_reason`.
