# cosmos3_super — as-is linear eval on the frozen Cosmos3-Super vision encoder (eval-only)

NVIDIA **Cosmos3-Super** (*Cosmos world foundation models*, NVIDIA, 2026;
[huggingface.co/nvidia/Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super);
[github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)), a video
world-foundation model whose vision encoder is a Qwen3-VL vision tower.

This is a **pure eval-only** port: a single `linear_eval` stage and **no
pretraining at all**. Cosmos3-Super's as-is comparison freezes the released
model's **vision encoder** and fits a linear probe on its **mean-pooled patch
tokens** (the 64B MoT / DiT / VAE are never loaded), because Cosmos3-Super's
from-scratch training is the excluded step. The probed representation is a genuine
learned feature, so the number **is** comparable — the multimodal
"pretrained-backbone reuse" row, the `transformers`-sourced sibling of `data2vec2`
and `sam3`.

## The checkpoint, stated plainly

The capture's Step-3 VideoGen comparison
(`methods_step3/VideoGen/Cosmos3-Super/adapter.py`) loads only the released
`nvidia/Cosmos3-Super` **vision encoder** (revision `fe77b66…`) and reads it through
`transformers`' `Qwen3VLVisionModel`. Unlike SAM 3, the released layout **matches**
the HF class: the checkpoint is a `vision_encoder/` directory (`config.json` +
`model.safetensors`) that `Qwen3VLVisionModel.from_pretrained(dir,
local_files_only=True)` loads directly — no trunk conversion is needed, and a
`save_pretrained` → `from_pretrained` round-trip is exact. All of this is recorded
in `provenance.json`.

The weights are NVIDIA's under **OpenMDW-1.1** (public, not gated). Nothing under
it is copied here; the model constructor is imported from the pinned `transformers`
dependency, and the `model.safetensors` is a sha256-pinned file (`bin/fetch-weights.py`
verifies the hash of a copy you supply). The sha256 and byte count are the ones the
capture's `SOURCE_SNAPSHOT.json` recorded and the Hugging Face LFS metadata reports.

## Why this method, and what is new here

**cosmos3_super is a multimodal, `transformers`-sourced eval-only port**, the
sibling of `data2vec2` and `sam3`. Its only stage is `linear_eval`. The model class
is `transformers`' `Qwen3VLVisionModel` — a pinned pip dependency
(`transformers==5.16.1`, the fleet pin), not a git submodule. What is new relative
to `sam3`: the official checkpoint is **directly loadable** by the HF class, so this
port carries **no** trunk converter — it loads with `from_pretrained` on the
`vision_encoder/` directory, and the loading path is unit-tested by saving a tiny
model and reloading it (the round-trip is exact) so the real-run path is covered
without the ~1.1 GB weights.

**There is no author submodule.** The model class is transformers' — imported and
never copied — and the released weights are a **sha256-pinned, public download**
recorded as `backbone_artifact` in `provenance.json`. So this port records **no**
upstream block and defines **no** `UPSTREAM` in the adapter (they must agree, and
here both are absent). The lock closes over transformers' own dependency tree
(huggingface-hub, tokenizers, safetensors, regex, …).

## No `encoder.pt` — a frozen, downloaded backbone

This is the frozen-backbone / weight-download shape that CONTRACT section 7 left
open — see `docs/EVAL_DOWNLOAD.md`. The stage trains nothing and produces no
`encoder.pt`: it builds a `Qwen3VLVisionModel`, loads the checkpoint into it frozen,
and fits a single linear layer on the mean-pooled patch feature. The manifest
therefore carries `encoder_absent_reason` rather than an encoder, and the backbone
it read is named in the config (`train.ckpt`).

Changed during the port (see `provenance.json`): the device is **resolved**
(`resolve_device`) rather than assumed CUDA; features are extracted in **fp32** (no
autocast / no bf16), so the frozen-feature probe runs identically on a CPU or a
pre-Ampere GPU, rather than the checkpoint's bf16; the feature is the vision tower's
patch tokens **mean-pooled** over the sequence, so `feature_dim = hidden_size`
(1152, the frozen-probe width the capture uses — not the merger's `out_hidden_size`
5120); the input normalisation is ImageNet's (the capture's adapter); the probe
follows this port's shared frozen-backbone protocol (mean-centre + L2-normalise, one
linear layer with SGD + cosine). The vision tower takes flattened patches, not a
pixel grid: each image is unfolded into `patch_size`×`patch_size` patches, each
repeated `temporal_patch_size` times (a single frame over the tower's minimal
temporal window), with `grid_thw = [[1, H_p, W_p]]` per image.

## The representation, and the caveat

The probe reads Cosmos3-Super's mean-pooled patch tokens, frozen. A real number
therefore measures the **pretrained** backbone, not something this port trained. The
released checkpoint is a **public download pinned by sha256** in `provenance.json`,
whose hash `bin/fetch-weights.py` verifies against a copy you supply. The hermetic
smoke leaves `train.ckpt` empty and builds a **random tiny** `Qwen3VLVisionModel` at
a small resolution, so nothing is downloaded and its accuracy is meaningless — only
the pipeline is exercised.

## What has and has not been exercised

- **Exercised:** a hermetic smoke fits the probe on a random backbone over a
  two-class ImageFolder, passes `contract-test`, writes the four comparable
  `linear_probe` accuracies, and writes **no** `encoder.pt` (the manifest carries
  `encoder_absent_reason`); two runs of one config agree bit for bit. The
  `from_pretrained` loading path is unit-tested by saving a tiny tower to a
  `vision_encoder/` directory and reloading it: the reloaded tower forwards finite
  and agrees bit for bit with the original, and a missing checkpoint directory is
  refused.
- **Not a full run:** `configs/linear_eval.yaml` pins the official Cosmos3-Super
  vision-encoder recipe, not a completed run; the real checkpoint (~1.1 GB) was not
  downloaded during the port.
- **GPU:** the device resolution is verified; the CUDA probe path is guarded and
  runs where a device is visible.

## Environment

The eval stack is torch / torchvision / transformers / numpy / PyYAML; transformers
supplies the model class and its config, and its own dependency closure
(huggingface-hub, tokenizers, safetensors, regex, …) is pinned in the lock. There is
**no submodule** to check out.

    pip install --require-hashes \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r methods/cosmos3_super/requirements.lock.txt -r requirements-tools.lock.txt

## Running

    # verify a copy of the released vision encoder (model.safetensors pinned by
    # sha256 in provenance.json). The weight is public (OpenMDW-1.1); a real run
    # also needs the sibling config.json in the same vision_encoder/ directory,
    # which from_pretrained reads to build the tower.
    python bin/fetch-weights.py --provenance methods/cosmos3_super/provenance.json \
        --out .weights/cosmos3_super --artifact backbone_artifact
    # DATA_ROOT has train/ and val/ (an ImageFolder each); COSMOS3_CKPT is the
    # vision_encoder/ directory (config.json + model.safetensors)
    python bin/resolve-config.py --config methods/cosmos3_super/configs/linear_eval.yaml \
        --set DATA_ROOT=/path/to/imagenet \
        --set COSMOS3_CKPT=/path/to/Cosmos3-Super/vision_encoder \
        --out resolved.json
    cd methods/cosmos3_super && PYTHONPATH="$PWD/../.." \
        python -m adapter --config /path/to/resolved.json --out /path/to/out

Success is exit status 0 and `status: "ok"` in `out/run_manifest.json`. This
stage writes `metrics.json` and **no** `encoder.pt`; the manifest carries
`encoder_absent_reason`. Read what that number means in the section above before
comparing it.
