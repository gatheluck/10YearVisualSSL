# CLIP (method 38)

Paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

Upstream: [openai/CLIP](https://github.com/openai/CLIP) (MIT), pinned as the
submodule `third_party/CLIP` and imported, never copied.

Two comparisons live here, selected by the stage (and, for the eval, a `recipe`):

## Step 1 (as-is)

`linear_eval` with no recipe. Freeze the released OpenAI **ViT-B/32** image tower
and linear-probe its pooled image embedding. CLIP's 400M image-text training data
and pipeline are not public, so the as-is row reuses the released checkpoint (a
sha256-pinned download, `provenance.json: backbone_artifact`, fetched by
`bin/fetch-weights.py`). This trains nothing and writes no `encoder.pt`. See
`docs/EVAL_DOWNLOAD.md`.

## Step 2 (label-text adaptation)

`pretrain` trains a CLIP **ViT-B/16** (image tower + 12-layer text tower) from
scratch on ImageNet-1k, pairing each labeled image with an official OpenAI
ImageNet class-name prompt (symmetric image-text contrastive loss), 300 epochs,
global batch 1024, AdamW, milestone frozen-backbone probes at 100/200/300. Then
`linear_eval` with `recipe: unified` probes the trained image tower (`encoder.pt`).

**This is a supervised label-text adaptation, not unlabeled VSSL.** Exact CLIP
pretraining is undefined under the ImageNet-1k Step-2 restriction (no captions),
so each labeled image is paired with a class-name prompt. Every config, checkpoint
and result records `supervised_label_text_adaptation=true` and
`main_vssl_comparability=false`; the number is a CLIP-adaptation reference and must
**not** be reported as a comparable self-supervised ImageNet result.

## Running

```
cd methods/38_clip && PYTHONPATH="$PWD/../.." python -m adapter \
    --config <resolved.json> --out <dir>
```

Resolve a config first with `bin/resolve-config.py` (it fills `${DATA_ROOT}`,
`${CLIP_VITB32_CKPT}`, `${ENCODER}`). The hermetic smokes build a random tiny CLIP
and download nothing; a real Step-1 run needs the pinned ViT-B/32 download and a
real Step-2 run needs ImageNet-1k. The from-scratch pretraining is GPU-class work
and has not been executed here; the settings in `configs/pretrain_vit.yaml` are the
captured recipe, not a result.

## What is imported from the submodule

The CLIP model constructor (`clip.model.CLIP` / `VisionTransformer`), the BPE
tokenizer (`clip.tokenize`; its vocab ships inside the submodule) and the official
1000 ImageNet class names + 80 prompt templates (from the pinned notebook) are
imported from `third_party/CLIP` through PYTHONPATH. The `clip` package's own
runtime deps (`ftfy`, `regex`, `tqdm`, `packaging`, `wcwidth`) are pinned in the
locks; they are the submodule's imports, not this port's, so they are not in
`requirements.txt`.
