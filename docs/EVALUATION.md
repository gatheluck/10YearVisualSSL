# What the capture evaluates, and what this repository currently runs

Last updated: 2026-08-12

This document records, **fact-based**, two things that are easy to conflate:

1. the **full evaluation design** the capture side uses to produce the paper's
   results table (Step 1 / Step 2 across several downstream tasks), and
2. **what this published repository actually implements today** — which is a
   strict subset of it.

It exists so the gap is written down and nobody assumes the current ports
reproduce the whole table. Sources are cited by path; the capture's experiment
code lives on the `snapshots` branch of
`gatheluck/10YearVisualSSLCapturePrivate` (read-only from here).

---

## 1. The capture's evaluation design (source of truth)

Each method is studied along **two axes** — call them the capture's **Step 1**
and **Step 2** — and each is evaluated on **five downstream tasks**.

### Step 1 — as-is (frozen native/official backbone)

`configs/step1_downstream_registry.yaml` (snapshots) states the rule:

> "Run downstream Step 1 tasks only from accepted Step 1 artifacts: official
> method-specific probe/eval when available, otherwise official/official-style
> frozen linear probe."

So **Step 1 takes each method's own released or original-architecture model,
freezes it, and evaluates it** on the downstream tasks. It trains no new
representation. For foundation models this is the officially released checkpoint;
for older methods it is the paper's native backbone (AlexNet / ResNet / ViT).

### Step 2 — unified SSL comparison, trained **from scratch**

`configs/step2_downstream_registry.yaml` (snapshots):

> "Run every downstream Step 2 task at ImageNet pretraining epochs 100, 200, and
> 300; frozen backbone, task-specific linear/FRCNN/DPT heads."

and each method's `methods/<m>/train_step2_vit.py` (snapshots) — e.g. InstDisc's
docstring: *"Step 2: Instance Discrimination with ViT-Base. Encoder: ViT-B/16 →
Linear(768,128) → L2-normalise. Epochs: 300 (checkpoint at 100, 200, 300)."*
with config header *"Step 2: Unified SSL Comparison — InstDisc with ViT-Base"*
(`configs/step2_vit.yaml`: `arch: vit_base_patch16_224`, `epochs: 300`,
`batch_size: 1024`, AdamW + cosine + warmup, ImageNet train/val).

Key facts, measured from the code:

- **Step 2 trains from scratch, not fine-tuning.** The model is built fresh
  (`build_vit_<method>(...)`); there is **no pretrained-weight load** anywhere
  (`pretrained` / `from_pretrained` do not appear; the only `load_state_dict` is
  under `--resume`, i.e. continuing an interrupted run).
- **It is a unified backbone.** Every method plugs its own SSL loss into the
  **same ViT-B/16**, the same ImageNet data, and the same optimiser recipe,
  differing only in the objective. That is the point: a fair, like-for-like
  comparison across ten years of methods.
- **Three checkpoints** (100 / 200 / 300 epochs) are each evaluated **frozen**.

This is why, in the results, a method's Step 2 number can be **lower** than its
Step 1 number (e.g. a foundation model): a from-scratch ViT-B/16 trained for
≤300 epochs need not match a released model trained on curated data at scale.

### The five downstream tasks (both steps)

`downstream/` (snapshots) implements, with **frozen backbones**:

| Task | Dataset | Head / protocol | Metric |
|---|---|---|---|
| Classification | ImageNet-1k | linear probe | Top-1 / Top-5 |
| Detection | COCO | frozen backbone + Faster R-CNN heads (`coco_frcnn.py`) | mAP |
| Segmentation | ADE20k | linear probe (`ade20k_segmentation.py`) | mIoU / pACC |
| Depth | NYUv2 | frozen backbone + DPT head (`nyuv2_depth.py`) | RMSE / AbsRel (↓) |
| Video | SSv2 | linear probe (`ssv2_linear.py`) | Top-1 / Top-5 |

So per method the full table is up to **5 tasks × (Step 1 + Step 2 at 3
epochs)** ≈ ~20 numbers.

---

## 2. What this repository implements today (measured)

- **Pipeline per method**: adapter stages `pretrain` (self-supervised pretraining →
  `encoder.pt`) and `linear_eval` (frozen linear probe).
- **The only downstream evaluation is ImageNet-1k classification linear probe
  (Top-1/Top-5).** This is enforced by the contract itself:
  `adapterlib.METRIC_VOCABULARY` contains only `final_pretext_*` (per-method) and
  `*_linear_probe_top1/5_accuracy` (comparable). There is **no vocabulary for
  mAP / mIoU / RMSE / AbsRel / video accuracy** — the other four tasks cannot
  even be recorded.
- **Datasets referenced by configs**: ImageNet, and MNIST (for the VAE). No
  COCO / ADE20k / NYUv2 / SSv2 anywhere in code or configs.
- **Eval-only download methods** (dinov2 / aim / franca) probe the official
  frozen backbone on ImageNet — i.e. the ImageNet-classification cell of the
  capture's **Step 1 (as-is)**.
- **Trained methods** run a from-scratch SSL pretraining on their **native**
  architecture (not necessarily a unified ViT-B/16) at a single configurable
  epoch count, then the ImageNet linear probe — i.e. one ImageNet-classification
  cell in the spirit of **Step 2**, but on the method's own backbone and at one
  epoch budget, not the unified-ViT 100/200/300 sweep.

There is no 100/200/300 sweep harness, no as-is-vs-from-scratch matrix, and no
unified-ViT Step-2 backbone in the published repository.

---

## 3. Gap at a glance

| Capture element | This repository |
|---|---|
| ImageNet-1k classification linear probe (Top-1/5) | **implemented** (`linear_eval`) |
| SSL pretraining (from scratch) | **implemented** (`pretrain`, epochs configurable) |
| Step 1 as-is (official/native frozen backbone) | partial: eval-only download methods, ImageNet only |
| Unified ViT-B/16 Step-2 backbone | **not implemented** (ports use native arch) |
| 100 / 200 / 300 epoch sweep + per-checkpoint eval | **not implemented** |
| COCO detection (frozen + FRCNN, mAP) | **not implemented** (no code, config, or metric vocabulary) |
| ADE20k segmentation (linear, mIoU/pACC) | **not implemented** |
| NYUv2 depth (frozen + DPT, RMSE/AbsRel) | **not implemented** |
| SSv2 video (linear, Top-1/5) | **not implemented** |

Net: the current implementation produces, in effect, **one column** of the
capture's table — ImageNet-1k classification — and none of the detection,
segmentation, depth, or video tasks, nor the structured Step-1/Step-2 ×
epoch-sweep matrix.

---

## 4. Terminology — "step" is the paper axis, not a code stage

The pipeline stage that used to be called `step1` is now `pretrain`, precisely so
that **"Step" belongs to the paper/results axis only** and never collides with a
code stage token (`tests/test_stage_vocabulary.py` forbids `step1` as a stage).

- **Capture experiment axis** (this document, and the results CSV): **Step 1** =
  as-is frozen evaluation; **Step 2** = from-scratch unified-ViT pretraining at
  100/200/300 epochs, evaluated frozen. This is the paper's comparison design.
- **Adapter pipeline stages** (`adapterlib.CONTRACT_STAGES`): `pretrain` = the
  SSL pretraining stage that writes `encoder.pt`; `linear_eval` = the frozen
  linear probe; `knowledge_transfer` = a pretext middle stage. No stage is named
  `step1` or `step2`.
- Mapping: the capture's **Step 2 (from-scratch)** corresponds, for a ported
  trained method, to running the adapter's `pretrain` + `linear_eval` (ImageNet
  only); the capture's **Step 1 (as-is)** corresponds to an eval-only method's
  `linear_eval` on a downloaded frozen backbone.

---

## 5. What closing the gap would require

Reproducing the full capture table in this repository would need, at least:

1. **New contract evaluation stages and metric vocabulary** for detection
   (mAP), segmentation (mIoU / pACC), depth (RMSE / AbsRel), and video
   (Top-1/5), alongside the existing `linear_eval`.
2. **Downstream data adapters and frozen-backbone eval code** for COCO
   (FRCNN head), ADE20k (linear), NYUv2 (DPT head), and SSv2 (linear).
3. **A unified ViT-B/16 Step-2 pretraining path** per method (the capture's
   "unified SSL comparison" backbone), and a **100/200/300 epoch sweep** with
   per-checkpoint frozen evaluation and result aggregation.

This is comparable in size to the porting effort itself and should be scoped and
built incrementally, strict TDD, one task at a time.
