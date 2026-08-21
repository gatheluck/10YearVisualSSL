# What the capture evaluates, and what this repository currently runs

Last updated: 2026-08-21

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
- **ImageNet-1k classification is the per-method probe**: the `linear_eval`
  stage, whose comparable metrics live in `adapterlib.METRIC_VOCABULARY`
  (`*_linear_probe_top1/5_accuracy`).
- **The other four downstream tasks are now implemented** (2026-08-21), in a
  cross-method `downstream/` subsystem — see docs/DOWNSTREAM.md: **ADE20K**
  segmentation (mIoU/pACC), **COCO** detection (bbox mAP / mAP@50), **NYUv2**
  depth (RMSE/AbsRel), **SSv2** video (Top-1/5). Each freezes a method's backbone,
  trains only a task head, and is decided by its **own** contract
  (`downstream/contract.py`, with its own metric names `ade20k_miou`, `coco_map`,
  `nyuv2_rmse`, `ssv2_top1`, …) — deliberately separate from the method vocabulary,
  which is method-scoped (every name there must be produced by a `methods/*/adapter`,
  `tests/test_metric_vocabulary.py`). **CI runs all four as hermetic smokes** (a
  random tiny ViT + synthetic data) via the `downstream` job in
  `.github/workflows/tests.yml`.
- **Datasets**: ImageNet + MNIST for the method pipeline. The downstream tasks read
  COCO / ADE20k / NYUv2 / SSv2 at run time through `COCO_ROOT` / `ADE20K_ROOT` /
  `NYUV2_ROOT` / `SSV2_ROOT` (docs/DOWNSTREAM.md §3); CI ships none of them and
  downloads nothing — a real number needs a GPU and the real dataset.
- **Eval-only download methods** (dinov2 / aim / franca) probe the official
  frozen backbone on ImageNet — i.e. the ImageNet-classification cell of the
  capture's **Step 1 (as-is)**.
- **Trained methods** run a from-scratch SSL pretraining on their **native**
  architecture (not necessarily a unified ViT-B/16) at a single configurable
  epoch count, then the ImageNet linear probe — i.e. one ImageNet-classification
  cell in the spirit of **Step 2**, but on the method's own backbone and at one
  epoch budget, not the unified-ViT 100/200/300 sweep.
- **Step-2 unified ViT-B/16 — pilot: `06_rotation_prediction`.** Its
  `configs/pretrain_vit.yaml` (`arch: vit`) trains the capture's Step-2 backbone —
  a timm ViT-B/16 from scratch — with the rotation objective, checkpointing at
  100/200/300 (`encoder_epoch{N}.pt`) so the existing ImageNet `linear_eval` runs
  per milestone. The native AlexNet path is unchanged. This is the pilot pattern;
  the other trained methods still use their native arch until ported the same way.

Apart from that pilot there is no 100/200/300 sweep **driver** (the probe is run
per milestone by hand), no as-is-vs-from-scratch matrix, and no unified-ViT
Step-2 backbone for the other methods yet.

---

## 3. Gap at a glance

| Capture element | This repository |
|---|---|
| ImageNet-1k classification linear probe (Top-1/5) | **implemented** (`linear_eval`) |
| SSL pretraining (from scratch) | **implemented** (`pretrain`, epochs configurable) |
| Step 1 as-is (official/native frozen backbone) | partial: eval-only download methods, ImageNet only |
| Unified ViT-B/16 Step-2 backbone | **pilot: `06_rotation_prediction`** (`configs/pretrain_vit.yaml`); other methods still native arch |
| 100 / 200 / 300 epoch sweep + per-checkpoint eval | **pilot: `06`** writes `encoder_epoch{100,200,300}.pt`, probe run per milestone; no cross-method driver yet |
| COCO detection (frozen + FRCNN, mAP) | **implemented** (`downstream/coco.py`; hermetic smoke in CI; real number needs GPU + `COCO_ROOT`) |
| ADE20k segmentation (linear, mIoU/pACC) | **implemented** (`downstream/ade20k.py`; hermetic smoke in CI; needs GPU + `ADE20K_ROOT`) |
| NYUv2 depth (frozen + DPT, RMSE/AbsRel) | **implemented** (`downstream/nyuv2.py`; hermetic smoke in CI; needs GPU + `NYUV2_ROOT`) |
| SSv2 video (linear, Top-1/5) | **implemented** (`downstream/ssv2.py`; hermetic smoke in CI; needs GPU + `SSV2_ROOT`) |

Net: the **task coverage** is now complete — the ImageNet-1k column plus all four
detection / segmentation / depth / video tasks are implemented and their pipelines
run in CI as hermetic smokes. What remains before the full capture *table* can be
reproduced is **real numbers** (GPU + the real datasets, which CI never ships) and
the **structured Step-1/Step-2 × 100/200/300 sweep driver** that aggregates them
across methods and milestones — not new task code.

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

Done (2026-08-21):

1. ~~**Downstream contract + metric vocabulary**~~ — **done**: the downstream
   contract (`downstream/contract.py`) with its own vocabulary (`coco_map`,
   `ade20k_miou`, `nyuv2_rmse`, `ssv2_top1`, …), separate from the method one.
2. ~~**Downstream data adapters and frozen-backbone eval code**~~ — **done**:
   `downstream/{ade20k,coco,nyuv2,ssv2}.py`, each on the shared frozen spatial
   backbone (`downstream/spatial_backbones.py`), run in CI as hermetic smokes.

What still remains before the full capture *table* is reproduced:

3. **Real numbers** — a GPU and the real datasets (`COCO_ROOT` / `ADE20K_ROOT` /
   `NYUV2_ROOT` / `SSV2_ROOT`); CI is hermetic and ships none of them.
4. **The Step-1/Step-2 × 100/200/300 sweep driver** — a cross-method orchestrator
   that runs each task on each method's as-is Step-1 backbone and its from-scratch
   Step-2 `encoder_epoch{100,200,300}.pt`, and aggregates the cells into the table.
   The per-method Step-2 milestone encoders already exist; this is the driver +
   aggregation over them, not new task code.

The remaining work is orchestration and real-data runs, not new evaluation code;
it should be scoped and built incrementally, strict TDD.
