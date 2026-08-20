# Downstream tasks beyond ImageNet-1k: detection, segmentation, depth, video

Last updated: 2026-08-20

Until now the port evaluates every method on **one** downstream task: an
ImageNet-1k linear probe (the `linear_eval` stage). The Capture repo, however,
evaluates each accepted backbone on a **shared battery of dense and
recognition tasks** — COCO detection, ADE20K semantic segmentation, NYUv2 depth,
and Something-Something-v2 video — and that harness is now in the capture. This
document records what was **measured** about that harness and the design the port
adopts to bring those tasks over. It is measurement and best practice, not
preference; the capture sources are named so the reasoning can be re-checked.

The evidence is the capture's `downstream/` package on the `snapshots` branch
(`origin/snapshots:downstream/`) and `configs/step{1,2}_downstream_registry.yaml`.

---

## 1. What the capture harness is (measured)

`downstream/` is a **single, cross-method package** — not per-method code. It
takes any method's *frozen* backbone, wraps it into a spatial feature extractor,
attaches a small task head, trains only the head, and reports the task metric.

- **Task runners** (one file each, all torchvision-native — measured; **no
  detectron2 / mmseg / mmcv anywhere**):
  - `coco_frcnn.py` — COCO detection, `torchvision.models.detection.FasterRCNN`
    on the frozen backbone, mAP via **pycocotools** (`COCOeval`).
  - `ade20k_segmentation.py` — ADE20K semantic segmentation, a 1×1-conv readout
    head + bilinear upsample, **mIoU**/pACC via a confusion matrix (torch + PIL +
    numpy only).
  - `nyuv2_depth.py` — NYUv2 depth, a depth head; needs **h5py** (the labelled
    `.mat`).
  - `ssv2_linear.py` — Something-Something-v2, a frame-based linear head; needs
    **av** (PyAV) for video decoding.
- **Backbone dispatch** — `backbone_adapters.build_frozen_backbone(artifact)`
  selects by `family`: `resnet50` → ResNet layer-4 spatial map
  (`resnet_adapters.py`); `vit`/`video_vit`/`step2_vit` → ViT patch-token spatial
  map (`vit_adapters.py`); alexnet/cnn → `classic_adapters.py`. Every built
  backbone exposes the **same two-symbol interface** the task heads rely on:
  `forward_features(x) -> [B, C, h, w]` and an `out_channels: int`.
- **Registry** — `registry.py` (`MethodArtifact`, `load_config`,
  `dataset_root`, `load_ready_artifact`) and
  `configs/step{1,2}_downstream_registry.yaml` list the datasets (each with a
  `root_env` such as `COCO_ROOT` and a `required:` path list) and the per-method
  accepted artifacts. Step 2 lists every method under one family, `step2_vit`
  (the unified ViT-B/16), pointing at `checkpoint_epoch_{100,200,300}.pth`.
- **Metric honesty** — every runner takes `max_train_samples` /
  `max_val_samples` / `max_steps_per_epoch`, records `subset_or_smoke` and sets
  `record_value: false` when any subsetting is on, so a smoke number can never be
  mistaken for a real one.

## 2. What differs from the port, and the design decision

The capture harness is built around ABCI-side artifacts: `registry.py` resolves
each method from `methods/<n>/local_artifacts/results/*.json` and a checkpoint
path on the cluster. The **port** does not have that layout — it produces
`encoder.pt` (and milestone `encoder_epoch{N}.pt`) in each run's `--out`
directory, verified by the contract. So the registry mechanism is **not
transplanted verbatim**; the port drives downstream tasks from its own
`encoder.pt` artifacts.

**Decision: a cross-method `downstream/` subsystem in the port, consuming
`encoder.pt` — not new per-method contract stages.** The alternative (adding
`coco_detection`/`ade20k_segmentation`/… stages to every method's `adapter`)
would copy the *same* task code across all 38 adapters, because every method's
Step-2 backbone is the one unified ViT-B/16. That violates the repository rule
"never implement the same rule twice" (CLAUDE.md, DESIGN §best-practice). The
capture itself keeps `downstream/` separate from `methods/` for the same reason.
The port mirrors that: **one implementation, imported, registry-driven.**

Concretely the port grows:

- a top-level `downstream/` package (task runners + backbone adapters +
  registry), added to the README `## Repository layout` tree;
- a **backbone interface** the runners depend on and each method's `encoder.pt`
  satisfies via one shared adapter — for Step 2 that is a single ViT-B/16 spatial
  adapter (`forward_features -> [B,C,h,w]`, `out_channels`), plus a CLIP branch
  (CLIP's `VisionTransformer` has no `patch_embed.proj`, so its conv-patch tap
  differs), and a ResNet-layer4 adapter for the ResNet Step-1 backbones;
- a **downstream contract**: each task run emits a manifest + a metric JSON, and
  a `contract-test`-style check decides "the task ran" by machine (exit 0 + a
  status, the same shape as the method contract). The metric vocabulary gains the
  task metrics (`ade20k_miou`, `coco_map`, `nyuv2_rmse`, `ssv2_top1`, …).

## 3. Hermetic CI, real numbers, and datasets

Exactly as with `linear_eval`, CI stays hermetic and downloads nothing:

- each task ships a **CPU smoke** built on a handful of synthetic images/masks
  (the ADE20K runner already supports this through `max_*_samples`) and a random
  or tiny backbone, so the pipeline is exercised end to end while the number is
  meaningless and is stamped `subset_or_smoke: true` / `record_value: false`;
- a **real** number needs a GPU and the real dataset, passed at run time through a
  `*_ROOT` environment variable (the capture's `COCO_ROOT`/`ADE20K_ROOT`/… —
  the same shape as this repo's `DATA_ROOT`), never shipped.

Dataset layouts the runners expect (from the registry configs, for whoever runs a
real evaluation):

- **COCO** (`COCO_ROOT`): `images/{train2017,val2017}`,
  `annotations/instances_{train2017,val2017}.json`.
- **ADE20K** (`ADE20K_ROOT`, `ADEChallengeData2016`):
  `images/{training,validation}`, `annotations/{training,validation}`.
- **NYUv2** (`NYUV2_ROOT`): `labeled/nyu_depth_v2_labeled.mat`.
- **SSv2** (`SSV2_ROOT`): `videos/`, `labels/{train,validation,labels}.json`.

## 4. Dependencies (all new fleet deltas — measured)

None of these are in any method's locks today (`pycocotools`, `h5py`, `av`,
`opencv`, `detectron2`, `mmseg`, `mmcv` all absent — measured 2026-08-20), and
the harness needs none of the heavy frameworks:

| Task | New dependency | Notes |
|---|---|---|
| ADE20K segmentation | **none** | torch + torchvision + PIL + numpy (all fleet) |
| COCO detection | `pycocotools` | torchvision `FasterRCNN`; mAP eval |
| NYUv2 depth | `h5py` | reads the labelled `.mat` |
| SSv2 video | `av` (PyAV) | video decoding |

Each is a single fleet delta: pin one version, add it to both locks (CPU +
cu130), as `timm`/`ftfy` were. Because the harness is torchvision-native, there
is no lock-breaking framework to reconcile.

## 5. Migration order (portable/light first, the ViT-Step-2-pilot pattern)

1. **Scaffolding + ADE20K segmentation (pilot).** No new heavy dependency. Stands
   up the `downstream/` package, the registry, the shared **ViT-B/16 spatial
   adapter** (+ a CLIP branch), the hermetic CPU smoke, and the downstream
   manifest/metric contract (mIoU). This becomes the template for the rest.
2. **COCO detection.** Adds `pycocotools`; reuses the spatial adapter; mAP.
3. **NYUv2 depth.** Adds `h5py`; a depth head; RMSE / δ metrics.
4. **SSv2 video.** Adds `av`; frame-based for an image backbone; the heaviest and
   most niche, so last.

Each step follows the repository's discipline: RED test first, hermetic smoke +
`contract-test`-style check, a measured mutation spec, `discover-not-list`
enforcement, per-task lock delta, and docs kept consistent.

## 6. CONTRACT note (design of record is capture-side)

The downstream evaluation protocol — which tasks count, which metric each
reports, and how "the task ran" is decided — is a **CONTRACT-level** extension.
The design of record is `docs/CONTRACT.md` / `docs/DESIGN.md` on the Capture side
(§7 already covers frozen-backbone / eval-only shapes). Those are **not edited
from this repository**; this document is the port-side design, and the CONTRACT
wording should be carried over on the capture side when the first downstream task
lands.
