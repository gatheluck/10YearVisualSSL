# INITIAL_STEP3_VIDEO_SSL_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

### 2026-09-26 task correction

The study authors confirmed that the COCO experiments use object detection.
The previous task attribution in this companion was incorrect and is superseded.
The detection reconstruction below comes from the supplied supplementary v4;
its detailed optimizer, resolution and seed settings still require matching to
individual run records. Confirmation of the task is not confirmation of every
candidate hyperparameter or a newly reproduced score. Initial conditions remain
separate from the later Unified LP/AP/FT specifications.

The inspected V-JEPA 2.1 ViT-L ImageNet launcher specifies 20 epochs and batch 128. The reconstruction describes other schedules and both linear and attentive or detection paths. COCO uses the bounding-box detection recipe specified below.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| COCO task | bbox AP (%), AP@[0.50:0.95]. Faster R-CNN, 12 epochs, batch 2, LR `0.001`, WD `1e-4`, 1-epoch warmup, cosine, best validation epoch, 224 square | `3DFM/_common/coco_probe.py`; `INITIAL_STEP3_3DFM_v1.md`; `INITIAL_STEP3_4DFM_v1.md` | Supplied detection reconstruction; run matching pending |
| Frames | 16 | `SSv2ActionDataset` default `num_frames=16` | Observed loader default |
| Resolution | 224 center crop on ImageNet and SSv2; 224 square on COCO; 448 square on ADE20K and NYUv2 | `_SSV2_IMG_TF` and the ImageNet 224 crop; `_ADE_IMG_SIZE = 448` and `_NYU_IMG_SIZE = 448`. COCO square 224 matches `INITIAL_STEP3_4DFM_v1.md` | Observed loader setting for ImageNet, ADE20K, NYUv2, and SSv2. COCO geometry is a unification |
| Depth | Masked L1, metric RMSE on `(0.1, 10)` m | `run_depth` and `compute_depth_metrics` | Observed main-run setting |
| Score epoch | Best validation primary metric | Returned fields `best_epoch`, `best_metric`, `best_rmse` | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

The proposed common schedule is 50 epochs with global batch 64. The historical seed remains unresolved; seed 0 is a proposal, not an observed assignment. Excluding attentive probes defines this candidate scope rather than disproving their original existence.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Schedule | 50 epochs, LR `0.01`, warmup 5 | Most `qsub_*_downstream.sh` echoes use `epochs=50` and `lr=0.01`. V-JEPA 2.1 ViT-L launcher uses 20 epochs and batch 128. V-JEPA 2 ImageNet launcher uses 7 epochs | Newly selected unification |
| Batches | 64 on ImageNet and SSv2, 32 on ADE20K and NYUv2, 2 on COCO | ADE launcher for V-JEPA 2.1-AC echoes `batch=32`. COCO batch 2 is the 3DFM/4DFM detector batch | Newly selected unification |
| Seed | Proposed value `0`, one trial | `downstream_eval.py` and `06_vjepa21/scripts/qsub_imagenet1k_vitl_downstream.sh` do not set a seed | Proposal. Evidence gap |

Candidate settings and execution instructions below are not evidence of completed runs.
They must not retroactively redefine the conditions of published result cells.
No experiment, seed assignment, preprocessing or model implementation is changed
by including this document.

## Supplied candidate specification (attributed reconstruction)

The following detailed reconstruction is retained for review, including its
own evidence/decision table. Its imperative language describes a candidate
procedure, not a command to execute or a claim that each setting was used.
New unifications remain proposals unless matched to a completed run. Statements
about stored files are source attributions, not a fresh audit of those files.

Internal paths and job identifiers have been removed. `${LOCAL_REFERENCE_PATH}`
is an intentionally unresolved placeholder, not a runnable dataset path.
Relative source paths name the original experimental archive; their presence
here does not claim those scripts are delivered or runnable in this package.
Public upstream model names identify comparison methods, not submission authors.

This document is a candidate specification reconstructed from the initial campaign under `methods_step3/VideoSSL/`. It is not a claim that every stored score already follows this file. Stored files include attentive probes and launchers whose epoch counts differ. Attentive probes stay out of this protocol. COCO is box detection.

### Purpose and category scope

Evaluate released video self-supervised trunks as frozen encoders. The example checkpoint is V-JEPA 2.1. Attentive probes under `*_attentive*` are a different readout and are not part of this protocol.

COCO in this protocol is box detection. The primary metric is bbox AP (%), AP@[0.50:0.95].

### Included models and datasets

| Directory | Checkpoint | ImageNet feature width stored on the linear run |
|---|---|---|
| `01_shufflelearn` | Shuffle-and-Learn ResNet-50 | 2048 |
| `02_videomoco` | Video MoCo R(2+1)D | 512 |
| `03_videomae` | `MCG-NJU/videomae-base` | 768 |
| `04_vjepa` | V-JEPA ViT-L/16 encoder used by `04_vjepa` | 1024 |
| `05_vjepa2` | `facebook/vjepa2-vith-fpc64-256` | 1280 |
| `06_vjepa21` | `facebook/vjepa2.1-vitl-fpc64-256` and `facebook/vjepa2.1-vitg-fpc64-256` | checkpoint width |
| `07_vjepa2ac` | `facebook/vjepa2ac` | checkpoint width |

Width is a property of the loaded trunk. Record it. Use the same head, loss, and schedule for every trunk.

| Task | Dataset | Split | Root in `VideoSSL/_common/data_downstream.py` |
|---|---|---|---|
| Image classification | ImageNet-1k | official train / validation | `${LOCAL_REFERENCE_PATH}` |
| Object detection | COCO 2017 | train2017 / val2017 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K | 20,210 / 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Metric depth | NYUv2 | `labeled/splits.mat` `trainNdxs` / `testNdxs`, 1-based indices converted to 0-based; valid depth `(0.1, 10.0)` m | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | official train / validation | `${LOCAL_REFERENCE_PATH}` |

ImageNet-100 linear files are pilots and stay out of the table.

### Global rules

- Load the released trunk. Freeze it. Keep it in `eval()`. The head is the only trainable module.
- Image classification and SSv2 use a 224 center crop after a shorter-side resize to 256. COCO uses a 224 square, and the detector keeps that square. ADE20K and NYUv2 use a 448×448 bicubic square resize (`_ADE_IMG_SIZE`, `_NYU_IMG_SIZE`). ImageNet normalization, mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- One trial. Seed `0` is a proposal: `downstream_eval.py` and the checked linear launchers do not call `manual_seed`. Do not treat seed `0` as an observed historical setting.
- Learning rate `0.01` is absolute at the stated effective batch. Do not sweep and do not rescale.
- The stored score is the best validation value of the primary metric. Report the 1-based epoch that achieved it.
- A backbone that cannot return a real patch grid is unsupported for COCO, ADE20K, and NYUv2. Do not tile a global vector into a 1×1 map.

### Common representation interface

Implementation: `VideoSSL/_common/downstream_eval.py` (`get_global_features`, `get_patch_features`, `get_video_features`).

- **Checkpoint.** The released file named above. No attentive-probe weights.
- **Included components.** The video or frame encoder through its final block. Exclude decoders and contrastive projectors.
- **Frozen scope.** All encoder parameters, including normalization.
- **Feature stage.** Final encoder block.
- **Global tensor.** Mean of the final patch tokens over space. On a clip, mean those vectors over the 16 frames in temporal order. The result is `(B, D)`.
- **Dense tensor.** The same final patch tokens as `(B, D, h, w)`, row-major, with class tokens removed. One 224×224 frame for COCO. One 448×448 frame for ADE20K and NYUv2.
- **Width.** Checkpoint `D`. Record it.
- **Special tokens.** Class tokens are not the feature and are not concatenated.
- **Pooling.** The means above. No L2 normalization before the head.
- **Video geometry.** 16 frames. SSv2 frames use a shorter-side resize to 256 and a 224 center crop (`_SSV2_IMG_TF`).
- **Head boundary.** The linear layer, the 1×1 conv, or the Faster R-CNN heads are the first trainable parameters. The trunk stays frozen.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | 224 | linear, 1000-way | SGD | 50 epochs | best validation Top-1 | Top-1 (%) |
| COCO 2017 | 224 square, detector keeps that square | Faster R-CNN | SGD | 12 epochs, warmup then cosine | best validation bbox AP | bbox AP (%) |
| ADE20K | 448 square | 1×1 conv, 150-way | SGD | 50 epochs | best validation mIoU | mIoU (%) |
| NYUv2 | 448 square | 1×1 conv + softplus | Adam | 50 epochs | best validation RMSE | RMSE (m) |
| SSv2 | 224, 16 frames | linear, 174-way | SGD | 50 epochs | best validation Top-1 | Top-1 (%) |

### Detailed task recipes

ImageNet, ADE20K, NYUv2, and SSv2 use a 5-epoch warmup, then cosine down to a floor of `1e-6` (`WarmupCosineLR`). Effective batch is 64 on ImageNet and SSv2, and 32 on ADE20K and NYUv2. Learning rate is `0.01` on those four tasks. NYUv2 uses Adam. ImageNet, ADE20K, and SSv2 use SGD. COCO uses the detector recipe below.

#### ImageNet-1k

- **Feature.** Spatial mean of the final patch tokens.
- **Head.** `Linear(D, 1000)`.
- **Loss.** Cross-entropy.
- **Optimizer.** SGD, momentum `0.9`, weight decay `0`.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

#### COCO 2017 detection

- **Split.** train2017 and val2017.
- **Target.** Instance boxes for the 80 COCO categories. Drop crowd boxes. Drop boxes whose width or height is at most 1 pixel.
- **Input.** Square resize to 224. Faster R-CNN uses `min_size=224` and `max_size=224`. No horizontal flip.
- **Feature.** Final patch grid as the single scale `"0"`. No global pool.
- **Head.** Torchvision Faster R-CNN. A trainable 1×1 conv maps width `D` to 256 channels. Anchors `(32, 64, 128, 256, 512)` and ratios `(0.5, 1.0, 2.0)`. `MultiScaleRoIAlign` on feature name `"0"`, output 7×7, sampling ratio 2. Train that 1×1 conv, the RPN, the RoI heads, and the box predictor. The trunk stays frozen. The constructor receives 81 classes: 80 foreground categories plus background.
- **Detector limits.** Train pre-NMS 2000 and post-NMS 1000. Test pre-NMS 1000 and post-NMS 500. Score threshold `0.05`. Box NMS `0.5`. 100 detections per image.
- **Loss.** Sum of RPN objectness, RPN box regression, classification, and box regression.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.001`, absolute, at effective batch 2.
- **Schedule.** 12 epochs. Warmup for 1 epoch, then cosine.
- **Seed.** One trial at the proposed seed `0`.
- **Checkpoint rule.** Epoch with the highest validation bbox AP.
- **Evaluation.** `COCOeval` on bbox, one view per image, over the val2017 ids in the loader.
- **Metrics.** Primary bbox AP (%) = `100 * COCOeval.stats[0]` (AP@[0.50:0.95]). Secondary AP50 (%) = `100 * stats[1]`.

#### ADE20K

- **Input.** Bicubic resize to 448×448. Nearest-neighbor resize of the label to 448. Label `0` becomes ignore `255`. Labels `1–150` become `0–149`.
- **Feature.** Final patch grid.
- **Head.** `LinearSegHead`: 1×1 conv to 150, bilinear upsample, `align_corners=false`.
- **Loss.** Pixel cross-entropy, `ignore_index=255`.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`.
- **Metrics.** mIoU (%) primary, pixel accuracy (%) secondary.

#### NYUv2 metric depth

- **Input.** Bicubic resize to 448×448. Depth is bilinearly resized to 448.
- **Split.** `splits.mat` `trainNdxs` for train and `testNdxs` for validation.
- **Feature.** Final patch grid, passed through GroupNorm with 1 group and `affine=false`, then a 1×1 conv and `softplus`, then bilinear upsample.
- **Loss.** L1 on valid pixels, depth in `(0.1, 10.0)` m. The training loss is L1, not SiLog.
- **Optimizer.** Adam, learning rate `0.01`, weight decay `1e-4`.
- **Evaluation.** `compute_depth_metrics` compares the prediction with metric depth. No per-image median scale.
- **Metrics.** RMSE (m) primary, AbsRel secondary, δ1 secondary.

#### Something-Something v2

- **Input.** 16 frames. Decode the video, take index `fi = clip(int((i + 0.5) * max(1, n/16)) + j, 0, n-1)` for `i = 0..15`, where `j` is an integer in `{-1, 0, 1}` from `random.Random(index ^ 0xA1B2C3D4)`. Each frame is a shorter-side resize to 256 and a 224 center crop. No horizontal flip. A failed decode is a black clip.
- **Feature.** Spatial mean per frame, then mean over the 16 frames.
- **Head.** `Linear(D, 174)`.
- **Loss.** Cross-entropy.
- **Optimizer.** SGD, momentum `0.9`, weight decay `0`.
- **Evaluation.** One clip per validation video.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

### Execution and seed policy

Use the linear `qsub_*_downstream.sh` scripts, not the `*_attentive.sh` scripts. Override `--epochs 50`, `--lr 0.01`, and the batch sizes in the summary table. Set the process seed to the proposed value `0` before the run.

V-JEPA 2.1 entry point for the linear ImageNet run is `06_vjepa21/scripts/qsub_imagenet1k_vitl_downstream.sh` and the ViT-G twin, with those overrides. The checked-in ViT-L launcher text says 20 epochs and batch 128; the candidate run uses 50 epochs and batch 64.

### Evaluation and reporting

Report the best epoch beside each primary number. The COCO column is bbox AP (%) = AP@[0.50:0.95], with AP50 (%) secondary. NYUv2 is metric RMSE in metres.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Readout | Linear or 1×1 on ImageNet, ADE20K, NYUv2, and SSv2. Faster R-CNN on COCO. Trunk frozen | `downstream_eval.py` `run_classification`, `run_segmentation`, `run_depth`. Attentive probes stay excluded | Newly selected unification of readout |
| COCO task | bbox AP (%), AP@[0.50:0.95]. Faster R-CNN, 12 epochs, batch 2, LR `0.001`, WD `1e-4`, 1-epoch warmup, cosine, best validation epoch, 224 square | `3DFM/_common/coco_probe.py`; `INITIAL_STEP3_3DFM_v1.md`; `INITIAL_STEP3_4DFM_v1.md` | Supplied detection reconstruction; run matching pending |
| Schedule | 50 epochs, LR `0.01`, warmup 5, on ImageNet, ADE20K, NYUv2, and SSv2 | Most `qsub_*_downstream.sh` echoes use `epochs=50` and `lr=0.01`. V-JEPA 2.1 ViT-L launcher uses 20 epochs and batch 128. V-JEPA 2 ImageNet launcher uses 7 epochs | Newly selected unification |
| Batches | 64 on ImageNet and SSv2, 32 on ADE20K and NYUv2, 2 on COCO | ADE launcher for V-JEPA 2.1-AC echoes `batch=32`. COCO batch 2 is the 3DFM/4DFM detector batch | Newly selected unification |
| Frames | 16 | `SSv2ActionDataset` default `num_frames=16` | Observed loader default |
| Resolution | 224 center crop on ImageNet and SSv2; 224 square on COCO; 448 square on ADE20K and NYUv2 | `_SSV2_IMG_TF` and the ImageNet 224 crop; `_ADE_IMG_SIZE = 448` and `_NYU_IMG_SIZE = 448`. COCO square 224 matches `INITIAL_STEP3_4DFM_v1.md` | Observed loader setting for ImageNet, ADE20K, NYUv2, and SSv2. COCO geometry is a unification |
| Depth | Masked L1, metric RMSE on `(0.1, 10)` m | `run_depth` and `compute_depth_metrics` | Observed main-run setting |
| Score epoch | Best validation primary metric | Returned fields `best_epoch`, `best_metric`, `best_rmse` | Observed main-run setting |
| Seed | Proposed value `0`, one trial | `downstream_eval.py` and `06_vjepa21/scripts/qsub_imagenet1k_vitl_downstream.sh` do not set a seed | Proposal. Evidence gap |

### Evidence references

- `methods_step3/VideoSSL/README.md`
- `methods_step3/VideoSSL/_common/downstream_eval.py`
- `methods_step3/VideoSSL/_common/data_downstream.py`
- `methods_step3/VideoSSL/06_vjepa21/scripts/qsub_imagenet1k_vitl_downstream.sh`
- Linear `results.json` files under `methods_step3/VideoSSL/<method>/results/`
- Detection recipe: `methods_step3/3DFM/_common/coco_probe.py` and `methods_step3/protocol/INITIAL_STEP3_3DFM_v1.md`
