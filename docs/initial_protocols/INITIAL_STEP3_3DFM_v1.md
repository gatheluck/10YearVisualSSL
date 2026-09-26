# INITIAL_STEP3_3DFM_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

The reconstruction records model-dependent native resolutions, frozen spatial features and historical learning-rate selection. Its evidence table attributes conditions to configurations and result records; those attributions are not a full independent result audit.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Native square size | The completed ImageNet-1k `image_size` per variant | `results/imagenet1k_linear_*/results.json` is global; dense JSON `config.image_size` matches the table | Observed main-run setting |
| ADE / NYUv2 / COCO epoch rule | Best validation primary metric | JSON keys `best_miou`, `best_rmse`, `best_mAP` | Observed main-run setting |
| Depth loss | SiLog, λ `0.85`; metric RMSE | `3DFM/_common/dense_heads.py` `silog_loss`; `dense_probes.py` `run_depth_probe` | Observed main-run setting |
| SSv2 length | 8 frames | `3DFM/README.md` action section | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

Removing the learning-rate sweep and selecting a shared dense batch size are new unifications. They must not be described as the settings of every historical score.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Classification LR | `0.1` at head batch 1024 | Sweep grid in `linear_probe.py`; ImageNet-1k `best_lr` is `0.1` for CroCo, DUSt3R, VGGT, DA3, Pi3, and MASt3R | Newly selected unification (sweep removed) |
| ADE / NYUv2 batch | Effective batch 32, LR not rescaled | Completed `batch_size_per_gpu` varies; `base_lr` does not | Newly selected unification |
| VGGT-Omega scene / camera / dense heads | Excluded | Separate result directories `*scene*`, `*camera*`, `*dense*` | Out of scope |

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

This document is a candidate specification reconstructed from the initial campaign under `methods_step3/3DFM/`. It is not a claim that every stored score already follows this file. Stored ImageNet and SSv2 scores selected a learning rate from a sweep. This file fixes one learning rate. Scene, camera, and dense-head ablations of VGGT-Omega are outside the table.

### Purpose and category scope

Evaluate released 3D foundation models as frozen visual trunks on the five tasks below. Do not rerun geometric pretraining. Use one feature stage: the final aggregator or encoder patch tokens, with camera and register tokens removed.

### Included models and datasets

Include a model only when an ImageNet-1k `results.json` exists for its default variant.

| Directory | Checkpoint identifier | Native square size used by the completed ImageNet-1k run |
|---|---|---|
| `33_croco` | CroCo v2 ViT-L, variant `croco_v2_vitl_base` | 224 |
| `34_dust3r` | `dust3r_vitl_512_dpt` | 512 |
| `35_vggt` | `facebook/VGGT-1B`, variant `vggt_1b` | 518 |
| `36_vggtomega` | `facebook/VGGT-Omega`, file `vggt_omega_1b_512.pt`, variant `vggt_omega_1b_512` | 512 |
| `37_da3` | DA3-Large, variant `da3_large` | 224 |
| `38_pi3` | Pi3, variant `pi3x` | 224 |
| `39_mast3r` | MASt3R ViT-L, variant `mast3r_vitl_512` | 512 |

VGGT-Omega’s Hugging Face repository id is `facebook/VGGT-Omega`. The candidate feature is the aggregator patch-token backbone (`VGGTOmegaBackbone`), not `VGGTOmegaSceneBackbone`, `VGGTOmegaCameraBackbone`, or `VGGTOmegaDenseBackbone`.

ImageNet-100 runs are pilots and stay out of the table. The text-alignment checkpoint `vggt_omega_1b_256_text.pt` stays out of the table.

| Task | Dataset | Split | Root wired in `3DFM/_common/data_*.py` |
|---|---|---|---|
| Image classification | ImageNet-1k | official train / validation | `${LOCAL_REFERENCE_PATH}` |
| Detection | COCO 2017 | train2017 / val2017 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K | 20,210 / 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Metric depth | NYUv2 | Eigen-style valid range 0.001–10 m; loader split used by the dense probe | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | official train / validation | `${LOCAL_REFERENCE_PATH}` |

### Global rules

- Load the released checkpoint. Freeze it. Run it in `eval()` under bfloat16 autocast for the forward, and compute the loss in float32.
- Use the native square size in the table. Record that size.
- Use the checkpoint normalization exposed as `backbone.mean` and `backbone.std`. Record the triple. VGGT-Omega’s wrapper uses mean `(0, 0, 0)` and std `(1, 1, 1)`, which leaves `[0, 1]` RGB unchanged.
- One trial, seed `0`.
- Classification learning rate is the fixed value below, not a sweep.
- Dense and detection scores are the best validation epoch inside the scheduled run. Record the epoch index next to the score.
- Do not scale the listed learning rates. They are absolute at the stated effective batch.

### Common representation interface

- **Checkpoint.** The default variant file in the table, loaded by that method’s `build_*_backbone`.
- **Included components.** The geometric encoder or aggregator through its last cached block. Exclude DPT heads, camera heads, and text-alignment heads.
- **Frozen scope.** Every trunk parameter. No gradient enters the trunk.
- **Feature stage.** Last cached aggregator or encoder layer. For VGGT-Omega this is the last non-empty entry of `aggregated_tokens_list` (the implementation caches layers 4, 11, 17, and 23 and keeps the last).
- **Tensor interface.** Patch tokens only, shape `(B, N, D)` then reshaped to `(B, D, h, h)`. Global tasks mean-pool the patch tokens. Dense tasks keep the grid. Detection uses that grid as the backbone feature map.
- **Prefix tokens.** Drop tokens before `patch_token_start`. On VGGT-Omega that index is 17 (1 camera token + 16 registers). Those tokens are not the feature.
- **Width.** Record `D`. CroCo and DUSt3R and MASt3R main runs store 1024. VGGT and the VGGT-Omega aggregator store `2 * embed_dim` (2048 when `embed_dim` is 1024).
- **Order.** Row-major patch grid. `h = round(sqrt(N))` and the probe rejects a non-square token count.
- **Pooling.** Mean over patch tokens for ImageNet and SSv2. No L2 normalization on the cached vector.
- **Head boundary.** `Linear` for classification, `LinearSegHead` / `LinearDepthHead` for dense tasks, and the Faster R-CNN heads for detection.

A model that only exposes a global scene token, a camera-head trunk, or a DPT fusion map does not satisfy this interface. Those VGGT-Omega ablations are not an alternate protocol.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | native square | linear, 1000-way | SGD, Nesterov | 100 epochs, warmup cosine | best validation Top-1 | Top-1 (%) |
| COCO 2017 | native square, then detector resize | Faster R-CNN | SGD | 12 epochs, warmup cosine | best validation bbox AP | bbox AP (%) |
| ADE20K | native square | 1×1 conv, 150-way | SGD | 50 epochs, warmup cosine | best validation mIoU | mIoU (%) |
| NYUv2 | native square | 1×1 log-depth | AdamW | 50 epochs, warmup cosine | best validation RMSE | RMSE (m) |
| SSv2 | native square, 8 frames | linear, 174-way | SGD, Nesterov | 100 epochs, warmup cosine | best validation Top-1 | Top-1 (%) |

### Detailed task recipes

#### ImageNet-1k

- **Driver.** `3DFM/_common/linear_probe.py`.
- **Input.** Square resize to the native size. Normalization is the wrapper mean and std.
- **Feature.** Mean of final patch tokens, cached once.
- **Head.** `Linear(D, 1000)`, weight std `0.01`, zero bias. Head batch 1024 on the cached matrix.
- **Loss.** Cross-entropy, label smoothing `0`.
- **Optimizer.** SGD, momentum `0.9`, Nesterov enabled, weight decay `0`.
- **Learning rate.** `0.1`, absolute, at head batch 1024.
- **Schedule.** 100 epochs. Linear warmup for 10 epochs, then cosine to 0 over the remaining iterations (`WarmupCosineLR`).
- **Checkpoint rule.** The epoch with the highest validation Top-1. Report that epoch.
- **Metrics.** Top-1 (%) primary, Top-5 (%) at that epoch secondary.

#### COCO 2017 detection

- **Config recorded by every completed 3DFM detection JSON.** 12 epochs, `batch_size_per_gpu=2`, `base_lr=0.001`, weight decay `1e-4`, warmup 1 epoch, bfloat16.
- **Detector limits.** `rpn_pre_nms_top_n_train=2000`, `rpn_post_nms_top_n_train=1000`, `rpn_pre_nms_top_n_test=1000`, `rpn_post_nms_top_n_test=500`, `box_score_thresh=0.05`, `box_nms_thresh=0.5`, `box_detections_per_img=100`.
- **Classes.** 80 COCO categories (`num_classes=80` in the result files).
- **Feature.** Spatial patch grid. No global pool.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.001`, absolute, at effective batch 2.
- **Schedule.** 12 epochs, 1-epoch warmup, then cosine.
- **Checkpoint rule.** Epoch with the highest validation bbox AP.
- **Metrics.** Bbox AP (%) primary. This number is detection AP. Completed runs are near zero AP; report the value anyway.

#### ADE20K

- **Head.** `LinearSegHead`: 1×1 conv from `D` to 150, bilinear upsample to the label size, `align_corners=false`.
- **Loss.** Pixel cross-entropy, `ignore_index=255`.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.01`.
- **Batch.** Physical batch 32, accumulation 1, effective batch 32. Learning rate stays `0.01`. Completed runs also stored physical batches 4 and 256 at that same learning rate; those batches are not the candidate batch.
- **Schedule.** 50 epochs, warmup 2 epochs, cosine.
- **Input.** Native square size. No extra scale jitter beyond the loader resize to that square.
- **Checkpoint rule.** Epoch with the highest validation mIoU.
- **Metrics.** mIoU (%) primary. Pixel accuracy (%) secondary.

#### NYUv2 metric depth

- **Head.** `LinearDepthHead`: 1×1 conv predicting log-depth, exponentiated back to metres by the probe.
- **Loss.** Scale-invariant log loss `silog_loss` with `lam=0.85`, on pixels with depth in `[0.001, 10.0]` m.
- **Optimizer.** AdamW, learning rate `0.001`, weight decay `1e-4`.
- **Batch.** Physical batch 32, accumulation 1, effective batch 32. Learning rate stays `0.001`.
- **Schedule.** 50 epochs, warmup 2 epochs, cosine.
- **Checkpoint rule.** Epoch with the lowest validation RMSE.
- **Evaluation.** RMSE is metric RMSE in metres on the valid range. It is not a median-aligned RMSE. AbsRel at that same epoch is secondary.

#### Something-Something v2

- **Input.** 8 frames, native square spatial size, temporally consistent resize. Mean-pool each frame’s patch tokens, then mean-pool the 8 frame vectors.
- **Head and optimizer.** Same linear SGD recipe as ImageNet, 174 classes, weight decay `0`, Nesterov, head batch 1024, learning rate `0.1`.
- **Schedule.** 100 epochs, warmup 10 epochs, cosine.
- **Checkpoint rule.** Best validation Top-1.
- **Metrics.** Top-1 (%) primary, Top-5 (%) secondary.

### Execution and seed policy

Submit through `methods_step3/3DFM/submit_all_5tasks.sh` with the native `IMAGE_SIZE` of the variant and seed `0`. ImageNet entry points are `eval_imagenet_linear.py`. Dense entry points are `eval_dense_probe.py`.

One seed. Do not average ablations that change `backbone_type`.

### Evaluation and reporting

State the native size, `D`, normalization mean and std, effective batch, fixed learning rate, and the epoch that won the primary metric. Label the COCO column “bbox AP (%)”. Label the NYUv2 column “RMSE (m)”.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| VGGT-Omega id and feature | `facebook/VGGT-Omega`, `vggt_omega_1b_512.pt`, aggregator patch tokens | `36_vggtomega/backbone/vggtomega_backbone.py` `hf_repo` and `VGGTOmegaBackbone` | Observed main-run setting |
| Native square size | The completed ImageNet-1k `image_size` per variant | `results/imagenet1k_linear_*/results.json` is global; dense JSON `config.image_size` matches the table | Observed main-run setting |
| Classification LR | `0.1` at head batch 1024 | Sweep grid in `linear_probe.py`; ImageNet-1k `best_lr` is `0.1` for CroCo, DUSt3R, VGGT, DA3, Pi3, and MASt3R | Newly selected unification (sweep removed) |
| ADE / NYUv2 batch | Effective batch 32, LR not rescaled | Completed `batch_size_per_gpu` varies; `base_lr` does not | Newly selected unification |
| ADE / NYUv2 / COCO epoch rule | Best validation primary metric | JSON keys `best_miou`, `best_rmse`, `best_mAP` | Observed main-run setting |
| Depth loss | SiLog, λ `0.85`; metric RMSE | `3DFM/_common/dense_heads.py` `silog_loss`; `dense_probes.py` `run_depth_probe` | Observed main-run setting |
| SSv2 length | 8 frames | `3DFM/README.md` action section | Observed main-run setting |
| VGGT-Omega scene / camera / dense heads | Excluded | Separate result directories `*scene*`, `*camera*`, `*dense*` | Out of scope |

### Evidence references

- `methods_step3/3DFM/README.md`
- `methods_step3/3DFM/_common/linear_probe.py`
- `methods_step3/3DFM/_common/dense_probes.py`
- `methods_step3/3DFM/_common/dense_heads.py`
- `methods_step3/3DFM/36_vggtomega/backbone/vggtomega_backbone.py`
- `methods_step3/3DFM/36_vggtomega/eval_imagenet_linear.py`
- Completed JSON under `methods_step3/3DFM/<method>/results/`
