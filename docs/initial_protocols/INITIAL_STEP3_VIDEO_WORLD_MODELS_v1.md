# INITIAL_STEP3_VIDEO_WORLD_MODELS_v1: historical evidence and candidate reconstruction

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

This reconstruction concerns an older Cosmos3 Super adapter, not all video world models. The inspected historical adapter uses ImageNet normalization and last_hidden_state readout. Inspected shared readers include median depth alignment and AdamW calls without an explicit weight_decay argument. These source facts do not identify every reported run.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Merger, MoT, DiT, VAE | Not loaded and not scored | Historical `checkpoint_identity.note`; `out_hidden_size` 5120 stored as `merger_out_hidden_size` only | Observed main-run setting |
| Normalization and patch order | ImageNet mean/std; row-major unfold; temporal repeat of 2; no block-major unshuffle | `adapter.py.pre_official_preproc.bak` `_transform` and `_extract_patches` | Observed main-run setting |
| Classification schedule | AdamW `1e-3`, WD `1e-4`, 100 epochs, cosine, last epoch, seeds 42/123/456 | `core20/registry.py` `PROBE["image_cls"]` and `train_linear_cls` | Observed main-run setting |
| ADE and NYUv2 weight decay | Executed AdamW default `0.01` | `run_ade` and `run_nyu` omit `weight_decay` | Observed main-run setting |
| Depth metric | Median-aligned RMSE | `depth_metrics` scales by `target.median()/pred.median()` before RMSE | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

The historical pre-merger readout must not be equated with the current final-merger representation. The supplied wording about spatially merged tokens needs reconciliation with the exact encoder output/version; it is not resolved here. Completion claims and all seed-to-result correspondences below remain to be matched to run records.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Later official preprocessing | Excluded | Current `adapter.py` and checkpoint id `nvidia_Cosmos3-Super_vision_encoder_official_preproc` | Excluded later revision |
| Other VideoGen trunks | Outside this interface | Wan, LTX, Hunyuan, and tokenizer probes read DiT or tokenizer states; `STEP3_CANONICAL_MANIFEST.md` records ADE as scene Acc@1 for that earlier table | Compatibility gap |

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

This document is a candidate specification reconstructed from the initial Cosmos 3 Super campaign. It is not a claim that every stored score already follows this file. The scores that follow it are the completed `r2` cells under checkpoint id `nvidia_Cosmos3-Super_vision_encoder`. The later checkpoint id `nvidia_Cosmos3-Super_vision_encoder_official_preproc` and the current `adapter.py` are a different preprocessing contract and are not this protocol.

### Purpose and category scope

Evaluate the frozen vision encoder of a multimodal world model. The evaluated tensor is the encoder’s pre-merger `last_hidden_state`. It is not the vision-merger output, not a MoT or DiT hidden state, not a VAE latent, and not a timestep-conditioned denoiser feature.

The completed initial cells are the five Core-20 probes of Cosmos 3 Super. Wan2.2, LTX-2, HunyuanVideo-1.5, and the Cosmos tokenizer extract denoiser or tokenizer states. Those runs, and the earlier VideoGen scene-classification and global-depth files under `VideoGen/output/`, do not satisfy this interface and stay out of the table.

### Included models and datasets

| Model | Checkpoint | Feature |
|---|---|---|
| Cosmos 3 Super | `nvidia/Cosmos3-Super`, directory `methods_step3/VideoGen/checkpoints/cosmos3/Cosmos3-Super`, vision tower `vision_encoder/` | `Qwen3VLVisionModel.last_hidden_state`, width 1152 |

Recorded weight hash of `vision_encoder/model.safetensors`: `3bdba32cec6f4f570a150f0a0e5ec60115fda49ab15873ecc86473a3da824b1e`. Vision config: `depth` 27, `hidden_size` 1152, `out_hidden_size` 5120, `patch_size` 16, `temporal_patch_size` 2, `spatial_merge_size` 2. The probe uses `hidden_size`. It does not use `out_hidden_size`.

| Task | Dataset | Split | Root |
|---|---|---|---|
| Image classification | ImageNet-1k | official `train` / `val` | `${LOCAL_REFERENCE_PATH}` |
| Object detection | COCO 2017 | train2017 / val2017 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K Challenge 2016 | training 20,210 / validation 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Monocular depth | NYUv2 labeled | `labeled/splits.mat` `trainNdxs` / `testNdxs`, converted from 1-based to 0-based | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | `labels/train.json` / `labels/validation.json` | `${LOCAL_REFERENCE_PATH}` |

COCO here is box detection. The primary metric is bbox AP (%), AP@[0.50:0.95]. NYUv2 RMSE here is median-aligned RMSE.

### Global rules

- Load only `Qwen3VLVisionModel` from `vision_encoder/`. Freeze every parameter. Keep the tower in `eval()`. Forward in bfloat16. Cast features to float32 before the head.
- Use the historical adapter `methods_step3/VideoGen/Cosmos3-Super/adapter.py.pre_official_preproc.bak`. Do not load the current `adapter.py`.
- Use a 448×448 square input. Record `input_size` 448 and `input_geometry` `square squash at the model's declared img_size`.
- Apply ImageNet normalization inside the adapter: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- Learning rates below are absolute at the stated head batch or dense batch. Do not rescale them. Do not accumulate gradients.
- Image classification and SSv2 report the mean over seeds `42`, `123`, and `456` of the score at the last scheduled epoch. ADE20K and NYUv2 use one trial and the last-epoch rule. COCO uses one trial and the epoch with the highest validation bbox AP.
- Image classification, SSv2, ADE20K, and NYUv2 do not select a checkpoint on the validation split. The unused `best` variable in `train_linear_cls` is not the score.

### Common representation interface

Implementation: historical `Cosmos3SuperAdapter` in `adapter.py.pre_official_preproc.bak`, driven by `core20/run_cell.py`.

- **Checkpoint.** `nvidia/Cosmos3-Super` vision encoder. Checkpoint id `nvidia_Cosmos3-Super_vision_encoder`. Initialize classification and dense heads as specified per task. Do not load MoT, DiT, or VAE weights.
- **Included components.** The Qwen3-VL vision tower through the block that emits `last_hidden_state`. Exclude the merger projection to 5120, the language model, and the generative stack.
- **Frozen scope.** All vision-tower parameters. The task head is the only trainable module.
- **Feature stage.** `last_hidden_state` before the merger. Semantic content: one vector per spatially merged token. Width 1152. The 5120-d merger width is metadata only (`merger_out_hidden_size`).
- **Global tensor.** Mean over the token axis, shape `(B, 1152)`.
- **Dense tensor.** The same tokens reshaped to `(B, 1152, h, w)`. The historical code takes `h = w = int(n ** 0.5)` and does not unshuffle a 2×2 block-major order into raster order.
- **Special tokens.** The tower output used here has no class token to drop.
- **Patch packing.** Row-major `unfold` on height, then width, patch 16. Each spatial patch is repeated `temporal_patch_size` (2) times along a dummy time axis and flattened to `3 * 2 * 16 * 16`. `grid_thw` is `(1, H/16, W/16)` per image. This is not Qwen2-VL block-major 2×2 merge order.
- **Preprocessing.** The dataset resizes with PIL bilinear (`resample` 2) to a 448 square (`core20/datasets.py` `cpu_to_tensor`). The adapter then applies a bicubic resize to 448, a center crop of 448, and ImageNet normalization. SSv2 applies that image preprocessing independently to each frame (`extract_split`).
- **Feature normalization before the head.** Image classification and SSv2 L2-normalize the 1152-d vector (`F.normalize`, `dim=-1`). ADE20K and NYUv2 L2-normalize across channels (`F.normalize`, `dim=1`). COCO uses the dense map with no extra L2 normalization.
- **Video.** 8 frames, indices `linspace(0, n-1, 8)`. Per-frame global vectors, then a mean over time. One vector per clip.
- **Head boundary.** The linear layer, the 1×1 convolution, or the Faster R-CNN heads are the first trainable parameters. The vision tower stays frozen.

A trunk that emits a denoiser state, a VAE latent, or the 5120-d merger vector does not satisfy this interface. Do not place that tensor in these tables.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | 448 square | linear, 1000-way, zero init | AdamW | 100 epochs, cosine | last epoch | Top-1 (%) |
| COCO 2017 | 448 square, detector keeps that square | Faster R-CNN | SGD | 12 epochs, warmup then cosine | best validation bbox AP | bbox AP (%) |
| ADE20K | 448 square | 1×1 conv, 150-way, zero init | AdamW | 20 epochs, constant LR | last epoch | mIoU (%) |
| NYUv2 | 448 square | 1×1 conv, softplus, zero init | AdamW | 30 epochs, constant LR | last epoch | median-aligned RMSE (m) |
| SSv2 | 448 square, 8 frames | linear, 174-way, zero init | AdamW | 100 epochs, cosine | last epoch | Top-1 (%) |

### Detailed task recipes

Image classification, ADE20K, NYUv2, and SSv2 use AdamW defaults from `core20/run_cell.py`: betas `(0.9, 0.999)`, `eps` `1e-8`. Those tasks have no warmup and no label smoothing. Feature extraction for them uses the registry batch size 8. COCO uses the SGD detector recipe below.

#### ImageNet-1k

- **Protocol name stored in `run_meta.json`.** `atlas_cls_linear_v1`.
- **Feature.** Global mean of pre-merger tokens, then L2 normalization.
- **Head.** `Linear(1152, 1000)`. Set weight and bias to zero. Head batch 1024.
- **Loss.** Cross-entropy.
- **Optimizer.** AdamW, learning rate `0.001`, weight decay `0.0001`.
- **Schedule.** 100 epochs. `CosineAnnealingLR` with `T_max=100` and `eta_min=0`, stepped once after each epoch.
- **Seeds.** `42`, `123`, `456`.
- **Checkpoint rule.** The weights after the last epoch. Evaluate once.
- **Metrics.** Top-1 (%) primary. Report the mean and the sample standard deviation across the three seeds (`_mean_ci`, divisor `n-1`). Top-5 (%) per seed is secondary.

#### COCO 2017 detection

- **Split.** train2017 and val2017.
- **Target.** Instance boxes for the 80 COCO categories. Drop crowd boxes. Drop boxes whose width or height is at most 1 pixel.
- **Input.** 448×448 square. Faster R-CNN uses `min_size=448` and `max_size=448`. No horizontal flip. ImageNet normalization from the historical adapter.
- **Feature.** Pre-merger dense map `(B, 1152, h, w)` as the single scale `"0"`. No global mean and no L2 normalization.
- **Head.** Torchvision Faster R-CNN. A trainable 1×1 conv maps 1152 channels to 256. Anchors `(32, 64, 128, 256, 512)` and ratios `(0.5, 1.0, 2.0)`. `MultiScaleRoIAlign` on feature name `"0"`, output 7×7, sampling ratio 2. Train that 1×1 conv, the RPN, the RoI heads, and the box predictor. The vision tower stays frozen. The constructor receives 81 classes: 80 foreground categories plus background.
- **Detector limits.** Train pre-NMS 2000 and post-NMS 1000. Test pre-NMS 1000 and post-NMS 500. Score threshold `0.05`. Box NMS `0.5`. 100 detections per image.
- **Loss.** Sum of RPN objectness, RPN box regression, classification, and box regression.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.001`, absolute, at effective batch 2.
- **Schedule.** 12 epochs. Warmup for 1 epoch, then cosine. Forward the frozen tower in bfloat16.
- **Seed.** One trial, seed `0`.
- **Checkpoint rule.** Epoch with the highest validation bbox AP.
- **Evaluation.** `COCOeval` on bbox, one view per image, over the val2017 ids in the loader.
- **Metrics.** Primary bbox AP (%) = `100 * COCOeval.stats[0]` (AP@[0.50:0.95]). Secondary AP50 (%) = `100 * stats[1]`.

#### ADE20K

- **Protocol name.** `ADE20K_DENSE_LINEAR_SEG_v1`.
- **Labels.** PNG values `0` become ignore `255`. Values `1–150` become classes `0–149`. Values above 150 become `255`. Nearest-neighbor resize of the label to 448.
- **Feature.** Dense pre-merger map, channel-normalized.
- **Head.** `Conv2d(1152, 150, kernel_size=1)`, weight and bias zero. Bilinear upsample to the label size, `align_corners=false`.
- **Loss.** Pixel cross-entropy, `ignore_index=255`.
- **Optimizer.** AdamW, learning rate `0.001`, executed weight decay `0.01` (argument omitted). Dense batch 8, `drop_last=true` on the train loader.
- **Schedule.** 20 epochs. Constant learning rate.
- **Seed.** One trial. Registry seed `42`. `run_ade` does not call `manual_seed` again.
- **Checkpoint rule.** Last epoch. One validation pass.
- **Metrics.** mIoU (%) primary, pixel accuracy (%) secondary. Both are means of per-batch scores. A batch mIoU averages IoU over classes with nonzero union in that batch.

#### NYUv2 median-aligned depth

- **Protocol name.** `NYU_DENSE_DEPTH_v1`.
- **Split.** `labeled/splits.mat` keys `trainNdxs` and `testNdxs`. Subtract 1 to obtain 0-based indices into `labeled/nyu_depth_v2_labeled.mat`.
- **Target.** Metric depth in metres, bilinearly resized to 448. A pixel is valid when depth is finite and lies in `(0.1, 10.0)` m.
- **Feature.** Dense pre-merger map, channel-normalized.
- **Head.** `Conv2d(1152, 1, kernel_size=1)`, weight and bias zero. Bilinear upsample, `align_corners=false`, then `softplus`.
- **Loss.** Masked L1 in metres. The denominator is the number of valid pixels.
- **Optimizer.** AdamW, learning rate `0.001`, executed weight decay `0.01`. Dense batch 8, `drop_last=true` on the train loader. `num_workers=0`.
- **Schedule.** 30 epochs. Constant learning rate.
- **Seed.** One trial at registry seed `42`.
- **Evaluation.** On each validation image, multiply the prediction by `target.median() / pred.median()` before the error. Then compute RMSE in metres and AbsRel.
- **Checkpoint rule.** Last epoch.
- **Metrics.** Median-aligned RMSE (m) primary. Median-aligned AbsRel secondary. Each is the mean of per-batch values.

#### Something-Something v2

- **Protocol name.** `SSV2_VIDEO_REPAIR_v1`.
- **Input.** 8 frames from `linspace` over the decoded video. Each frame is a 448 square. Missing or unreadable videos contribute a zero clip and are counted in `n_video_empty`.
- **Feature.** Per-frame global mean, L2-normalized, then the mean over the 8 frames.
- **Head, loss, optimizer, schedule, seeds, and metrics.** Same numbers as ImageNet-1k, with 174 classes: zero-init linear head, cross-entropy, AdamW learning rate `0.001`, weight decay `0.0001`, head batch 1024, 100 epochs, cosine to `0`, last-epoch Top-1 (%) mean and sample standard deviation over seeds `42`, `123`, `456`. Top-5 (%) is secondary.
- **Views.** One 8-frame clip per video. No multi-clip average.

### Execution and seed policy

```bash
python ${LOCAL_REFERENCE_PATH} \
  --method cosmos3_super --dataset imagenet_1k --device cuda
```

Repeat with `--dataset` in `ade20k`, `nyu_depth_v2`, `something_something_v2`. The adapter loaded by that command must be the historical ImageNet-normalized, row-major adapter. Completed cells record `canary: false`, `precision: bf16`, and `run_id: r2`. COCO is the Faster R-CNN recipe in this file, on that same frozen 1152-d map.

### Evaluation and reporting

Write `run_meta.json` and `result.json` under `VideoGen/Cosmos3-Super/results/<dataset>/Cosmos3-Super/nvidia_Cosmos3-Super_vision_encoder/r2/`. Record `feature_dim` 1152, `merger_out_hidden_size` 5120, the sha256 above, input 448, and the protocol name of the task. Report Top-1, Top-5, bbox AP, AP50, mIoU, and pixel accuracy in percent. Report median-aligned RMSE in metres and AbsRel as a unitless ratio after the median scale. State the seed list, or the single seed, beside the number.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Evaluated component | Pre-merger `last_hidden_state`, width 1152 | `vision_encoder/config.json` `hidden_size` 1152; historical adapter sets `_feature_dim` from `hidden_size`; `r2` `checkpoint_identity.feature_dim` is 1152 | Observed main-run setting |
| Merger, MoT, DiT, VAE | Not loaded and not scored | Historical `checkpoint_identity.note`; `out_hidden_size` 5120 stored as `merger_out_hidden_size` only | Observed main-run setting |
| Normalization and patch order | ImageNet mean/std; row-major unfold; temporal repeat of 2; no block-major unshuffle | `adapter.py.pre_official_preproc.bak` `_transform` and `_extract_patches` | Observed main-run setting |
| Later official preprocessing | Excluded | Current `adapter.py` and checkpoint id `nvidia_Cosmos3-Super_vision_encoder_official_preproc` | Excluded later revision |
| Classification schedule | AdamW `1e-3`, WD `1e-4`, 100 epochs, cosine, last epoch, seeds 42/123/456 | `core20/registry.py` `PROBE["image_cls"]` and `train_linear_cls` | Observed main-run setting |
| COCO task | bbox AP (%), AP@[0.50:0.95]. Faster R-CNN, 12 epochs, batch 2, LR `0.001`, WD `1e-4`, 1-epoch warmup, cosine, best validation epoch, 448 square | `3DFM/_common/coco_probe.py`; `INITIAL_STEP3_3DFM_v1.md`; `INITIAL_STEP3_4DFM_v1.md` | Supplied detection reconstruction; run matching pending |
| ADE and NYUv2 weight decay | Executed AdamW default `0.01` | `run_ade` and `run_nyu` omit `weight_decay` | Observed main-run setting |
| Depth metric | Median-aligned RMSE | `depth_metrics` scales by `target.median()/pred.median()` before RMSE | Observed main-run setting |
| Other VideoGen trunks | Outside this interface | Wan, LTX, Hunyuan, and tokenizer probes read DiT or tokenizer states; `STEP3_candidate_MANIFEST.md` records ADE as scene Acc@1 for that earlier table | Compatibility gap |

### Evidence references

- `methods_step3/VideoGen/Cosmos3-Super/adapter.py.pre_official_preproc.bak`
- `methods_step3/VideoGen/checkpoints/cosmos3/Cosmos3-Super/vision_encoder/config.json`
- `methods_step3/core20/registry.py` keys `DATASETS`, `PROBE`, and the historical note on `METHODS["cosmos3_super"]`
- `methods_step3/core20/run_cell.py` functions `extract_split`, `train_linear_cls`, `run_ade`, `run_nyu`, `depth_metrics`
- Detection recipe: `methods_step3/3DFM/_common/coco_probe.py` and `methods_step3/protocol/INITIAL_STEP3_3DFM_v1.md`
- `methods_step3/core20/datasets.py`
- `methods_step3/VideoGen/Cosmos3-Super/results/<dataset>/Cosmos3-Super/nvidia_Cosmos3-Super_vision_encoder/r2/run_meta.json`
