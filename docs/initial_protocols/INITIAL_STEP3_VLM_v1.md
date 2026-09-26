# INITIAL_STEP3_VLM_v1: historical evidence and candidate reconstruction

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

This reconstruction concerns the earlier SigLIP2 So400m/14 at 384 pixels with width 1152. It is distinct from the later SigLIP2-G provider. The shared historical readers include median-aligned depth and optimizer defaults; exact run-to-score correspondence is not independently established here.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| Global feature | MAP pool from `get_image_features` | `SigLIPAdapter.extract_features` | Observed main-run setting |
| Dense feature | `vision_model.last_hidden_state` patch grid | `SigLIPAdapter.extract_dense_features` | Observed main-run setting |
| Normalization | mean = std = `(0.5, 0.5, 0.5)` | `SIGLIP_MEAN` and `SIGLIP_STD` in `VLM/SigLIP/adapter.py` | Observed main-run setting |
| Non-COCO probe schedule | The non-COCO task rows in this file | `core20/registry.py` `PROBE` and the matching functions in `run_cell.py` | Observed main-run setting |
| ADE and NYUv2 weight decay | Executed AdamW default `0.01` | `run_ade` and `run_nyu` pass `lr` and omit `weight_decay`. The `PROBE` value `0.0` is not applied | Observed main-run setting |
| Depth metric | Median-aligned RMSE | `depth_metrics` | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

Do not apply this model identity or readout to later giant-model results. The supplied recipe is an attributed reconstruction; complete historical seed and score correspondence still needs the original run records.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| CLIP, SigLIP 2 Giant, and the seven later VLMs | Outside this completed campaign | CLIP has a registry row and no matching result tree. Giant and the seven adapters are later `METHODS` entries | Compatibility gap for this protocol’s completed set |

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

This document is a candidate specification reconstructed from the initial VLM campaign. It is not a claim that every later VLM score already follows this file. The completed cells are `r1` and `r2` under `VLM/SigLIP/results/` for checkpoint id `google_siglip2-so400m-patch14-384`. CLIP has a Core-20 registry entry and no completed five-task result directory in that campaign. SigLIP 2 Giant and the seven later VLM adapters are a later registry revision.

### Purpose and category scope

Evaluate a frozen vision-language vision tower. The global vector is the tower’s MAP-pooled image embedding. The dense tensor is that tower’s final patch-token grid. Text-tower embeddings, contrastive logit scale, and captioning decoders are not the evaluated representation.

### Included models and datasets

| Model | Checkpoint | Width | Input |
|---|---|---|---|
| SigLIP 2 So400m/14 | `google/siglip2-so400m-patch14-384`, directory `methods_step3/VLM/SigLIP/checkpoints/google_siglip2-so400m-patch14-384` | 1152 | 384 |

Recorded weight hash of `model.safetensors`: `9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2`. Patch size 14. Precision fp16. The adapter class is `SigLIPAdapter` in `VLM/SigLIP/adapter.py`.

| Task | Dataset | Split | Root |
|---|---|---|---|
| Image classification | ImageNet-1k | official `train` / `val` | `${LOCAL_REFERENCE_PATH}` |
| Object detection | COCO 2017 | train2017 / val2017 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K Challenge 2016 | training 20,210 / validation 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Monocular depth | NYUv2 labeled | `labeled/splits.mat` `trainNdxs` / `testNdxs`, converted from 1-based to 0-based | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | `labels/train.json` / `labels/validation.json` | `${LOCAL_REFERENCE_PATH}` |

COCO here is box detection. The primary metric is bbox AP (%), AP@[0.50:0.95]. NYUv2 RMSE here is median-aligned RMSE.

### Global rules

- Load `google/siglip2-so400m-patch14-384` with `AutoModel`, `local_files_only=true`, dtype float16. Freeze every parameter. Keep the model in `eval()`. Cast features to float32 before the head.
- Use a 384×384 square input. Record `input_size` 384 and `input_geometry` `square squash at the model's declared img_size`.
- Normalize with mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`.
- Learning rates below are absolute at the stated head batch or dense batch. Do not rescale them. Do not accumulate gradients.
- Image classification and SSv2 report the mean over seeds `42`, `123`, and `456` of the last-epoch score. ADE20K and NYUv2 use one trial and the last-epoch score. COCO uses one trial and the epoch with the highest validation bbox AP.
- Image classification, SSv2, ADE20K, and NYUv2 do not select a checkpoint on the validation split.

### Common representation interface

- **Checkpoint.** The directory above. Checkpoint id `google_siglip2-so400m-patch14-384`. Initialize heads as specified per task.
- **Included components.** The vision tower used by `get_image_features` and by `vision_model`. Exclude the text tower.
- **Frozen scope.** All loaded parameters, including the MAP attention pool. The task head is the only trainable module.
- **Feature stage, global.** `get_image_features(pixel_values=...)`, the released MAP-pooled embedding, shape `(B, 1152)`.
- **Feature stage, dense.** `vision_model(...).last_hidden_state`, every token, reshaped to `(B, 1152, h, w)` with `h = w = int(N ** 0.5)`. There is no class token to drop. At patch 14 the grid side is the integer square root of the token count.
- **Special tokens.** None are concatenated or removed.
- **Pooling.** Image classification uses the MAP pool inside `get_image_features`. ADE20K and NYUv2 keep the patch grid. COCO keeps that same patch grid as detector scale `"0"`. Video mean-pools the per-frame MAP vectors over time.
- **Preprocessing.** The dataset resizes with PIL bilinear (`resample` 2) to a 384 square. The adapter then applies a bicubic resize to 384, a center crop of 384, and the `0.5` / `0.5` normalization. SSv2 applies that image preprocessing to each frame.
- **Feature normalization before the head.** Image classification and SSv2 L2-normalize the 1152-d vector (`F.normalize`, `dim=-1`). ADE20K and NYUv2 L2-normalize across channels (`F.normalize`, `dim=1`). COCO uses the dense map with no extra L2 normalization.
- **Video.** 8 frames, indices `linspace(0, n-1, 8)`. One MAP vector per frame, then the mean over time.
- **Head boundary.** The linear layer, the 1×1 convolution, or the Faster R-CNN heads are the first trainable parameters. The vision tower stays frozen.

A trunk that cannot emit the MAP-pooled `get_image_features` vector and a square `last_hidden_state` patch grid does not satisfy this interface.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | 384 square | linear, 1000-way, zero init | AdamW | 100 epochs, cosine | last epoch | Top-1 (%) |
| COCO 2017 | 384 square, detector keeps that square | Faster R-CNN | SGD | 12 epochs, warmup then cosine | best validation bbox AP | bbox AP (%) |
| ADE20K | 384 square | 1×1 conv, 150-way, zero init | AdamW | 20 epochs, constant LR | last epoch | mIoU (%) |
| NYUv2 | 384 square | 1×1 conv, softplus, zero init | AdamW | 30 epochs, constant LR | last epoch | median-aligned RMSE (m) |
| SSv2 | 384 square, 8 frames | linear, 174-way, zero init | AdamW | 100 epochs, cosine | last epoch | Top-1 (%) |

### Detailed task recipes

Image classification, ADE20K, NYUv2, and SSv2 use AdamW defaults from `core20/run_cell.py`: betas `(0.9, 0.999)`, `eps` `1e-8`. Those tasks have no warmup and no label smoothing. Feature extraction for them uses the registry batch size 32. COCO uses the SGD detector recipe below.

#### ImageNet-1k

- **Protocol name stored in `run_meta.json`.** `atlas_cls_linear_v1`.
- **Feature.** MAP-pooled image embedding, then L2 normalization.
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
- **Input.** 384×384 square. Faster R-CNN uses `min_size=384` and `max_size=384`. No horizontal flip. Normalization mean = std = `(0.5, 0.5, 0.5)`.
- **Feature.** Dense patch grid `(B, 1152, h, w)` from `vision_model.last_hidden_state`, as the single scale `"0"`. No MAP pool and no L2 normalization.
- **Head.** Torchvision Faster R-CNN. A trainable 1×1 conv maps 1152 channels to 256. Anchors `(32, 64, 128, 256, 512)` and ratios `(0.5, 1.0, 2.0)`. `MultiScaleRoIAlign` on feature name `"0"`, output 7×7, sampling ratio 2. Train that 1×1 conv, the RPN, the RoI heads, and the box predictor. The vision tower stays frozen. The constructor receives 81 classes: 80 foreground categories plus background.
- **Detector limits.** Train pre-NMS 2000 and post-NMS 1000. Test pre-NMS 1000 and post-NMS 500. Score threshold `0.05`. Box NMS `0.5`. 100 detections per image.
- **Loss.** Sum of RPN objectness, RPN box regression, classification, and box regression.
- **Optimizer.** SGD, momentum `0.9`, weight decay `1e-4`, learning rate `0.001`, absolute, at effective batch 2.
- **Schedule.** 12 epochs. Warmup for 1 epoch, then cosine. Forward the frozen tower in float16.
- **Seed.** One trial, seed `0`.
- **Checkpoint rule.** Epoch with the highest validation bbox AP.
- **Evaluation.** `COCOeval` on bbox, one view per image, over the val2017 ids in the loader.
- **Metrics.** Primary bbox AP (%) = `100 * COCOeval.stats[0]` (AP@[0.50:0.95]). Secondary AP50 (%) = `100 * stats[1]`.

#### ADE20K

- **Protocol name.** `ADE20K_DENSE_LINEAR_SEG_v1`.
- **Labels.** PNG values `0` become ignore `255`. Values `1–150` become classes `0–149`. Values above 150 become `255`. Nearest-neighbor resize of the label to 384.
- **Feature.** Dense patch grid, channel-normalized.
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
- **Target.** Metric depth in metres, bilinearly resized to 384. A pixel is valid when depth is finite and lies in `(0.1, 10.0)` m.
- **Feature.** Dense patch grid, channel-normalized.
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
- **Input.** 8 frames from `linspace` over the decoded video. Each frame is a 384 square. Missing or unreadable videos contribute a zero clip and are counted in `n_video_empty`.
- **Feature.** Per-frame MAP vector, L2-normalized, then the mean over the 8 frames.
- **Head, loss, optimizer, schedule, seeds, and metrics.** Zero-init `Linear(1152, 174)`, cross-entropy, AdamW learning rate `0.001`, weight decay `0.0001`, head batch 1024, 100 epochs, cosine to `0`, last-epoch Top-1 (%) mean and sample standard deviation over seeds `42`, `123`, `456`. Top-5 (%) is secondary.
- **Views.** One 8-frame clip per video. No multi-clip average.

### Execution and seed policy

```bash
python ${LOCAL_REFERENCE_PATH} \
  --method siglip --dataset imagenet_1k --device cuda
```

Repeat with `--dataset` in `ade20k`, `nyu_depth_v2`, `something_something_v2`. Completed cells record `canary: false`, `precision: fp16`, checkpoint id `google_siglip2-so400m-patch14-384`, and `run_id` `r1` or `r2`. COCO is the Faster R-CNN recipe in this file, on the frozen dense patch grid.

### Evaluation and reporting

Write `run_meta.json` and `result.json` under `VLM/SigLIP/results/<dataset>/SigLIP/google_siglip2-so400m-patch14-384/<run_id>/`. Record `feature_dim` 1152, patch size 14, input 384, normalization mean and std `0.5`, and the protocol name of the task. Report Top-1, Top-5, bbox AP, AP50, mIoU, and pixel accuracy in percent. Report median-aligned RMSE in metres and AbsRel as a unitless ratio after the median scale. State the seed list, or the single seed, beside the number.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Included checkpoint | SigLIP 2 So400m/14 at 384 | Completed `run_meta.json` files under `VLM/SigLIP/results/`; `METHODS["siglip"]` | Observed main-run setting |
| Global feature | MAP pool from `get_image_features` | `SigLIPAdapter.extract_features` | Observed main-run setting |
| Dense feature | `vision_model.last_hidden_state` patch grid | `SigLIPAdapter.extract_dense_features` | Observed main-run setting |
| Normalization | mean = std = `(0.5, 0.5, 0.5)` | `SIGLIP_MEAN` and `SIGLIP_STD` in `VLM/SigLIP/adapter.py` | Observed main-run setting |
| Non-COCO probe schedule | The non-COCO task rows in this file | `core20/registry.py` `PROBE` and the matching functions in `run_cell.py` | Observed main-run setting |
| COCO task | bbox AP (%), AP@[0.50:0.95]. Faster R-CNN, 12 epochs, batch 2, LR `0.001`, WD `1e-4`, 1-epoch warmup, cosine, best validation epoch, 384 square | `3DFM/_common/coco_probe.py`; `INITIAL_STEP3_3DFM_v1.md`; `INITIAL_STEP3_4DFM_v1.md` | Supplied detection reconstruction; run matching pending |
| ADE and NYUv2 weight decay | Executed AdamW default `0.01` | `run_ade` and `run_nyu` pass `lr` and omit `weight_decay`. The `PROBE` value `0.0` is not applied | Observed main-run setting |
| Depth metric | Median-aligned RMSE | `depth_metrics` | Observed main-run setting |
| CLIP, SigLIP 2 Giant, and the seven later VLMs | Outside this completed campaign | CLIP has a registry row and no matching result tree. Giant and the seven adapters are later `METHODS` entries | Compatibility gap for this protocol’s completed set |

### Evidence references

- `methods_step3/VLM/SigLIP/adapter.py`
- `methods_step3/core20/registry.py` keys `DATASETS`, `PROBE`, and `METHODS["siglip"]`
- `methods_step3/core20/run_cell.py`
- Detection recipe: `methods_step3/3DFM/_common/coco_probe.py` and `methods_step3/protocol/INITIAL_STEP3_3DFM_v1.md`
- `methods_step3/core20/datasets.py`
- `methods_step3/VLM/SigLIP/results/<dataset>/SigLIP/google_siglip2-so400m-patch14-384/r2/run_meta.json`
