# INITIAL_STEP1_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

The supplied reconstruction describes method-official historical ImageNet probes and shared full-mode downstream readers. Its NYUv2 split is labeled-file order (795/654), with metric depth and no median alignment; COCO is bounding-box AP.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| COCO, ADE, NYUv2, SSv2 recipes | The full-mode arguments in the Step 1 launcher and `downstream/*.py` | Launcher `EPOCHS`/`LR`/`BATCH_SIZE` defaults; completed JSON `run_config` on the Step 2 sibling uses the same modules | Observed main-run setting |
| Reported epoch | Last scheduled epoch | `downstream/ade20k_segmentation.py` stores `final = history[-1]` | Observed main-run setting |
| NYUv2 split | First 795 / remaining 654 of the labeled mat | `downstream/nyuv2_depth.py` `train_indices = list(range(795))` | Observed main-run setting |
| NYUv2 loss and RMSE | Masked L1; metric RMSE in metres, no median scale | `masked_l1` and `evaluate` in `nyuv2_depth.py` | Observed main-run setting |
| COCO metric | Bbox AP from `COCOeval`, reported in percent | `coco_frcnn.py` `evaluator.stats[0]` | Observed main-run setting |
| Learning-rate scaling | Absolute LR at the stated batch | The downstream trainers do not scale LR by batch | Observed main-run setting |
| Seed | `0` | Launcher `SEED` default | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

The single ImageNet SGD recipe (LR 0.1, batch 256, 100 epochs) is a newly selected unification, not the collection of historical method-official probes.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| ImageNet probe | One SGD recipe, LR `0.1` at batch 256, 100 epochs, final epoch | Stored Step 1 ImageNet JSON files use different official probes (examples: MoCo `lr=30` at batch 256; DINO `lr=0.001` per GPU; MAE LARS `lr=6.4`) | Newly selected unification |

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

This document is a candidate specification reconstructed from the initial Step 1 campaign in `methods/`. It is not a claim that every stored historical score already follows this file. Stored ImageNet linear scores used method-official probe recipes. The four shared downstream tasks below match the full-mode launcher and the `downstream/` implementation that wrote `local_artifacts/downstream_full_resumable/`.

### Purpose and category scope

Step 1 compares visual SSL methods from their accepted as-is checkpoints: original architecture, original pretext training, and the checkpoint named by `configs/step1_downstream_registry.yaml`. This protocol does not retrain those pretext tasks and does not apply the Step 2 ViT-B/16 contract.

Evaluate every included checkpoint with the same frozen-backbone recipes. The backbone stays in `eval()` with gradients disabled. Train only the task head.

### Included models and datasets

Include every method row in `configs/step1_downstream_registry.yaml` whose `imagenet_result` is an accepted Step 1 artifact. That registry is the membership list (Context Prediction through CLIP, including the later-numbered PIRL, MSN, V-JEPA, Franca, and LeJEPA rows). Skip a row that has no accepted checkpoint file.

| Task | Dataset | Split | Root used by the Step 1 launcher |
|---|---|---|---|
| Image classification | ILSVRC2012 ImageNet-1k | official train / validation | `${LOCAL_REFERENCE_PATH}` |
| Object detection | COCO 2017 | train2017 / val2017 | `${LOCAL_REFERENCE_PATH}` |
| Semantic segmentation | ADE20K Challenge 2016 | official training 20,210 / validation 2,000 | `${LOCAL_REFERENCE_PATH}` |
| Metric depth | NYUv2 labeled | first 795 images train, remaining 654 validation, in `nyu_depth_v2_labeled.mat` order | `${LOCAL_REFERENCE_PATH}` |
| Action recognition | Something-Something v2 | official train / validation JSON | `${LOCAL_REFERENCE_PATH}` |

NYUv2 here is file order inside `labeled/nyu_depth_v2_labeled.mat`. It is not the Eigen `splits.mat` index list. Depth values are metric metres. RMSE is metric RMSE, not median-aligned RMSE.

COCO here is box detection. The primary metric is bbox AP, not image-level multilabel mAP.

### Global rules

- Load the accepted Step 1 checkpoint named by the registry. Do not run intermediate adaptation.
- Freeze every backbone parameter. Keep the backbone in `eval()` for training and evaluation.
- Read one spatial feature map from `forward_features`. Drop class tokens and register tokens before the map is built.
- Apply ImageNet normalization in the dataset loader: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- Use seed `0` for every task head.
- Use the stated learning rate at the stated physical batch. Do not rescale the learning rate when the physical batch changes. Accumulate only if the effective batch stays equal to the stated physical batch.
- Report the last scheduled epoch. Do not select a checkpoint on the reported validation or test split.
- Record the checkpoint path, feature width, grid shape, resolution, effective batch, learning rate, and seed in `results.json`.

### Common representation interface

The adapter boundary is `downstream/backbone_adapters.py` (`FrozenViTSpatialBackbone.forward_features` and the ResNet/classic adapters that return the same tensor layout).

- **Checkpoint.** The registry `imagenet_result` artifact and its `source` checkpoint. Initialize the task head randomly. Do not load a detection or segmentation head from pretraining.
- **Included components.** The pretrained trunk through its final normalization. Exclude the pretext projection head, prototype layer, decoder, and classifier.
- **Frozen scope.** All trunk parameters, including normalization affine weights and positional embeddings. Batch-norm running statistics stay at the checkpoint values.
- **Feature stage.** The final trunk block. The tensor is a spatial map of patch tokens or convolutional features after prefix tokens are removed. It is not a class token, not a concatenation of layers, and not a pretext-head embedding.
- **Tensor interface.** `forward_features(x) -> (B, C, H_grid, W_grid)` with row-major patch order. `forward` exposes that map as the single detection scale `"0"`.
- **Width and grid.** `C` is the trunk width of the loaded checkpoint. At 224 input and patch 16 the grid is 14×14. Record `C`, `H_grid`, and `W_grid` in the result.
- **Special tokens.** Drop the class token and any register tokens. Do not append them to the grid.
- **Pooling.** Detection and segmentation and depth use the spatial map. ImageNet and SSv2 apply adaptive average pooling to 1×1, then flatten. SSv2 then averages the frame vectors.
- **Preprocessing.** RGB, scale to `[0, 1]`, ImageNet normalization, then the trunk. Square resize for ADE, NYUv2, and SSv2. COCO leaves resize to the Faster R-CNN `min_size` / `max_size` transform.
- **Head boundary.** The first trainable layer is the task head. Nothing in the trunk receives a gradient.

A checkpoint that cannot emit this spatial map cannot enter the dense or detection tables. Do not replace the map with a repeated global vector.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Primary metric |
|---|---|---|---|---|---|
| ImageNet-1k | 224 center evaluation | linear, 1000-way | SGD | 100 epochs, cosine | Top-1 (%) |
| COCO 2017 | short side 800, long side ≤ 1333 | Faster R-CNN, one scale | SGD | 12 epochs, constant LR | bbox AP (%) |
| ADE20K | 224 square | 1×1 conv, 150-way | AdamW | 30 epochs, constant LR | mIoU (%) |
| NYUv2 | 224 square | DPT-style depth head, hidden 256 | AdamW | 30 epochs, constant LR | RMSE (m) |
| SSv2 | 224 square, 4 frames | linear, 174-way | AdamW | 30 epochs, constant LR | Top-1 (%) |

### Detailed task recipes

#### ImageNet-1k

- **Split.** Official ILSVRC2012 train and validation.
- **Target.** The 1,000-way class index.
- **Input.** Train: random resized crop to 224 (scale 0.08–1.0, ratio 0.75–1.333) and horizontal flip with probability 0.5. Evaluate: resize the shorter side to 256 and center-crop 224. ImageNet normalization.
- **Feature.** Spatial map from the interface, then adaptive average pooling to a vector of width `C`. No extra L2 normalization.
- **Head.** `Linear(C, 1000)`, normal weight initialization with std `0.01`, zero bias. Train only this head.
- **Loss.** Cross-entropy. No label smoothing.
- **Optimizer.** SGD, momentum `0.9`, Nesterov disabled, weight decay `0`.
- **Learning rate.** `0.1` at effective batch 256. The learning rate is absolute at that batch.
- **Batch.** Physical batch 256. No gradient accumulation.
- **Schedule.** 100 epochs. Linear warmup for 10 epochs from `1e-6` to `0.1`, then cosine decay to `1e-6`.
- **Augmentation.** Crop and flip only.
- **Checkpoint rule.** Last scheduled epoch.
- **Metrics.** Top-1 (%) primary. Top-5 (%) secondary.

#### COCO 2017 detection

- **Split.** train2017 (118,287 images) and val2017 (5,000 images).
- **Target.** Instance boxes. Crowd annotations and boxes with width or height ≤ 1 are dropped. Category ids stay in the COCO 1–90 indexing used by torchvision, with `num_classes=91`.
- **Input.** `TF.to_tensor` only in the dataset. Faster R-CNN resizes the shorter side to 800 and caps the longer side at 1333. No horizontal flip.
- **Feature.** The single spatial map `"0"`. `size_divisible` equals the trunk patch size.
- **Head.** Torchvision Faster R-CNN. Anchors `(32, 64, 128, 256, 512)` and ratios `(0.5, 1.0, 2.0)`. `MultiScaleRoIAlign` on feature name `"0"`, output 7×7, sampling ratio 2. Train the RPN, RoI heads, and box predictor. The trunk stays frozen.
- **Loss.** The standard Faster R-CNN sum of classification, box regression, objectness, and RPN box regression.
- **Optimizer.** SGD, momentum `0.9`, weight decay `5e-4`.
- **Learning rate.** `0.005`, absolute, at effective batch 4. No warmup and no decay.
- **Batch.** Physical batch 4. Launcher: `scripts/submit_step1_downstream_resnet.sh` with `MODE=full`.
- **Schedule.** 12 epochs.
- **Augmentation.** The internal resize only.
- **Checkpoint rule.** Last scheduled epoch.
- **Evaluation.** `COCOeval` on bbox, restricted to the image ids in the loader. One view per image.
- **Metrics.** Primary bbox AP (%) = `100 * stats[0]` (AP@[0.50:0.95]). Secondary AP50 (%) = `100 * stats[1]`. The JSON field `bbox_mAP` stores the unit interval; the reported table uses percent.

#### ADE20K

- **Split.** `images/training` with `annotations/training` (20,210) and `images/validation` with `annotations/validation` (2,000).
- **Target.** 150 semantic classes. ADE label 0 and void become `ignore_index=255`. Labels 1–150 become class indices 0–149.
- **Input.** Square resize to 224×224. Images use bilinear interpolation. Masks use nearest-neighbor interpolation. ImageNet normalization. No scale jitter and no flip.
- **Feature.** Spatial map, then a 1×1 convolution, then bilinear upsampling to 224×224 with `align_corners=false`.
- **Head.** `Conv2d(C, 150, kernel_size=1)`, normal weight initialization with std `0.01`, zero bias. Train only this convolution.
- **Loss.** Pixel cross-entropy with `ignore_index=255`.
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`. Default AdamW betas `(0.9, 0.999)`.
- **Learning rate.** `1e-3`, absolute, at effective batch 16. Constant.
- **Batch.** Physical batch 16.
- **Schedule.** 30 epochs.
- **Checkpoint rule.** Last scheduled epoch.
- **Metrics.** Primary mIoU (%) = `100 *` the mean of per-class IoUs over classes that appear in the union. Secondary pixel accuracy (%) = `100 * pACC`.

#### NYUv2 metric depth

- **Split.** Indices `0:795` train and `795:1449` validation inside `labeled/nyu_depth_v2_labeled.mat`.
- **Target.** Metric depth in metres. A pixel is valid when it is finite and strictly positive. There is no 10 m cap in this loader.
- **Input.** Bilinear resize of RGB to 224×224, nearest resize of depth to 224×224, ImageNet normalization. No flip and no color jitter.
- **Feature.** Spatial map passed to the depth head. The head upsamples with bilinear interpolation and `align_corners=false`.
- **Head.** `DPTDepthHead`: 1×1 projection from `C` to 256, GroupNorm(8) and GELU, four residual 3×3 refine blocks that each double spatial size until 224, then a 3×3 and a 1×1 to one channel, then `softplus`. Train only this head. Hidden width is 256.
- **Loss.** Masked L1 between the softplus prediction and metric depth, averaged over valid pixels.
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`.
- **Learning rate.** `1e-3`, absolute, at effective batch 16. Constant.
- **Batch.** Physical batch 16.
- **Schedule.** 30 epochs.
- **Checkpoint rule.** Last scheduled epoch.
- **Evaluation.** Full validation split. No per-image median scale alignment.
- **Metrics.** Primary RMSE (m) on valid pixels. Secondary AbsRel (unitless) on the same pixels.

#### Something-Something v2

- **Split.** `labels/train.json` and `labels/validation.json`, joined to `labels/labels.json` templates. 168,913 train and 24,777 validation videos in the completed Step 2 sibling run; Step 1 uses the same loader.
- **Target.** 174 action classes.
- **Input.** 4 frames at indices `linspace(0, T-1, 4)`, rounded. Each frame is square-resized to 224 and ImageNet-normalized. No horizontal flip. The same frame indices are used at train and evaluation time.
- **Feature.** Per frame, adaptive average pool of the spatial map. Mean over the 4 frame vectors.
- **Head.** `Linear(C, 174)`, normal weight initialization with std `0.01`, zero bias. Train only this head.
- **Loss.** Cross-entropy.
- **Optimizer.** AdamW, learning rate `1e-3`, weight decay `0.01`.
- **Learning rate.** `1e-3`, absolute, at effective batch 8. Constant.
- **Batch.** Physical batch 8.
- **Schedule.** 30 epochs.
- **Checkpoint rule.** Last scheduled epoch.
- **Evaluation.** One clip per validation video. Full validation list.
- **Metrics.** Primary Top-1 (%). Secondary Top-5 (%).

### Execution and seed policy

Launch the four non-ImageNet tasks with `MODE=full` through `scripts/submit_step1_downstream_resnet.sh`. The modules are `downstream.coco_frcnn`, `downstream.ade20k_segmentation`, `downstream.nyuv2_depth`, and `downstream.ssv2_linear`. Pass `--seed 0`. ImageNet uses the same seed and the recipe in this file.

Run one trial per method per task. The candidate number is that single seed-0 result. Do not average with the later three-seed campaign under `methods/_seed3/`.

Write outputs under `local_artifacts/downstream_full_resumable/<task>_method<id>/`. A run that sets `max_train_samples`, `max_val_samples`, or `max_steps_per_epoch` above 0 is a pilot and stays out of the table.

### Evaluation and reporting

Report one row per method per task: the last-epoch metric, the checkpoint path, feature width and grid, input size, frame count, effective batch, learning rate, and seed. State units in the column header. Keep bbox AP, multilabel mAP, metric RMSE, and median-aligned RMSE in separate tables. This protocol uses bbox AP and metric RMSE.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Checkpoint source | Accepted as-is Step 1 file in `configs/step1_downstream_registry.yaml` | Registry `imagenet_result` paths and `scripts/submit_step1_downstream_resnet.sh` | Observed main-run setting |
| COCO, ADE, NYUv2, SSv2 recipes | The full-mode arguments in the Step 1 launcher and `downstream/*.py` | Launcher `EPOCHS`/`LR`/`BATCH_SIZE` defaults; completed JSON `run_config` on the Step 2 sibling uses the same modules | Observed main-run setting |
| Reported epoch | Last scheduled epoch | `downstream/ade20k_segmentation.py` stores `final = history[-1]` | Observed main-run setting |
| NYUv2 split | First 795 / remaining 654 of the labeled mat | `downstream/nyuv2_depth.py` `train_indices = list(range(795))` | Observed main-run setting |
| NYUv2 loss and RMSE | Masked L1; metric RMSE in metres, no median scale | `masked_l1` and `evaluate` in `nyuv2_depth.py` | Observed main-run setting |
| COCO metric | Bbox AP from `COCOeval`, reported in percent | `coco_frcnn.py` `evaluator.stats[0]` | Observed main-run setting |
| ImageNet probe | One SGD recipe, LR `0.1` at batch 256, 100 epochs, final epoch | Stored Step 1 ImageNet JSON files use different official probes (examples: MoCo `lr=30` at batch 256; DINO `lr=0.001` per GPU; MAE LARS `lr=6.4`) | Newly selected unification |
| Learning-rate scaling | Absolute LR at the stated batch | The downstream trainers do not scale LR by batch | Observed main-run setting |
| Seed | `0` | Launcher `SEED` default | Observed main-run setting |

### Evidence references

- Scope: the private study outline (not distributed) section “Step 1: As-is SSL Comparison”.
- Membership: `configs/step1_downstream_registry.yaml`.
- Launcher: `scripts/submit_step1_downstream_resnet.sh`.
- Implementation: `downstream/coco_frcnn.py`, `downstream/ade20k_segmentation.py`, `downstream/nyuv2_depth.py`, `downstream/ssv2_linear.py`, `downstream/vit_adapters.py`.
- Result lineage: `local_artifacts/downstream_full_resumable/`.
- Later three-seed reruns under `methods/_seed3/` are outside this protocol.
