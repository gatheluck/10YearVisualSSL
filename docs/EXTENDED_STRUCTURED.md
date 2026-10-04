# Extended pose, localization and spatial reasoning

See [package scope and terminology](SUBMISSION_SCOPE.md) for the distinction
between original experiments, portable components and reproduced results.

This component port provides LP/AP training and evaluation for MPII Human Pose,
CrowdPose, 7-Scenes, Cambridge Landmarks and 3DSRBench. Original experimental
implementations exist; this guide describes their portable integration, not
certification of the final manuscript's scores. Every result retains
`canonical_eligible: false` and `record_value: false`.

## Supported behavior

| Dataset | Inputs and head | Evaluation |
| --- | --- | --- |
| MPII | Explicit person crops, 16 heatmaps, Gaussian sigma 2 on a 56 grid; missing joints masked | HRNet validation-sidecar PCKh, prediction +1 to one-based annotation coordinates; pelvis/thorax excluded; percentage |
| CrowdPose | GT person crops, 14 heatmaps; GT boxes are part of the top-down protocol | CrowdPose OKS AP/AP50/AP75, percentages; maxDets 20, mean peak confidence, no NMS; all annotation images included |
| 7-Scenes | Single RGB, known scene, absolute translation and continuous 6D rotation head | Equal-weight mean of scene medians, metres and degrees separately; bounded released-pose deviations and SVD rotation projection |
| Cambridge | Single RGB and known scene; same absolute-pose head | Equal-scene medians; camera-to-world quaternion/centre interpretation |
| 3DSRBench | Full structured question and each answer option as UTF-8 bytes; GRU and nonlinear visual/text fusion | Local COCO-2100 fitted-holdout accuracy, percentage; **not CircularEval** |

The five existing Extended providers are DINOv3-7B, SigLIP2 Giant, RAEv2-K7,
VGGT-Omega and V-JEPA 2.1. LP uses their frozen global readout for
localization/reasoning (MAP magnitude is retained; CLS/patch-pooled profiles are normalized) and spatial features for pose. AP uses the provider's
explicit spatial adapter for pose/localization and query reader for reasoning.
The encoder remains frozen and in evaluation mode. RGB crops for pose use
bilinear affine sampling; localization/reasoning use fixed bicubic 224 resizing.
Provider input normalization is applied once through the existing interface.

## Prepare native inputs

Run from the repository root with the downstream environment installed:

`python -m downstream.structured_membership --dataset seven_scenes --data-root /data/seven-scenes --out /data/seven-scenes/samples.json`

The converter never overwrites an existing output. It accepts these explicit
layouts under `--data-root`; it does not download or extract archives:

- `seven_scenes`: each scene has `TrainSplit.txt`, `TestSplit.txt` and the listed
  `seq-NN.zip` files, containing matched `.color.png` and `.pose.txt` members.
  All seven scenes and full per-scene counts are required, including all stairs
  sequences. Training and evaluation sequences must be disjoint.
- `cambridge_landmarks`: five scene ZIPs (`GreatCourt`, `KingsCollege`,
  `OldHospital`, `ShopFacade`, `StMarysChurch`), each containing its scene directory,
  `dataset_train.txt`, `dataset_test.txt` and images. Full source counts are checked.
- `crowdpose`: `archives/CrowdPose_annotations.tar.gz` and
  `archives/CrowdPose_images.zip`. Uses `crowdpose_train.json` and
  `crowdpose_val.json`; preserves ignored/crowd people and annotation keypoint counts.
- `mpii_pose`: `archives/mpii_release_v1.json`, extracted `images/`,
  `mpii_pins.json`, and the pinned HRNet sidecar under `archives/`.
  Pins require `release_sha256`, `gt_val_sidecar_sha256`, and
  `gt_val_sidecar_filename`. The converted release must use the recorded
  image-disjoint training rule and exactly 2,958 validation persons. The sidecar
  must identify that release and join every validation person exactly once.
  A missing sidecar is an error; the RELEASE metric is not substituted.
- `three_d_srbench`: `3dsrbench_v1.csv`, `sr_holdout.json`, and `images/` with
  the relative paths from the CSV image URLs. The CSV hash is pinned to the
  inspected COCO-2100 source; the supplied holdout must enumerate disjoint
  `train_ids`/`eval_ids` and matching question counts. No fresh random split is
  generated. The question renderer never reads the answer to construct options.

Private annotations, pins, split manifests and images stay outside Git. Keep the
exact source files with the experiment evidence. Parser success and counts do
not authenticate a release or establish that it generated a paper table cell.
The Python builder's `fixture_counts` argument exists only for reduced tests;
the CLI does not expose it.

## Explicit manifest contract

The converter emits `schema_version: 1`, `dataset`, `split_evidence`, `train`,
`validation`, and, for localization, an ordered `scenes` list. Each row has a
unique string `id` and `image: {"path": "relative/file"}`, with optional
`"member": "relative/member"` for ZIP images. Absolute/escaping paths, missing
members, duplicate IDs and train/evaluation image-byte overlap fail before model
construction. Multiple questions or persons may share an image within one split.

Localization rows contain `scene` and a camera-to-world 4x4 `pose`. Reasoning rows
contain `question`, distinct `options`, and the zero-based `target` option index.
Pose rows contain `people`, with `id`, `joints`, boolean `labeled`, and xywh `crop`.
CrowdPose additionally retains per-image `crowd_index` and each person's 14x3
`keypoints`, `bbox`, `num_keypoints`, `iscrowd`. MPII validation persons require
`pos_gt_src`, `jnt_missing`, and `headboxes_src` from the joined sidecar.

## Train, evaluate and continue

Copy and edit one of the explicit configuration examples:
[pose](examples/extended_pose.json), [localization](examples/extended_localization.json),
[reasoning](examples/extended_reasoning.json). Local model files and actual dataset
paths must be supplied; the examples are not bundled pretrained experiments.

`python -m downstream.extended_structured --config /work/run.json --out /work/run-001`

`torchrun --nproc-per-node=2 --module downstream.extended_structured --config /work/run.json --out /work/run-002`

The shared [execution](EXTENDED_EXECUTION.md) and
[continuation](EXTENDED_EXECUTION.md#epoch-checkpoints-and-continuation) contracts apply: explicit accumulation, actual
provider tail policy, FP32 or supported CUDA BF16, rank-zero full-population
evaluation, and epoch-boundary continuation. Set `resume` to the previous
`resume.pt` and increase the target epoch count into a **new output directory**.
Resume identity includes the manifest, image-byte digest, configuration and
encoder state; changing any of them is not a valid continuation.

Outputs include a hashed run manifest, named metrics, `results.json`, `probe.pt`
and `resume.pt`. Frozen encoder tensors are excluded from exported probe state.
Pose/localization train for at most 30 epochs; reasoning at most 50. Seed is 0.
The source resolves short `cosine`/warmup descriptions to a zero final rate and
warmup starting at 1e-6. LP/AP weight decay and effective-batch LR scaling remain
separate. Evaluation refuses repeated or missing samples and empty targets.

## Evidence and unresolved differences

The port was checked against captured implementations and current read-only
experimental sources. Tests compare numerical behavior, input geometry,
independent membership, frozen gradients, all five provider interfaces, LP/AP
updates, continuation, metric output and failure handling. Reduced reference
head comparisons cover initialization, outputs, gradients and three AdamW updates.
These are component tests, not measurements with released full-size weights.

The supplied localization companion describes retrieval/geometric verification
and contrastive loss; the inspected executable run settings use absolute pose
regression with metre L1 plus rotation-matrix L1. This implementation names and
preserves the latter component behavior without claiming to resolve historical
score attribution. Likewise 3DSRBench's local fitted holdout must not be labelled
an official CircularEval result. Historical split/audit issues and the corrected
runnable catalog must be matched to individual final-table runs before assigning
scores. A runnable catalog entry is not proof of a completed experiment.

Released-weight CUDA/BF16/NCCL, full-data numerical reproduction, legacy checkpoint
import, and whole-builder random-number consumption parity remain unverified.
The portable builder constructs the final heads directly; the original can
construct and replace preliminary heads, consuming additional random numbers.
FT, DAVIS, SUN RGB-D and SpaceNet are not enabled by this module. The
[coverage ledger](FINAL_PAPER_COVERAGE.md) records the remaining paper-level work.
