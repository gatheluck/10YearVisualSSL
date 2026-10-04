# Final submitted manuscript: implementation coverage

Audit: 2026-09-27, against the final 70-page submitted manuscript and portable
baseline after PR 203 and the following two-family component port. This supersedes the table numbering and current-priority
claims in the [September 25 ledger](PAPER_REPRODUCTION_GAPS.md), not its dated
research history. See [terminology](PAPER_TERMINOLOGY.md) and
[submission scope](SUBMISSION_SCOPE.md). Private manuscript/source fingerprints
and run evidence remain outside Git.

Updated 2026-10-04 for structured-task LP/AP integration. Remaining priorities
below distinguish executable component support from actual result reproduction.

**The portable package does not yet reproduce the entire paper.** Original
experimental implementations exist for many remaining gaps. A provider, extracted
feature file, test fixture score, or tracked protocol is not proof of a complete
training/evaluation port or a reproduced table cell. Workbook values predate
some final manuscript revisions and must be matched by actual run identity.

## Coverage by final appendix

| Final location | Portable basis | Remaining verification or integration |
| --- | --- | --- |
| C.2 Table 10: ASIS, 37 methods | Method adapters, checkpoint/readout and extraction components | Per-row checkpoint/preprocessing, released-weight evaluation and score/run matching; extraction counts are not table coverage |
| C.3 Tables 11-14: CTRL, 32 methods | Controlled training adapters and checkpoint interfaces | Native distributed settings, 100/200/300-epoch checkpoint identities and full-run parity |
| C.4 Tables 15-22: DINO design/scaling, InstDisc, colorization, rotation, SplitBrain | [DINO components](DINOV3_STEP4.md), [instance-discrimination components](IDV2_COMPONENTS.md) and existing method code | Full native training, conversion/continuation and each ablation's result-to-run evidence |
| C.5 Table 23: supervised references | Existing supervised adapters | Exact reference training/evaluation recipes and score parity |
| C.6 Tables 24-25: 3D; C.7 Table 26: 4D | Existing inference/extraction providers where available | Variant/readout identity, task-specific wrappers, gradients and native metrics |
| C.8 Tables 27-28: generative; C.9 Table 29: VideoSSL | Existing providers and selected native video components | RAEv2 K7 versus other readouts; distinct video checkpoints/pipelines and task integrations |
| C.10 Tables 30-34: world/video generation; C.11 Table 35: VLM and related models | Selected provider/component integrations | All rows' exact component boundaries, adapters, task recipes and checkpoint identities |
| C.12 Tables 36-38: BasicFive LP/AP/FT | Existing task runners, [five-family readers/native detection](BASIC5_NATIVE_PATHS.md), [K7](BASIC5_K7.md), [Omega](BASIC5_OMEGA.md) and [C-RADIO](BASIC5_RADIO.md) | Remaining recipes, full-size checkpoint/GPU validation and actual full-data measurements; blank/incomplete FT cells must stay distinct from completed runs |
| C.13 Tables 39-40: Extended LP/AP, 45 datasets and five models | Tracked registries, protocol companions, [image classification](EXTENDED_CLASSIFICATION.md), [semantic segmentation](EXTENDED_SEGMENTATION.md) and [video classification](EXTENDED_VIDEO.md) components for five providers, with explicit [accumulation/precision](EXTENDED_EXECUTION.md) | Remaining task families, native input builders, conflicting task/metric identities, legacy checkpoint import, released-weight distributed/CUDA parity and measured-run manifests; 81 catalog entries do not mean 81 measured datasets |
| C.14 Table 41: six representative configurations | Per-method adapters and protocol documents | Different original protocols require separate recipe identities, not one inferred common leaderboard |
| C.15 Table 42: frontier matched subsets | Supplied system prompts and protocol | Matched 500-sample manifests, raw outputs/retries, parsing, scoring and recomputation |
| D.1-D.3: pairplots, CTRL progress and DINO scaling | Selected extraction/training artifacts | Complete provenance-linked analysis inputs and plot regeneration |
| E: scope, run accounting and protocols | Eight current companions, separate historical companions and registries | Final-epoch results, actual per-cell run counts, sample standard deviations and task-specific run-to-table evidence |

The BasicFive AP/FT tables list eight families. All eight now have portable
downstream components, including C-RADIO's 14 fixture routes added on 2026-09-28.
This does not complete their experimental recipes or numerical reproduction.
SAM3 is useful elsewhere but is not one of
those eight. V-JEPA2.1 already has differentiable image/native-video components,
LP/AP and four-task FT with optimizer groups and schedules; its remaining gap is
complete recipe and released-weight/result validation, not an absent provider.
ImageNet FT execution remains pending across these components. Component
availability does not establish complete native-video or paper-score coverage.

## Prioritized next work

1. Complete the remaining BasicFive family/task integrations in source-supported
   groups, with initialization/output/gradient/update parity. The preceding port
   exercised 35 explicit routes, the two-family port added 28, and C-RADIO adds 14.
   C-RADIO's local custom-code loader and task interfaces now exist; full-size
   released-checkpoint/GPU validation remains.
   Full-scale result reproduction is separate. ImageNet FT augmentation still
   needs per-run reconciliation.
2. Integrate Extended task trainers and metrics for the **45 actually reported
   datasets**, retaining each of the five model/readout identities. The image
   LP/AP execution path now exists, with explicit sample manifests and native
   CUB-200-2011/DTD/Aircraft annotation conversion, extended on September 30 with
   Food-101/Pets/IP102/MIT Indoor-67 annotation conversion, followed by Cars MAT
   annotation conversion and lossless MNIST IDX staging. This addresses the common
   path used by 24 image-classification rows, not 24 independently validated
   dataset ports or reproduced scores. [Semantic LP/AP](EXTENDED_SEGMENTATION.md)
   now adds five-provider spatial heads/adapters, 224-grid metrics and native
   ADE/VOC/BDD membership conversion. It is relevant to five reported semantic
   rows; Cityscapes is catalog-only. SpaceNet's metric and SUN RGB-D's task differ
   between the supplied registry and final table, so both remain refused until
   run identity is reconciled. Other native builders, complete recipes and other
   non-image tasks remain. The 2026-09-29 shared execution port adds accumulation
   and CUDA BF16 selection to both paths. CPU update parity does not establish
   released-weight CUDA parity or resolve the captured schedule clock's mapping
   to final table cells. CPU-tested distributed execution now exists for both runners, including rank-zero
   full-population evaluation. Portable epoch-boundary continuation now restores probe/optimizer/scheduler
   state and per-rank RNG streams across both runners. Legacy checkpoint import
   and released-weight distributed/CUDA continuation validation remain.
   The 2026-10-01 video port adds HMDB51/UCF101 split-1 conversion and five-provider
   LP/AP with shared execution and continuation. The original rerun catalog flags
   historical split contamination; final-table-to-corrected-run identity remains
   unverified. Strict membership and component tests do not certify those scores.
   The 2026-10-02 [detection component port](EXTENDED_DETECTION.md) adds COCO/LIVECell/LVIS inputs,
   five-provider LP/AP detector compositions and official prediction evaluation.
   The October 3 port connects those components to LP/AP training, accumulation,
   distributed execution, continuation and final scoring. FT, source/catalog geometry,
   released-weight/full-data validation and score provenance remain pending.
   Registry presence alone is insufficient.
   The October 4 [NYUv2 depth port](EXTENDED_DEPTH.md) adds the five-provider
   Extended LP/AP head/loss, native split-file inputs, training, distributed
   continuation and unaligned pixel-pooled metrics. These explicitly retain
   differences from the BasicFive component recipe. Full-data/released-weight
   validation, historical run attribution and other depth datasets remain pending.
   Charades and AVA remain distinct tasks.
   The October 4 [optical-flow port](EXTENDED_FLOW.md) adds five-provider LP/AP
   execution, Sintel/Middlebury local held-out membership conversion, explicit
   Spring inputs, input-coordinate metrics and distributed continuation. Sintel
   and Middlebury appear in the final tables; Spring is catalog-only. Captured
   feature-difference/L1 behavior conflicts with the companion's correlation/
   robust-endpoint description. This remains a named experimental component
   profile, not a resolution of that conflict or certification of the table cells.
   The October 4 [structured-task port](EXTENDED_STRUCTURED.md) adds five-provider
   LP/AP execution for MPII/CrowdPose, 7-Scenes/Cambridge and 3DSRBench, native
   annotation conversion, task-specific metrics and continuation. Absolute-pose
   regression conflicts with the companion retrieval description; the local
   reasoning holdout is not CircularEval. These component paths do not certify
   table cells. DAVIS, SpaceNet/SUN RGB-D identity reconciliation and full-run
   score attribution remain pending.
3. Validate native distributed/BF16 training and continuation/export against
   original checkpoints for controlled and ablation training. Small FP32
   component tests do not validate these paths.
4. Apply [downstream run accounting](DOWNSTREAM_ACCOUNTING.md) to actual recorded
   runs, with final-epoch selection, missing/partial repeats and sample standard
   deviation. The portable artifact checker now exists; historical schema
   conversion and run-to-table matching remain.
   The existing method linear-evaluation three-seed aggregator does not cover
   all downstream LP/AP/FT accounting described by E.1.
5. Port frontier response parsing/scoring and matched-subset provenance, then
   regenerate plots and table inputs from verified manifests.

## Do not resolve these differences by guessing

- Common experimental AP has cross attention plus self attention, while the
  protocol describes one attention block; DINO's reader is a separate variant.
  Explicit profiles retain both rather than relabeling one as canonical.
- Final E.4 requires unaligned NYUv2 metric RMSE; inspected experimental
  evaluators also contain median-aligned evaluation. Map runs before attributing
  either metric to a paper value. The public metric remains unaligned.
- Historical reconstructions and current Unified companions are separate
  specifications. Initial COCO is object detection. E.6's frontier category-
  presence Micro-F1 is a separate task and must not be confused with bbox AP.
- Final manuscript tables, older workbook rows and local artifact availability
  may differ. Missing physical files do not establish that an experiment was
  never run, and a source implementation does not establish a verified score.
- Flowers102's registry specifies train+validation, whereas the current inspected
  builder uses `setid.mat:trnid` only and checks/excludes `valid`. Its native
  membership port remains pending final run reconciliation.
- MNIST's native builder uses the shared `small=False` crop/flip path, whereas
  its registry describes the small-image recipe. IDX staging preserves input
  pixels and splits but does not resolve which transform produced a table cell.
- Flow's executable head/loss, local held-out splits and 224-pixel metric units
  must be reconciled with the companion and historical final-table runs. A
  runnable catalog entry is not a completed experiment. BlinkVision prepared
  data revisions and Spring native archive partitioning remain outside this port.

This audit used read-only comparison of captured and current experimental
sources. No original code, checkpoint or existing feature artifact was changed;
no compute job was submitted for this component port. GPU/released-weight and
full-score validation remain pending. Local and CI outcomes are recorded in
the implementation PR, not inferred from this document.
