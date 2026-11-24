# Basic Five v1 protocol audit

See [submission scope](SUBMISSION_SCOPE.md) for the limits of result attribution.

The three v1 Markdown files and three JSON files in [current protocols](current_protocols/) match the inspected
archived protocol files byte for byte. LINEAR identifies itself as
`BASIC5_FAIR_v1`. LP and AP mean linear and attentive **probing**. These v1
specifications prescribe three seeds (0, 1, 2) and final-epoch results. The
publication drafts also describe a single-seed replication option. One seed
can describe a run, but cannot establish the original three-seed spread.

## What was compared

At reference effective batch, the portable optimization resolver matched the
JSON algorithm, base learning rate, decay, reference batch and epoch horizon
for all 15 task/track settings. All five FT layer-decay factors matched.
This comparison exercises configuration resolution with a supported provider;
it does not establish which configuration produced a figure cell.

| Task | Epochs | LP / AP / FT optimizer and reference LR | Execution evidence and remaining distinctions |
| --- | ---: | --- | --- |
| ImageNet-1k | 100 | SGD .1 / AdamW .001 / AdamW .0005 | LP and AP have provider-specific preprocessing; FT uses provider-specific augmentation and parameter groups. The captured soft targets omit the specification's 0.1 smoothing. See [image preprocessing](IMAGENET_PROBE_PREPROCESSING.md) and [finetuning](IMAGENET_FINETUNE.md). |
| COCO | 12 | SGD .02 for all three | Faster R-CNN/FPN bbox detection, resize 800 to at most 1333 and the 1x step schedule. Verify the provider's pyramid and normalizer; see [native detection](BASIC5_NATIVE_PATHS.md). Report AP, AP50, AP75 and all three size bins. |
| ADE20K | 20 | AdamW .001 / .001 / .0001 | The linear and FT profiles use the minimal segmentation head; AP adds a spatial adapter. Compare mIoU and pixel accuracy. Final valid-pixel counts measure a different quantity from image counts. |
| NYUv2 | 30 | AdamW .001 / .001 / .0001 | Use the 1x1 positive-depth head and valid-pixel log-depth objective for v1 components. Verify the split, valid pixels and alignment per run. The LoRA/DPT unaligned paper-field rule has separate provenance. |
| SSv2 | 50 | SGD .1 / AdamW .001 / AdamW .0005 | The specification uses 16 segments, one evaluation view and no flip. LP and FT have video components. See the [confirmed Figure 3 AP inputs and readers](SSV2_EXECUTION.md#figure-3-attentive-probes). |

Dense execution profiles and distributed continuation are documented in
[dense execution](DENSE_EXECUTION.md). Frozen heads, trainable readers and FT
require different parameter scopes. Reader architecture, model-specific
readout, sample membership, augmentation, optimization updates and the exact
checkpoint all matter to numerical reproduction. Matched scalar settings alone
cannot verify these behaviors.

Three archived campaign indexes each contain 45 distinct task/track/seed
entries. Of the 135 rows, 71 are labeled completed-valid, 61 pending and three
incomplete in those snapshots. In all 71 completed rows, the recorded primary
score equals the named task metric and the track matches its protocol ID.
The indexes describe a historical snapshot; this count does not measure present
cluster completion or certify a result in the revised manuscript.

## Figure 3 score check

The six live protocol files and three campaign indexes matched the captured
bytes. A separate read-only comparison used the five displayed models and
excluded the five held SSv2 AP cells. From the original campaign directories,
36 of 45 LP/AP cells had three readable final-epoch records with matching
protocol hashes whose mean **and sample SD** rounded to the plotted values.
An additional three-seed corrected V-JEPA 2.1-G ADE20K AP run yielded
39.132634 +/- 0.116073, matching 39.13 +/- 0.12 and bringing the count to 37.
The earlier directory for this cell yields different numbers. Count the
corrected set only once. These comparisons verify reported statistics;
checkpoint contents, inputs and independent training were outside this check.

At the time of that audit, eight cells lacked an independently matched three-seed set: VGGT-Omega
ImageNet AP, both COCO and both NYUv2 tracks, SigLIP 2-G ImageNet AP and both
NYUv2 tracks. The original Omega NYUv2 runs give different means, and the
corrected depth files examined were unreadable. The corrected Omega COCO LP
files examined were also unreadable. Access failures do not imply missing runs.
In the supplied manuscript, SigLIP 2-G NYUv2 LP is **0.848 +/- 0.007**;
the earlier figure table shows **2.023 +/- 0.032**. Use the manuscript as the
display source. Numeric agreement with a separate LoRA/DPT control would not
establish which run produced this LP cell.

## COCO evaluation and completed repeats

For native Basic Five comparisons, COCO uses bounding-box AP on val2017:
IoU 0.50 to 0.95 in steps of 0.05, all areas, at most 100 detections per image.
The standard Basic Five evaluator reports AP as a fraction. The revised LoRA
report and the paper use percentage points. Both evaluators check the observed
image IDs against the requested validation population, including explicit
subsets and rank-local shards. Images with no predicted boxes still count.

The November correction records cover all 15 targeted historical settings with
bbox evaluation over 5,000 images. Eleven corrected values match the displayed
ASIS entries in Tables 15–17. For example, the printed APs are 20.83 for the
RAEv2 generator, 2.32 for DiT and 0.12 for iGPT. Table 15 leaves MAGE and JiT
blank despite separate completed evaluations. DINOv3-7B square-224 and
V-JEPA 2.1-L also have separate correction records without corresponding cells
in these tables. Preserve the printed blanks and configuration identities.

A separate comparison of Figure 4 plotting inputs against Tables 10 and 21
covered all 32 methods and five primary metrics. All 160 final CTRL values
agree within the printed rounding interval. For ASIS, 159 of 160 agree;
Context Prediction COCO is 0.30 in the Figure 4 plotting input and 1.47 in
Table 10 and the Figure 2 plotting input. Treat this row as disputed when
reporting the ASIS-to-CTRL change.

Figure 6 and supplementary Table 29 specify a separate 500-image COCO
category-presence Micro-F1 comparison. Keep these observations outside native
detection ranks and aggregate bounding-box AP only within a matching task.
The frontier scorer is documented in the repository's frontier evaluation guide.

Figure 3 and Tables 22–25 agree for all 85 displayed means and sample SDs.
Tables 22–25 report three runs per displayed cell. Full fine-tuning in Table 26
uses the actual completed repeats. Of 25 possible entries, 18 have three runs,
VGGT-Omega on SSv2 has n=1 (63.17, without a sample SD), and six are blank.

### Run attribution checked against the revised supplement

The adoption record and completed results identify RAEv2's DINOv3-L K7 tokenizer
for the Unified experiments. For 22 cells (LP 5, AP 5, FT 5, LoRA 5, DPT 2),
the three original final values per cell reproduce the printed mean and sample
SD in Tables 22–26. LP/AP/FT records explicitly disable the diffusion transformer
and decoder. Keep the generator block-29, timestep-0.1 setting for historical
ASIS; the generalization of this setting to Unified tracks in C.3/E.4 conflicts
with the adopted runs. Original-run agreement establishes attribution; independent
training with the portable implementation has separate results.

The figure revision identifies the SigLIP 2-G NYUv2 LP runs using upright depth,
unaligned RMSE and SILog with lambda 0.5. Their three values are 0.839582,
0.849808 and 0.853224 m (printed 0.848 +/- 0.007). This resolves the source of
that cell in the earlier eight-cell audit; the older 2.023 result used a
different training configuration.

## Auditing outputs

For portable LP/AP/FT outputs run
`python -m downstream.accounting --basic5-v1 --config cell.json --out report.json`.
Supply expected seeds 0, 1 and 2 even for incomplete inputs. This reporting
mode checks the full epoch horizon, explicit zero sample and step caps and
each required primary and secondary metric against its final raw value.
Missing attempts remain visible. The mode checks reporting fields; it cannot
verify full dataset membership from zero caps or establish the depth alignment.

For the two supported native Basic Five schemas, a cell may additionally
supply `protocol_document` with the local JSON file's `path` and `sha256`.
The historical auditor checks its bytes against the recorded protocol hash,
its protocol ID, the selected dataset's epoch horizon and the expected seed
set. Output includes the document digest. Keep original source documents and
record paths outside Git. See [portable accounting](DOWNSTREAM_ACCOUNTING.md)
and [native accounting](REFERENCE_ACCOUNTING.md).

The original audit excluded SSv2 AP. The [later record check](SSV2_EXECUTION.md#figure-3-attentive-probes) identifies its model-specific routes. See also the [Figure 5 ledger](CVPR_EXTENDED_COVERAGE.md) and [Figure 7 attribution](REPRESENTATION_ANALYSIS.md).

## Corrected V-JEPA 2.1 ADE20K attentive run

For `downstream.ade20k`, select `training_revision: dense_ap_fp32_norm_bias_v1`
with `profile: capture_basic5_components`, `adaptation: attentive`,
`backbone.kind: vjepa2_1` and `optimizer_profile: basic5_frozen_v1`.
In the corrected runs the frozen encoder can execute under autocast. The
spatial reader, classification head and output resize execute in float32.
AdamW applies the recipe weight decay to projection weights and zero decay
to trainable biases and normalization parameters. The selection is included
in the optimization report and in the configuration used for continuation.
The three-seed result above came from the original corrected checkpoints.
A byte comparison of nine source modules found that the encoder, datasets,
heads, RNG utilities, metrics and readers matched the archived snapshot.
The live training entry point, task model and optimizer differed. The FP32
readout and bias/norm decay settings above were checked against the live
correction and the saved settings for all three seeds.

The corrected reference used eight ranks, one image per rank, no gradient
accumulation, two loader workers per rank and 20 epochs. With the public BF16
execution option, the frozen encoder uses BF16 storage and autocast in both
training and evaluation. The reader and segmentation head retain FP32.
An entirely ignored training crop contributes a differentiable zero loss so
that rank can still participate in the update. Each rank evaluates the full
validation set once after training. The published scores are the arithmetic
means of those rank scores.

The corrected path saves the completed-epoch RNG state before its final
evaluation. Its checkpoint format is `basic5_corrected_ade_ap_epoch_v2`.
Earlier corrected checkpoints used intermediate validation passes that
advanced the random streams. Start fresh to use the new training sequence.

A reduced CPU check used a BF16 feature grid with autocast enabled around the
model. Against the captured V-JEPA reader and head, the portable path matched
initialization, logits, clipped gradients and parameters after three AdamW
updates exactly. This test starts at the feature grid.

A paired H200 check at seed 0 used the first validation image, the released
encoder and both complete model constructors. The normalized input, target,
BF16 encoder feature grid and initial segmentation head weights matched
exactly. Both paths had 1,723,670 trainable parameters and cross-entropy
5.00047636. The greatest absolute difference was 4.77e-7 across sampled
logits (every fourth spatial position), and 4.01e-8 over the clipped head
weight gradients. These observations describe one image before training.

The first three submissions stalled before receiving their first training
batch. All 16 loader workers in each job lacked a multiprocessing listener.
The launcher's temporary path exceeded the Unix socket address limit. A
standalone listener reproduced the failure and succeeded under a short path.
All three jobs ended with zero checkpoints. The retries use a short
job-private temporary path and check listener creation before launching.

### Full-data public rerun

The corrected public CLI trained on all 20,210 training images and evaluated
all 2,000 validation images on every rank. Three independent jobs each used
eight H200 GPUs, one image per GPU, two workers per rank and 20 epochs.
Distributed sampler padding produced 2,527 optimizer updates per epoch
(50,540 per run). All three final metric contracts and checkpoint hashes
verified. The ordered training and validation membership hashes matched
across seeds.

| ADE20K AP mIoU (%) | Seed 0 | Seed 1 | Seed 2 | Mean +/- sample SD |
| --- | ---: | ---: | ---: | ---: |
| Corrected original records | 39.24 | 39.15 | 39.01 | 39.13 +/- 0.12 |
| Public rerun | 39.19 | 39.12 | 39.21 | 39.17 +/- 0.05 |

Figure 3 reports 39.13 +/- 0.12. At its two-decimal precision the difference
in mean mIoU (public minus Figure 3) is +0.04 percentage points.
