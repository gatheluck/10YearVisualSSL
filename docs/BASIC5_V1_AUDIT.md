# Basic Five v1 protocol audit

See [submission scope](SUBMISSION_SCOPE.md) for the limits of result attribution.

The three author-supplied Markdown files and three JSON files match the inspected
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
| SSv2 | 50 | SGD .1 / AdamW .001 / AdamW .0005 | The specification uses 16 segments, one evaluation view and no flip. LP and FT have video components. The AP reader and input tokens used for the revised manuscript are still pending. |

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

Eight cells still lacked an independently matched three-seed set: VGGT-Omega
ImageNet AP, both COCO and both NYUv2 tracks, SigLIP 2-G ImageNet AP and both
NYUv2 tracks. The original Omega NYUv2 runs give different means, and the
corrected depth files examined were unreadable. The corrected Omega COCO LP
files examined were also unreadable. Access failures do not imply missing runs.
In the supplied manuscript, SigLIP 2-G NYUv2 LP is **0.848 +/- 0.007**;
the earlier figure table shows **2.023 +/- 0.032**. Use the manuscript as the
display source. Numeric agreement with a separate LoRA/DPT control would not
establish which run produced this LP cell.

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

SpaceNet, SUN RGB-D, the revised SSv2 AP reader and Figure 7 reference-free
dimensionality remain pending. This audit does not change their interpretation.
