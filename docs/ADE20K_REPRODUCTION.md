# Full-data ADE20K reproduction

The public revised CLI at commit `76a6554` trained the five Figure 3
backbones on both revised dense tracks, with seeds 0, 1 and 2. Each run
used one H200, the released backbone, six loader workers, 20 epochs and a
batch of eight. There were 20,210 training images and 2,000 validation
images. Dropping the incomplete training batch gave 2,526 updates per epoch
and 50,520 per run. The final-epoch comparison uses validation mIoU in
percentage points. SD is the sample standard deviation across the three seeds.

| Backbone | Track | Seed 0 | Seed 1 | Seed 2 | Public mean +/- SD | Figure 3 mean +/- SD | Public minus Figure 3 (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3-7B | DPT | 47.61 | 47.12 | 47.32 | 47.35 +/- 0.25 | 47.39 +/- 0.38 | -0.04 |
| DINOv3-7B | LoRA | 51.73 | 51.44 | 51.69 | 51.62 +/- 0.16 | 51.67 +/- 0.16 | -0.05 |
| VGGT-Omega | DPT | 42.73 | 43.09 | 42.84 | 42.89 +/- 0.18 | 42.86 +/- 0.15 | +0.03 |
| VGGT-Omega | LoRA | 44.33 | 44.40 | 44.70 | 44.47 +/- 0.20 | 44.46 +/- 0.24 | +0.01 |
| RAEv2 K7 | DPT | 46.56 | 46.72 | 45.43 | 46.24 +/- 0.70 | 46.24 +/- 0.34 | +0.00 |
| RAEv2 K7 | LoRA | 46.83 | 46.84 | 46.45 | 46.70 +/- 0.22 | 46.70 +/- 0.22 | +0.00 |
| V-JEPA 2.1-G | DPT | 37.71 | 37.98 | 38.19 | 37.96 +/- 0.24 | 38.26 +/- 0.12 | -0.30 |
| V-JEPA 2.1-G | LoRA | 45.01 | 44.88 | 45.19 | 45.03 +/- 0.16 | 44.99 +/- 0.13 | +0.04 |
| SigLIP 2-G | DPT | 42.90 | 43.20 | 43.27 | 43.12 +/- 0.19 | 43.03 +/- 0.30 | +0.09 |
| SigLIP 2-G | LoRA | 49.12 | 49.25 | 49.24 | 49.20 +/- 0.07 | 49.08 +/- 0.23 | +0.12 |

At two decimal places, 2 of ten means match Figure 3 and 1 of
ten cells match both mean and SD. Reference values were checked against
Figure 3 on manuscript page 4. The original three-seed records agree
with all ten printed reference cells. The source revisions recorded for
these native runs were v6 for LoRA and v9 for DPT.

The portable audit verifies the final trainable artifact, encoder digest,
exit status, recipe, complete epoch history, population counts and published
metric. An independent census of the 2,000 nearest-neighbor-resized
224-square validation masks found 92,007,036 labeled pixels across all
150 classes and 8,344,964 ignored pixels. All 30 original records report
the same labeled count. The portable Figure 3 audit now also rejects
different labeled-pixel counts, class counts and inconsistent pixel
accuracy. Counts alone do not identify a split by its image bytes.

The initial batch of scheduler jobs stopped at the read-only source mount
before starting training. All measured runs used fresh personal output
directories with a real mount point. No completed training was repeated
in this recovery.

The [submission scope](SUBMISSION_SCOPE.md) describes which experiments ship in the public package.
For the CLI, manifest and aggregation commands, see the
[revised adaptation guide](CVPR_ADAPTATION.md#auditing-portable-figure-3-training-outputs).

The largest absolute difference between the displayed means is 0.30 percentage points (V-JEPA 2.1-G, DPT).

## First training update

A separate seed-0 check ran the original and portable training entry points
through the first AdamW step for each of the ten model/track pairs. Target
masks matched byte-for-byte in all ten first batches. Sampled logits matched
exactly in 10/10 pairs and full-batch cross-entropy matched in 9/10.
These are first-update diagnostics; the table above uses independent
complete training runs.

On the same H200, two fresh runs of the native SigLIP 2-G LoRA entry point
at seed 0 produced identical target masks, initial head weights and sampled
logits. The sampled head gradients differed by up to 0.000109 and
the sampled head weights after the update by up to 2.56e-09.
This control measures repeatability of one update. It does not explain the
size or direction of the final-score differences.

## Independent scoring of the saved weights

For RAEv2 K7 and SigLIP 2-G DPT at seed 0, both scorers evaluated the
saved trainable weights with their released encoder on all 2,000 validation
images. The portable scorer recovered each recorded mIoU and pixel accuracy
exactly. The native v9 and portable scorers agreed on pixel accuracy and
the support counts, and differed by less than 1e-12 percentage points in
mIoU. For these two artifacts, the choice of scorer therefore leaves the
reported two-decimal result unchanged.
