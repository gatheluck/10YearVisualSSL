# Full-data LoRA comparison targets

See the [package scope and replication terminology](SUBMISSION_SCOPE.md) for the
distinction between original results and portable validation.

Figure 3 reports ImageNet-1k and SSv2 top-1 accuracy and COCO bounding-box
AP in percentage points. Each entry is the mean and sample standard
deviation across seeds 0, 1 and 2. The three completed original records per
cell reproduce the displayed statistics. The table gives the manuscript
references for comparing public runs. Eleven cells also passed the original
single-code-identity audit. VGGT-Omega COCO and all three RAEv2 cells
each contain two code digests across seeds. Their arithmetic matches the
printed values, while their original implementation identity remains mixed.

| Model | ImageNet-1k top-1 | SSv2 top-1 | COCO bbox AP |
| --- | ---: | ---: | ---: |
| DINOv3-7B | 87.90 +/- 0.03 | 56.12 +/- 0.11 | 52.63 +/- 0.04 |
| VGGT-Omega | 85.49 +/- 0.05 | 68.12 +/- 0.10 | 42.61 +/- 0.16 |
| RAEv2 K7 | 87.34 +/- 0.01 | 48.70 +/- 0.15 | 47.37 +/- 0.11 |
| V-JEPA 2.1-G | 83.88 +/- 0.03 | 75.72 +/- 0.20 | 46.82 +/- 0.04 |
| SigLIP 2-G | 87.31 +/- 0.10 | 48.92 +/- 0.29 | 49.70 +/- 0.09 |

Each original run used eight ranks. ImageNet used an effective batch of 256
with one update per physical batch. SSv2 used eight clips per rank and four
accumulation steps, for an effective batch of 256. COCO used two images per
rank and an effective batch of 16. DINOv3 used 50 ImageNet epochs and 24
SSv2 epochs. The other models used 100 and 50, respectively. Each COCO run
used 12 epochs. The [schedule guide](REVISED_SCHEDULES.md) describes the
task-specific recomputation policy.

The public campaign uses the [verified full splits](CVPR_ADAPTATION.md#full-data-input-checks-for-the-remaining-lora-routes)
and the [completed-epoch continuation mechanism](CVPR_ADAPTATION.md).
Its workers restart from the checkpointed loader generator at each training
epoch. This differs from the original persistent worker streams. A complete
comparison requires an audited final metric contract for all three seeds,
including the source-checkpoint digest for each resumed seed.
