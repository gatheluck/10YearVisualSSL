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

## Generate the 45 training configurations

Copy the [local path example](../configs/figure3-lora-paths.example.json) to
`paths.json` outside the repository. Set each value to the corresponding
absolute dataset root or released checkpoint file or directory on your
training host. The generator runs with the Python standard library and can
prepare configurations before the inputs are mounted. Actual input and model
validation happens in the training CLI.

Run the following from the repository with the downstream training environment
activated. Set `CAMPAIGN_DIR` to the private directory holding your path file.
Choose a new configuration directory for each invocation.

```sh
python -m downstream.figure3_lora_configs --paths "$CAMPAIGN_DIR/paths.json" --out "$CAMPAIGN_DIR/configs-local"
```

The output contains one JSON per task, encoder and seed. The following
example runs SSv2 seed 0 on one allocated node with eight GPUs. Use the
corresponding JSON and a fresh output directory for each remaining run.
The generator itself starts no training jobs.

```sh
torchrun --standalone --nnodes=1 --nproc-per-node=8 -m downstream.revised_tasks \
  --config "$CAMPAIGN_DIR/configs-local/ssv2-dinov3_hf-s0.json" --out "$CAMPAIGN_DIR/runs/ssv2-dinov3_hf-s0"
```

The [five-encoder panel](../configs/figure3-lora-models.json) records the
released architecture, geometry and short schedules. The configurations
select the two recorded short schedules, eight clips per
rank for SSv2, and the public campaign's worker counts and restart policy.
For a time-limited allocation, follow the [completed-epoch continuation
instructions](CVPR_ADAPTATION.md) and preserve the preceding output directory.

## Compare public outputs with Figure 3

The reporting CLI covers all 35 LoRA/DPT cells across the five released
encoders. It reads the [printed means and sample standard deviations](../configs/figure3-adaptation-targets.json)
and the same model panel used to generate the training configurations.
Start with this private plan:

```json
{
  "schema": "portable_figure3_comparison_v1",
  "cells": [
    {"model": "raev2_k7", "task": "coco", "track": "lora", "runs": []}
  ]
}
```

For each finished seed, append an object to its cell's `runs` list with
`seed`, `config` (the final attempt's training JSON path), `out` (its artifact
directory), and `exit_status` (the recorded process exit code). Use absolute
paths or run from the directory against which those paths were recorded.
Supply one cell per model/task/track. From the training environment, run:

```sh
python -m downstream.revised_reporting --config comparison-plan.json --out new-comparison.json
```

The JSON always lists the full panel. Omitted cells are `missing`. A valid
subset of seeds is `incomplete`, with the audited individual scores.
An incompatible configuration, failed contract or inconsistent set of seeds
is `invalid`. Only three valid seeds (0, 1, 2) produce an observed mean and
sample SD. A `match` requires equality of **both** printed values. Otherwise
the row is a `mismatch`. Inspect these statuses after running the command.

The comparison checks the released provider and its declared geometry,
source-selected schedule, recorded configuration and manifest, and the
validated public execution topology (one GPU for dense tasks, eight for the
other three, four accumulation steps for SSv2, BF16). It reuses the full
[portable artifact audit](CVPR_ADAPTATION.md#auditing-portable-figure-3-training-outputs).
NYUv2 uses unaligned RMSE in meters to three decimals. Other tasks use
percentage points to two decimals, including COCO bbox AP. Numerical
agreement alone does not establish historical run or checkpoint identity.

On 2026-11-24 the CLI read the 60 completed ADE20K/NYUv2 runs (including the
corrected depth runs) and five completed single-seed runs from the other
three tasks. Both printed statistics matched in 1/10 ADE20K and 4/10 NYUv2
cells. The remaining 15 dense cells had a numerical difference. Five
classification/detection cells were incomplete and ten had no completed
seed at this checkpoint. The [ADE20K results](ADE20K_REPRODUCTION.md) and
[corrected NYUv2 results](CVPR_ADAPTATION.md) give the underlying comparisons.
