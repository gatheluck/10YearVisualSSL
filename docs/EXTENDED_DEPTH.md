# Extended NYUv2 depth execution

This is the NYUv2 LP/AP execution path for the Extended experiment family in
Tables 39–40. It integrates the five existing spatial providers, captured
depth head and loss, AdamW schedule, accumulation, distributed execution,
portable epoch continuation and final evaluation. It does not certify the
published scores. See [coverage](FINAL_PAPER_COVERAGE.md) and
[scope](SUBMISSION_SCOPE.md).

Use the [configuration example](examples/extended_depth.json), replacing local
data, split and encoder paths, then run:

`python -m downstream.extended_depth --config /data/depth.json --out /results/depth-lp`

For distributed execution:

`torchrun --standalone --nproc-per-node=2 --module downstream.extended_depth --config /data/depth.json --out /results/depth-lp-ddp`

Only the leader evaluates the entire validation population and publishes
artifacts. The output directory must be new. Set `resume` to an earlier
`resume.pt` and increase `probe.epochs` to continue in another output directory.
Data, splits, encoder, batch, precision and world size must remain unchanged.
The protocol horizon is 30 epochs; shorter component runs do not change it.
Checkpoints include the head/reader, optimizer, schedule and per-rank RNG,
excluding frozen encoder weights. See [execution](EXTENDED_EXECUTION.md) and
[continuation](EXTENDED_EXECUTION.md).

## Inputs and exact behavior

`data_root` is the labeled HDF5 `.mat` file, and `samples` is `splits.mat` with
one-based `trainNdxs` and `testNdxs`. Both captured storage layouts are supported:
images `[N,3,H,W]` with depths `[N,H,W]`, or images `[3,H,W,N]` with depths
`[H,W,N]`. Indices must be nonempty, integer, disjoint and cover the entire
file exactly once. Ordering comes from the supplied split file, never a
first-N or random split. The normal release contains 795/654 samples; smaller
complete files are permitted for component tests. Hashes, actual counts and
indices are recorded. Structural checks do not authenticate an official release.

RGB uses bicubic resize to 224 square, depth uses bilinear resize, with a paired
training horizontal flip. There is no color jitter in this Extended path.
Valid depths are finite and strictly between 0.1 and 10 metres. The raw RGB
interface is translated once into the portable providers' ImageNet-normalized
input convention; each provider retains its own internal normalization.

The 1x1 head applies `softplus + 0.001` **before** bilinear upsampling. The loss
is mean squared log difference minus the **full** squared mean log difference,
with the captured `1e-4` clamp. These differ from the existing BasicFive
component head/loss; neither path silently replaces the other. LP freezes the
encoder; AP additionally trains the provider-declared spatial reader. Choose
`attentive` with the matching `reader_profile` to use AP.

Read-only inspection found another experimental source copy with a half-mean
loss penalty. This entry point follows the inspected Extended execution tree's
full-variance definition. Mapping either revision to historical table cells
remains unverified; do not treat the other copy as an automatic correction.
The current rerun catalog has NYUv2 entries for two provider families. The
other three provider executions validate available component compositions,
not the original per-family NYUv2 run recipe. All five historical row mappings
remain unverified, including whether a result reuses a BasicFive run.

Evaluation pools valid pixels across batches and reports unaligned RMSE,
AbsRel and delta1 percentage. It retains the captured `[0.001,80]` evaluation
clamp. No median alignment or per-image averaging is applied. Unlike the
reference's zero fallback, empty-valid training batches and empty-valid total
evaluation fail explicitly; nonfinite predictions and invalid valid-targets
also fail. Empty-valid evaluation batches contribute no pixels.

LP/AP use the tracked 30-epoch recipes, effective-batch learning-rate scaling,
one-microbatch-epoch warmup and the captured optimizer-update schedule clock.
AP keeps provider-specific tail handling and gradient clipping. FP32 is the
default; explicit BF16 requires native CUDA support, with no precision fallback.

## Verification boundary

Reduced CPU provider routes, source comparisons, CLI artifacts and exact
epoch continuation are validation of this port. They are not released-weight
CUDA/BF16/NCCL validation, full-data measurements or run-to-table attribution.
Results remain `record_value: false` and `canonical_eligible: false`.
Other depth datasets, FT and legacy native checkpoint import are not integrated
in this entry point. In particular, SUN RGB-D's task conflict remains pending.
Original experimental implementations and historical results are separate from
the portable package's current integration status.
