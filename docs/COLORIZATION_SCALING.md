# Indexed colorization scaling training

See [submission scope](SUBMISSION_SCOPE.md), [paper terminology](PAPER_TERMINOLOGY.md)
and the [frozen scaling evaluation providers](SCALING_BASIC5_EVALUATION.md).
This entry implements fresh and portable-continuation ImageNet-21K Winter 2021 colorization training for
the 30/60/90-epoch trajectories in Appendix B.9 and Table 20. The original
experimental implementation exists; this is its separate portable integration,
not a claim that the paper's numerical results have been reproduced here.

## Recipe

- Explicit ViT-B/16 or ViT-L/16, one-channel input and a 313-class spatial
  decoder. Reuse the existing model components while selecting the original
  scaling initialization order: construct the decoder before initializing CLS
  and position tokens, then initialize module weights. Existing Step-2 defaults
  retain their previous initialization behavior.
- Resize 256, random crop 224 and horizontal flip; original float32 gamma
  conversion followed by float64 Lab arithmetic; L/100 input and the bundled
  313-bin lookup-table targets. No runtime download or approximate bins.
- Compute class weights from up to 10,000 unique membership indices using
  NumPy RandomState(42), resize/center-crop and mean-normalized inverse smoothed
  frequencies with lambda 0.5. Save the resulting weights in the run.
- AdamW, betas 0.9/0.999, weight decay 0.05, weighted per-pixel cross entropy,
  gradient clipping 1.0. The epoch schedule uses ten warmup epochs and a fixed
  90-epoch cosine horizon, including the original float32 cosine evaluation.
  Peak and minimum rates are respectively `0.00015 * global_batch / 256` and
  `1e-6 * global_batch / 256`. At the original batch of 1,024 these are 0.0006
  and 0.000004. An early stop does not shorten the horizon.
- Explicit FP32 or CUDA FP16 autocast plus GradScaler. This is not the BF16
  rotation recipe. One update attempt per batch, with no accumulation option.
  AMP overflow skips the update and backs off the scale through GradScaler.
- Single process or DDP, distributed sampler with dropped tail, rank-local
  BatchNorm as in the source, and propagation of read/nonfinite-loss failures before
  gradient collectives. DDP averages each rank's weighted mean loss gradient;
  this differs from globally normalizing a concatenated weighted batch.

## Running and outputs

Edit the [resolved example](examples/colorization_scaling.json) with explicit
input roots. Use the method's locked environment, from the repository root:

`PYTHONPATH=. python methods/03_colorization/color_scaling.py --config /path/to/resolved.json --out /path/to/new-output`

For an independently verified allocation use `torchrun` with the same script.
`microbatch` counts images per rank; global batch is `WORLD_SIZE * microbatch`.
The inspected original B/L configurations both use 32 ranks and 32 images per
rank, seed 42. Smaller test geometries are component runs. The entry chooses no
queue and submits no jobs; use only an independently verified reservation.

The shared indexed reader freezes membership and excludes ImageNet validation
filenames. Original inputs are read-only. This is not a content-level overlap
audit. Existing output directories and input/output overlap are refused.

Outputs include membership records, `class_weights.pt`, atomic
`checkpoint_latest.pth`, head-free `encoder.pt`, metrics, a contract manifest
and `results.json`. At completed epochs 30, 60 and 90 it also retains
`checkpoint_epoch_<epoch>.pth` and `encoder_epoch<epoch>.pt`. Encoder files load
through the existing frozen colorization scaling provider. Checkpoints include
model/optimizer/scaler state, per-rank RNG, configuration and membership.
`global_step` counts actual optimizer updates; `optimizer_attempts` also includes
AMP-overflow attempts skipped by GradScaler.

For portable epoch-boundary continuation, set optional `resume` to an immutable
`checkpoint_latest.pth` produced by this entry and choose a new output directory.
`epochs` is the total target epoch, strictly greater than the saved epoch. The
`color_scaling_epoch_v2` format binds the resolved recipe, frozen membership,
world size, steps, geometry and Torch/NumPy versions. Model, optimizer, scaler,
per-rank RNG and rank-local BatchNorm buffers are restored. Malformed counters,
nonfinite state, altered class weights or changed recipes are rejected before
training. All-overflow AMP epochs can resume with zero optimizer updates.

Legacy v1 and native historical checkpoints are deliberately rejected: their
saved fields do not establish the required identity. Do not discard those
originals. Identity covers manifest metadata, not every byte in indexed tar
payloads, and does not certify cross-version/platform equivalence. Worker
scheduling and historical sample order remain separate from portable resume.

## Evidence and limits

The selected source/configuration files at Capture revision
`d8e82adc094ca68624d0f98bd6425c9ec976966f` match freshly read originals. The
bundled quantization constant is byte-identical to the original. Independent
CPU FP32 comparisons execute the original model and data functions and match
initialization, augmented inputs, class weights, two updates' parameters and
gradients exactly; eight schedule points also match.

Tests exercise fresh training contracts, retained 30/60/90 exports, frozen
feature equivalence, rejection of invalid configuration and unreadable input,
and two-rank CPU updates against a rank-weighted oracle. AMP sequencing is
tested with both a simulated scaler and real CPU GradScaler overflow/recovery,
including two ranks. A separate reserved-node CUDA FP16 diagnostic ran full-size
B/L models on four synthetic indexed images: continuous two-epoch training and
one epoch plus continuation matched model, optimizer, scaler, RNG, buffers and
counters exactly. This does not establish historical numerical parity,
multi-node execution, full-data runs, paper scores or historical run attribution.

The portable loader uses a dedicated generator and restarts workers each epoch;
historical persistent-worker sample order is not established. Bad images and
nonfinite losses or FP32 gradients fail rather than silently producing gray
substitutes or skipped batches. AMP gradient overflow retains native GradScaler
skipping and scale backoff. Reduced physical batches scale the learning rate by their
actual size; the original diagnostic smaller-world invocation could retain the
configured 1,024-image learning rate. Only matching original geometry is a
recipe comparison. All outputs remain `canonical_eligible: false`.

[SplitBrain scaling training](SPLITBRAIN_SCALING.md) now has its own explicit
recipe. Colorization portable continuation is tested for exact model, optimizer,
RNG and final-loss equality in single-process CPU runs and model/buffer equality
in two-rank CPU runs. These results do not imply native historical continuation.
