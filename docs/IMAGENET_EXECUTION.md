# ImageNet execution and portable continuation

This opt-in component adds physical microbatches, gradient accumulation,
FP32/CUDA BF16, replicated distributed execution and completed-epoch continuation
to the eight BasicFive ImageNet LP/AP/FT families. It does not certify historical
table cells, import original resume files, or implement FSDP.
See [package scope and replication status](SUBMISSION_SCOPE.md) for the
distinction between portable support and original experimental results.

Select `execution` as in the [configuration example](examples/imagenet_execution.json).
The fields are `profile: captured_imagenet_v1`, positive integer
`accumulation_steps`, `precision: fp32` or `bf16`, and `tail_policy: discard`.
BF16 requires native CUDA BF16 support; there is no automatic precision fallback.
Configurations without `execution` retain the existing single-process behavior.
Select LP/AP [preprocessing](IMAGENET_PROBE_PREPROCESSING.md) or the provider's
[FT recipe](IMAGENET_FINETUNE.md) separately.

`probe.batch_size` is the physical batch **per rank**. The effective batch is
physical batch times world size times accumulation steps. Optimization uses that
effective batch for learning-rate scaling. The reference schedule requires 256
for LP/AP and 1024 for FT. Its horizon remains 100 times the complete per-rank
loader length, while its clock advances only on optimizer updates, as inspected
in the source. Do not substitute a ceiling-divided update horizon. Without the
reference schedule, the existing constant-rate component remains constant.

Every loss is divided by the full accumulation count. Only complete groups cause
an optimizer update; an incomplete group at the end of an epoch is discarded.
Clipping occurs after accumulation for AP/FT. An epoch that cannot produce any
update fails. `max_steps_per_epoch` must be zero: checkpoints represent completed
epochs, not truncated iterations. Each requested run covers 1 through 100 epochs.
With the reference schedule, a short run retains the full 100-epoch horizon.

| Provider family | Rank seed stride | Replicated distributed component |
| --- | ---: | --- |
| CLIP, SigLIP2, C-RADIO, Cosmos | 1000 | Enabled |
| DINOv3, RAEv2 K7, V-JEPA2.1 | 17 | Enabled |
| VGGT-Omega | 1 | Refused pending source reconciliation |

Use `torchrun` with explicit `device: cpu` or `cuda`. For example:

`torchrun --standalone --nproc-per-node=2 -m downstream.imagenet --config /data/two-rank.json --out /results/run-01`

Adjust accumulation to retain the intended effective batch when changing world
size. Only rank zero evaluates the full validation population and writes the
results, avoiding sampler padding in the metric. Training retains distributed
sampler padding and per-rank drop-last behavior. Omega's inspected engine wraps
a model but calls unwrapped forwards, so its distributed equivalence is not
assumed. Replicated DDP is not a replacement for original FSDP memory behavior;
released-weight CUDA/distributed numerical parity remains unverified.

## Continuing a run

Each completed epoch atomically publishes `resume.pt`. LP/AP checkpoints contain
the trained probe; FT checkpoints also contain the trainable encoder. Both save
optimizer/scheduler state, counters and per-rank Python/NumPy/Torch/loader RNG.
The original input checkpoint and any existing output directory are preserved.

To continue, add `resume: /results/previous/resume.pt`, increase `probe.epochs`
to the desired total, and choose a new output directory. Keep the training
configuration, data, initial encoder, precision and world size unchanged. The
runner checks ordered sample/label membership and image bytes; scanning a full
ImageNet tree therefore incurs input I/O before training. Checkpoint identity or
state mismatches fail rather than silently restarting. This strict portable
format is intentionally distinct from original experimental `resume.pt` files.

`results.json` records the actual execution counters, selected preprocessing,
optimization and continuation provenance. `canonical_eligible` and `record_value`
remain false. Exact portable interrupted/uninterrupted CPU tests do not establish
historical RNG identity, native checkpoint compatibility, full-data scores or
released-weight BF16/FSDP parity. Original source RNG/checkpoint completeness and
collective ordering vary; these limits remain separate from component support.
