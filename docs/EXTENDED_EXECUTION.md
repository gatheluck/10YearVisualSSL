# Extended accumulation and precision

See [package scope](SUBMISSION_SCOPE.md) for the distinction between original
experiments, portable components and verified numerical reproduction.

The [image](EXTENDED_CLASSIFICATION.md), [semantic](EXTENDED_SEGMENTATION.md)
and [video](EXTENDED_VIDEO.md)
LP/AP runners share this execution component. It applies to their five verified
providers; it does not add new dataset memberships or certify paper scores.
Omitting `execution` preserves one physical batch per update in FP32.
All three runners also accept `torchrun` for distributed execution and support
portable epoch-boundary continuation. Legacy experimental checkpoints use a
different format and are not imported.

## Explicit configuration

Add this top-level object as the value of `execution` in any runner's
configuration for DINOv3 or RAEv2 attentive probing:

```json
{"accumulation_steps": 8, "precision": "fp32", "tail_policy": "flush_scaled"}
```

On a CUDA device with native BF16 support, the same selection can use:

```json
{"accumulation_steps": 8, "precision": "bf16", "tail_policy": "flush_scaled"}
```

The caller must choose the physical `probe.batch_size` and accumulation count
from the matching run record. For example, physical batch 1 with eight
microbatches realizes batch 8; physical batch 8 with 32 microbatches realizes
batch 256. These are observed dense and classification run settings, not
universal choices for every model or dataset. Seed remains 0.

BF16 applies to **both training and evaluation forward passes**; classification
and pixel losses consume FP32 logits outside autocast. Trainable parameters and
optimizer state stay FP32. There is no FP16 scaler or automatic precision
fallback. CPU BF16 is not a supported experiment selection. A CUDA device
without native BF16 support fails before loading data or the backbone.

## Captured update semantics

| Provider | LP tail | AP tail |
|---|---|---|
| DINOv3 | `discard` | `flush_scaled` |
| RAEv2-K7 | `discard` | `flush_scaled` |
| SigLIP2-G | `discard` | `discard` |
| V-JEPA 2.1 | `discard` | `discard` |
| VGGT-omega | `discard` | `discard` |

Select the matching tail policy explicitly when using `execution`; a mismatch
is rejected. The table reflects the inspected training entrypoints, not a claim
that every historical result used this source revision. A dataset row still
needs its matching source, checkpoint, preprocessing and run identity.

Every microbatch loss is divided by the **full accumulation count**. At a full
boundary, AP clips the accumulated trainable gradients to global norm 1 before
the optimizer step; LP does not clip. A `flush_scaled` tail updates once with
that same divisor, even when fewer microbatches remain. It is not renormalized
to the shorter group size. `discard` drops the remaining gradients; they never
carry across epochs. An epoch that cannot produce any update is rejected.
The existing data loader still drops an incomplete **physical** training batch.
That is distinct from a tail of full microbatches in an accumulation group.

Learning-rate scaling uses physical batch times world size times accumulation
count. The source
schedule indexes **optimizer updates**, but its warmup and total horizon use
the full number of loader **microbatches** per epoch. For example, accumulation
8 advances the schedule approximately one eighth as far per epoch as count 1.
This is an observed source behavior; the component does not silently replace it
with a conventional effective-update horizon. Whether a particular final table
cell used this exact clock remains a run-provenance question. Protocol prose
about epoch-level cosine decay alone does not resolve that mapping.

The backbone stays frozen and in evaluation mode. Nonfinite losses or gradients
and unexpected backbone gradients fail the run. Terminal probe export still
excludes backbone weights and never overwrites an existing CLI output directory.

## Distributed launch and ownership

Use the same configuration on every rank, with `device: cuda` for NCCL or
`device: cpu` for Gloo. The launcher chooses each CUDA device through
`LOCAL_RANK`; an explicit `cuda:0` is refused in a multi-process launch.
For example, after preparing the image configuration and its sample manifest:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 -m downstream.extended_classification --config extended-image.json --out runs/extended-image-ddp
```

For semantic segmentation, replace the module with
`downstream.extended_segmentation` and supply its configuration. `probe.batch_size`
is **per rank**, not a global batch. Changing world size without adjusting the
physical batch or accumulation changes the realized batch and scaled LR.
The schedule horizon uses each rank's local loader length. Reported microbatch,
update and discarded-tail counters are per rank; synchronized updates are not
multiplied by the number of ranks.

Python, NumPy and Torch use the captured `seed + 1000 * rank` initialization;
rank zero and single-process initialization remain unchanged. DDP synchronizes
model state from rank zero after construction. The training sampler independently
uses seed 0 and `set_epoch(epoch)`, pads a non-divisible
population as PyTorch's captured `DistributedSampler` does, then distributes
indices by rank. Incomplete physical batches are dropped locally. Unlike the
single-process compatibility route's dedicated generator, the distributed loader
uses the global Torch RNG for iterator/worker seeding, matching the captured
loader. This is not complete historical RNG replay: the portable component does
not replay the experimental trainer's preflight forward/backward, and exact
random-transform parity still needs matching model construction, workers and runs.
Unlike the experimental trainer's small-population fallback, this package rejects a
population smaller than one full global physical batch; choose explicit smaller
settings instead of silently changing the batch or accumulation. The DDP wrapper
uses `find_unused_parameters=True`, including adapters with initially unused
parameters. Gradients synchronize on every microbatch, before accumulated AP
clipping and the optimizer update. Frozen encoder parameters remain excluded.

Only rank 0 evaluates the **complete** validation population, using the unwrapped
model. Validation is neither padded nor sharded; exported probe keys have no DDP
prefix and contain no backbone weights. Only rank 0 creates the output directory
and writes metrics, probe state, results and the final manifest. An existing
output is refused collectively and left unchanged. Configurations and membership
metadata must agree across ranks before training. Setup and evaluation errors,
and nonfinite loss/gradient guards, propagate across ranks. A process crash or
failure inside a data-loader/model collective still relies on the process-group
timeout (three hours, as in the captured launcher) and torchrun's worker termination;
long rank-zero evaluation must fit that window. This is not fault-tolerant
training. Launch one run per torchrun invocation. An already initialized matching
group can be borrowed, but the runner does not destroy a group owned by its caller.

## Epoch checkpoints and continuation

Both runners atomically publish `resume.pt` after every **completed training
epoch**, before final evaluation. The file contains head/adapter tensors,
optimizer and scheduler state, execution counters, and each rank's Python,
NumPy, Torch, local CUDA and loader-generator RNG states. Frozen encoder weights
are excluded. The encoder state is hashed once when starting or resuming, so
even changed weights at the same path are refused. Hashing large encoders adds
startup I/O and CPU transfer; checkpoint writes contain only the trainable probe
and its optimizer state. Keep sufficient space for the old and temporary files.

To continue, copy the previous JSON configuration, add the top-level field
`"resume": "runs/extended-first/resume.pt"`, and increase `probe.epochs` to
the desired **total** completed epoch count.

For example, changing `probe.epochs` from 10 to 100 runs epochs 11 through 100;
it does not run 100 additional epochs. The protocol schedule horizon stays
100 epochs for image classification, 50 for video, or 20 for semantic segmentation, including a
short initial invocation. The checkpoint epoch cannot exceed the requested
target. If it equals the target, all three runners skip training and retry evaluation
from the completed checkpoint. This allows recovery from a failed terminal
evaluation without repeating training; a new checkpoint is delivered as well.
Run the existing command with the copied configuration and a new output:

```bash
python -m downstream.extended_classification --config extended-resumed.json --out runs/extended-resumed
```

For semantic segmentation, use `downstream.extended_segmentation`. For
distributed continuation, use the same torchrun world size and device type as
before; every rank must be able to read the same checkpoint and dataset. Only
rank zero writes checkpoints and outputs. The source checkpoint is never
overwritten. Existing CLI output directories are still refused. If a write
fails, the previously published epoch remains intact; a partial temporary file
does not become a valid checkpoint. A failed run may therefore have a usable
earlier checkpoint but still has `status: failed`, not a successful experiment.

Only `resume`, the output directory, and the requested epoch count may change.
Configuration, membership manifest, class vocabulary, loader length, world
size, execution settings, recipe-derived learning rates, Torch version and
frozen encoder state must agree. Missing/corrupt states and optimizer restore
errors fail; there is no silent fresh start. Checkpoints are loaded with
`weights_only=True`. Use the same code and dependency environment and immutable
input assets: identity checks hash the membership manifest and encoder, **not
every image/mask file or every dependency/source file**. They do not certify
unchanged raw data bytes or replay across software versions.

Resume starts at the next complete epoch, with no saved partial gradients.
Mid-epoch progress is intentionally replayed from the last complete epoch;
there is no automatic restart, signal handler, mid-batch continuation or
walltime policy. Data-loader workers are recreated each epoch as before, not
persistent. The terminal `probe.pt` remains an inference export, not a resume
checkpoint. `results.json` adds the starting epoch and source-checkpoint SHA-256;
execution counters and metrics count all epochs, including restored progress.

The captured trainers provide the epoch/step and optimizer continuation basis,
but do not preserve these per-rank RNG states and may suppress optimizer load
errors. This portable format deliberately requires stricter state identity and
does not accept their legacy files, replay their preflight, or establish a
historical paper-score trajectory. The two formats must not be interchanged.

## Verification boundaries

The `results.json` execution record reports precision, accumulation count,
nominal effective batch, tail policy, scaled base LR, schedule clock/horizon,
microbatch count, optimizer updates, discarded microbatches and tail updates.
The nominal effective batch does not imply a flushed short tail was full-sized.
`canonical_eligible` and `record_value` remain false.

The 2026-09-29 audit inspected captured training sources and read current
experimental entrypoints without modifying originals. CPU tests compare exact
updates, clipping order, non-divisible tails, schedule indexing and frozen
parameters. A private harness also executes unchanged update statements from
five original loops: ten task/loop combinations over three epochs match the
portable parameters and per-update gradients exactly on reduced CPU inputs.
BF16 forward/loss boundaries are exercised with CPU autocast in an
explicit test harness; that is not evidence of CUDA numerical parity.
Two real CPU/Gloo ranks exercise both runners' LP/AP training, full-population
evaluation, output protection and failure propagation. A separate global-batch
oracle compares three epochs of synchronized updates for both accumulation-tail
policies, including sampler padding and AP clipping. These reduced tests are not
released-weight NCCL/BF16 numerical validation or multi-node evidence.
The September 30 continuation tests compare uninterrupted and resumed image and
semantic LP/AP runs, including momentum/AdamW state, stochastic samples,
accumulation tails, schedules and rank-specific RNG streams. Real two-rank Gloo
tests cover all four routes; a separate real-image CLI test includes a loader
worker and verifies the output contract. These fixtures do not use released
weights. Atomic write failures, changed identity, incomplete state and source
output protection have negative tests. Released-weight CUDA/NCCL continuation,
full-data scores and legacy checkpoint import remain unverified or unported. Existing
224-grid semantic evaluation and unresolved task/metric identities are unchanged.
