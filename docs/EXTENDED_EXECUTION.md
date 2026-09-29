# Extended accumulation and precision

See [package scope](SUBMISSION_SCOPE.md) for the distinction between original
experiments, portable components and verified numerical reproduction.

The [image](EXTENDED_CLASSIFICATION.md) and [semantic](EXTENDED_SEGMENTATION.md)
LP/AP runners share this execution component. It applies to their five verified
providers; it does not add new dataset memberships or certify paper scores.
Omitting `execution` preserves one physical batch per update in FP32.
Distributed execution and native checkpoint continuation are not integrated.

## Explicit configuration

Add this top-level object as the value of `execution` in either runner's
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

Learning-rate scaling uses physical batch times accumulation count. The source
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

## Evidence and limits

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
Released-weight CUDA execution, full-data scores, distributed semantics and
historical checkpoint continuation remain unverified or unported. Existing
224-grid semantic evaluation and unresolved task/metric identities are unchanged.
