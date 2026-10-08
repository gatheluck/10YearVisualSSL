# BasicFive dense-task accumulation

The explicit `captured_dense_v1` execution block connects the existing ADE20K,
NYUv2 and COCO component runners to accumulated single-process LP/AP/FT updates.
Eight inspected provider families expose this policy: CLIP, SigLIP2, C-RADIOv4-H,
Cosmos3 Super, DINOv3, RAEv2 K7, V-JEPA2.1 and VGGT-Omega. This is execution
component coverage, not evidence that their complete experiments or table scores
have been reproduced. See [the coverage ledger](FINAL_PAPER_COVERAGE.md) and
[submission scope](SUBMISSION_SCOPE.md) for original experiments versus portable
components.

## Configuration and execution

Start with the task's example and set the local data and encoder paths:

- [ADE20K LP](examples/ade20k_segmentation_execution.json)
- [NYUv2 LP](examples/nyuv2_depth_execution.json)
- [COCO LP](examples/coco_detection_execution.json)

Run `python -m downstream.ade20k --config resolved.json --out output`, or replace
the module with `downstream.nyuv2` / `downstream.coco`. These examples require
the actual datasets and released encoder; they do not download weights. NYUv2
requires the official `splits.mat`. COCO remains object detection.

Use `profile: capture_basic5_components`, the matching BasicFive optimizer,
and this explicit block:

```json
{
  "execution": {
    "profile": "captured_dense_v1",
    "accumulation_steps": 8,
    "precision": "bf16",
    "tail_policy": "discard"
  }
}
```

`batch_size` is the physical microbatch. Effective batch is physical batch times
accumulation; optimizer learning rates use that effective batch. The examples
use effective batch 8 for ADE20K/NYUv2 and 16 for COCO. Select `fp32` explicitly
for CPU verification. BF16 requires native CUDA support and is refused before
data/model loading on CPU; there is no silent precision fallback.

AP requires `adaptation: attentive`, the provider's explicit reader profile where
applicable, and `basic5_frozen_v1`. FT requires `adaptation: finetune` and
`basic5_finetune_v1`; remove any AP-only reader selection. Keep the existing
provider-specific readout and native detection contracts described in
[downstream tasks](DOWNSTREAM.md). The execution block does not select or replace
an encoder, detector, augmentation, loss or evaluation recipe.

## Update behavior and evidence

Every microbatch loss is divided by the full accumulation count. Only complete
groups update parameters; short tails are cleared without an optimizer step.
An epoch with no complete group is refused. AP/FT clip all trainable gradients
to norm 1 at update boundaries. LP clips only the inspected SigLIP2 COCO route;
the other seven inspected LP detection routes do not clip. The detection
pyramid remains trainable while the frozen-encoder guard checks its body.

Explicit schedules retain the existing portable schedule formulas and full
loader microbatch horizons; scheduler steps occur only on optimizer updates.
The reference schedule still requires its reference effective batch. This port
allows complete, shorter runs up to 20/30/12 epochs respectively, with portable
epoch continuation described below. `max_steps_per_epoch` is refused with this execution
block. Without a schedule selection, the existing constant-rate component
behavior is preserved.

`results.json` records effective batch, precision, clipping, full schedule
horizon, actual microbatches, optimizer updates and discarded tails. FT reports
the batch-scaled base rate separately from layer-decayed parameter groups.
Results keep `canonical_eligible: false` and `record_value: false`. Existing
evaluation and its FP32 behavior are preserved; BF16 here selects training
forwards, not a new evaluation recipe. The runners still evaluate each epoch;
their complete RNG/validation trajectory has not been matched to native runs.

On 2026-10-07, isolated source update blocks from both the capture snapshot and
current source matched 288 CPU cases, including 1,536 gradient tensors and
1,152 updated parameter tensors. These use identical tiny models and losses;
they verify accumulation/clipping, not full provider/loss/optimizer parity.
Public tests exercise all 72 configuration combinations, independent update
loops, real tiny-backbone task runners and failure paths. Local, mutation and
CI outcomes are recorded in the PR; skips are not validated routes.

## Portable epoch continuation

Each explicit execution run writes `resume.pt` atomically after every completed
training epoch **and its existing validation pass**. The checkpoint contains
optimizer/scheduler state, update counters, Python/NumPy/Torch/CUDA and loader
RNG state. LP/AP retain the trained head/adapter and, for detection, the pyramid
and detector state; the frozen encoder is excluded. FT includes the full model.
Checkpoints can therefore be large, especially for FT.

To continue, keep the original resolved configuration, add
`"resume": "/local/previous-output/resume.pt"`, increase `probe.epochs` (or
`detector.epochs` for COCO) to the total desired count, and run the same module
with a **new, empty** output directory. The target must remain within the same
protocol horizon. Constant-rate and reference-horizon schedules retain their
original clocks; the separate dense AP schedule still requires its full horizon.
A target equal to the saved epoch evaluates without further training. The saved
checkpoint retains the original post-validation RNG boundary in that case.

Configuration (except the target epoch and resume path), Torch version, initial
encoder state, selected ordered samples, input bytes and annotations must match.
ADE20K hashes selected images/masks, COCO hashes selected images and complete
annotation files, and NYUv2 hashes its complete MAT container and split file plus
selected indices. Hashing can take time on full datasets; inputs must remain
unchanged during a run. Missing files or mismatches are errors, not warnings.
Nonempty output directories are refused by both the runner and CLI before they
can replace existing results. Checkpoint format, epoch, counters, model tensors,
optimizer and scheduler are checked by the shared continuation machinery.
The [shared state-integrity contract](CONTINUATION_INTEGRITY.md) also validates
optimizer group recipes, moment tensors and every saved RNG record before
restoring live state; it documents the remaining corruption-detection limits.

`results.json` includes content-digest membership and continuation provenance.
Private paths and actual checkpoint contents are runtime artifacts, not files to
commit or include in an anonymized submission. `basic5_dense_epoch_v1` is a
portable format: it does **not** import native/FSDP checkpoints, certify native
validation/RNG trajectories, or resolve the scientific differences below.
Fixture integration compares uninterrupted and interrupted LP/AP/FT runs for all
three tasks, including exact checkpoint tensors, optimizer/scheduler states and
RNG streams. It does not run all released encoders or establish paper-score parity.

## Remaining boundaries

Distributed dense execution, FSDP and native checkpoint import are not
provided by this block. Released-weight CUDA/BF16 execution, complete datasets,
and correspondence to the paper's recorded scores remain unverified.

Source/protocol disagreements remain pending. In particular, some original
dense LP schedules use a zero cosine floor while the portable reference schedule
uses the documented nonzero floor. Some original COCO implementations use the
outer epoch for milestones after update-count warmup, while others use the
optimizer update divided by loader length; those clocks diverge with
accumulation. NYUv2 implementations also differ in their logarithm clamp
(`1e-4` versus `1e-6`); the existing portable loss uses `1e-6`. The shared
source loss also changed its squared-log-mean coefficient from 1 in the snapshot
to 0.5 in the current source; the portable component uses 0.5. Historical
run-to-source attribution is still required. The new execution
block preserves these existing portable components and does not certify them
as reconciled native recipes or silently change them to match one source.
