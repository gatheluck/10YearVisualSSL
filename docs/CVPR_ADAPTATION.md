# CVPR adaptation checkpoint

This checkpoint concerns the 15-page revised manuscript inspected on
2026-11-18. Its referenced appendices were not supplied; historical ICLR
protocols must not be treated as the new recipes. See
[package scope](SUBMISSION_SCOPE.md) for the distinction between experimental
implementations, portable components and result reproduction.

## Attention LoRA component

`downstream.lora` implements explicit attention-projection adapters. Call
`inject_attention_lora(backbone, block_groups, rank=16, alpha=16)` with the
actual transformer blocks, after loading the base checkpoint and placing it
on the intended device. It freezes the backbone and returns a mapping from
full projection paths to adapters. Create the optimizer afterwards, including
the task head separately. Do not cast the whole adapted model to a lower
precision: adapter parameters are intended to remain float32.

The update is `W(x) + (alpha / rank) * B(A(x))`. A uses Kaiming uniform
initialization and B starts at zero. Base weights are unchanged; adapted
checkpoints include base parameters and both adapter matrices. Recreate the
same adapter structure before loading their state dictionaries.

Supported projection leaves are `qkv`, `proj`, `q_proj`, `k_proj`, `v_proj`,
`o_proj` and `out_proj`, under a path containing `attn` or `attention` inside
each supplied block. Each block must expose two or four eligible linears.
The function checks membership, repeated blocks, target counts and shared
selected projections before freezing or replacing modules. It does not infer
transformer blocks or validate arbitrary attention architectures.

Call `assert_adapter_usage(adapters, reset=True)` after a representative forward
to detect paths that read a projection's base weight without calling its
adapter. This is a call check, not proof of nonzero gradients, task coverage or
numerical reproduction. Check gradients and updates separately. The read-only
`weight` and `bias` properties exist for compatibility with introspection;
functional attention bypasses must be corrected in a model-specific integration.

The inspected experimental implementation uses the same equation, parameter
precision, initialization and projection selection. On CPU, float32, float16
and bfloat16 comparisons matched initialization, outputs, input/adapter
gradients and three AdamW updates exactly. Public behavioral tests independently
exercise the equation, freezing, selection, refusal atomicity, bypass detection
and checkpoint round trips. CUDA, distributed training and full-model/task
runs are not validated by these component checks.

The [revised task CLI](#revised-image-and-semantic-task-cli) connects these
components to seven explicitly versioned task/track routes. Older task defaults are
unchanged. Model bindings alone do not establish every task recipe, released
checkpoint result or continuation behavior.

## Training-facing checks and four-tap dense probe

The follow-up adds three shared modules, without relabeling an old task recipe:

- `downstream.adaptation_runtime.adapter_parameter_groups` separates head and
  adapter learning rates and decay/no-decay groups. It includes only trainable
  parameters, rejects duplicates and unexpected non-head/non-adapter parameters,
  and preserves a `base_lr` per group for a task-owned scheduler. As in the
  reference, one-dimensional parameters and names containing `norm` have no
  decay. The caller must supply verified task rates and optimizer/schedule;
  this helper does not select an algorithm or infer a manuscript recipe.
- `check_adapter_effect(forward, adapters)` runs a representative feature
  forward, perturbs adapter B matrices, restores them, and repeats the forward.
  All projections must be called, and the perturbation effect must exceed
  restoration drift by the reference threshold. A called-but-discarded adapter
  is not sufficient. The caller controls inputs, evaluation mode and autocast.
  B matrices and call counters are restored even on exceptions; CPU and selected
  CUDA RNG states are preserved. This is a feature-effect check, not a backward
  gradient test. MPS RNG preservation has not been verified.
- `downstream.feature_taps` collects four explicitly selected blocks during one
  forward and removes its temporary hooks on success or failure. It preserves
  gradients and existing hooks, takes the first element of tuple/list block
  outputs, and refuses missing taps. `even_taps(L)` uses `ceil(k*L/4)-1` for
  `k=1..4`, requiring at least four blocks. `tokens_to_grid` validates the
  caller's grid and prefix before removing special tokens. These helpers do
  not choose model-specific blocks, grids or normalization.

`downstream.dpt_probe.DPTProbe(in_dims, out_channels=..., hidden=512,
positive=False)` accepts four BCHW feature maps and a target `(height, width)`.
It adds channel-wise LayerNorm to each tap and runs the pinned upstream DPT,
capturing its final convolution **before** the upstream fixed resize. It then
resizes directly to the target with bilinear interpolation and
`align_corners=False`; positive depth adds `softplus + 0.001`.
Set `positive=True` for that depth readout. Inputs are not modified in place.
The unchanged upstream forward still computes its final resize, whose result
is discarded; this eager compatibility wrapper has that extra compute/memory
cost. Concurrent/reentrant forwards on the same instance and compiled execution
have not been validated. State-dictionary round trips are tested; recreating
the same architecture is required before loading.

Initialize recursive submodules before using DPT. The
[license inventory](THIRD_PARTY_LICENSES.md) binds the upstream revision and
retains both the direct and antecedent notices. No DPT upstream source was
copied into the package's own modules.

CPU comparisons with the captured dense head matched initialization, outputs,
input/parameter gradients and three AdamW updates exactly for both semantic
logits and positive depth, using reduced dimensions. Optimizer grouping, three
updates, feature-effect metrics, tap tensors and grid ordering also matched the
captured function bodies. Those functions were loaded in isolation because the
original module imports the experimental environment; this is not evidence of
the complete original trainer executing locally. Native CUDA autocast was
disabled by PyTorch in the CPU diagnostic and is not validated here. The
portable diagnostics additionally restore state and clean up hooks on errors.

The task CLI below adds inspected readers and schedules for all five tasks,
with portable completed-epoch continuation. See the dated GPU validation below
for released-weight and continuation checks; full-data execution remains separate.
The model-specific tap bindings below provide component access, not proof of
complete execution across every model and task.
In particular, the generic evenly spaced rule must not override the captured
VGGT tap selection or RAEv2's normalized intermediate-layer readout.

## Explicit model bindings

The provider APIs `adaptation_block_groups()`, `enable_adaptation_grad()` and
`forward_dpt_features(images)` bind the inspected source to the existing model
loaders. Inputs use this package's **ImageNet-normalized BCHW images**, not the
raw RGB inputs of the experimental adapter. Existing provider conversions apply
normalization exactly once; loading local checkpoints is unchanged.

| Provider | LoRA blocks | DPT features |
| --- | --- | --- |
| SigLIP2 vision tower | Unique full-depth vision block list | Raw outputs of four evenly spaced blocks; no prefix |
| DINOv3 HF vision tower | Unique full-depth vision block list | Raw evenly spaced block outputs, removing CLS and register tokens |
| V-JEPA 2.1 | Encoder blocks | Raw evenly spaced blocks, image T=1, no prefix |
| VGGT-Omega | Both frame and inter-frame block lists | Cached layers 4, 11, 17, 23; concatenate features as returned by the aggregator and remove its prefix |
| RAEv2 K7 tokenizer | The tokenizer's 24 blocks | Official normalized intermediate-layer maps at evenly spaced indices; not raw hooks or the K7 aggregate |

The RAEv2 row is explicitly the **tokenizer experiment source**, not the revised
paper's generator block-29/time-0.1 LP/AP representation. No generator weights,
identity or results are silently substituted. The projected-CLS HF family has
no new binding in this change.

After building and placing a frozen provider, call
`install_backbone_lora(backbone, rank=16, alpha=16)` from
`downstream.adaptation_runtime`. It uses the provider's explicit block groups,
installs adapters, then enables gradients through the frozen base in eval mode.
Create the head and optimizer afterwards. Calling `train()` on the parent must
not turn the adapted backbone's dropout or rotary augmentation back on. Adapter
weights stay float32; do not cast the complete adapted model afterwards.
Check representative feature effects and backward gradients separately.

For frozen dense probing, `FrozenDPTModel(backbone, in_dims, out_channels=150)`
composes those taps with the existing head. For a positive depth output, select
`out_channels=1, positive=True`. The caller declares all four tap channel sizes;
use `hidden=512` for the inspected head. Calling `model(images, (height, width))`
returns dense predictions at the requested resolution. The backbone remains
frozen and in eval mode when the head trains; feature collection uses no-grad.
This composition deliberately does not select depth alignment, loss, dataset,
optimizer, scheduler or a canonical experiment. Those remain task-level work.

The new behavioral tests exercise reduced real upstream/HF models, raw versus
normalized tap selection, prefix removal, frozen-base updates, gradient flow,
eval-mode preservation and invalid bindings. These tests do not certify released
weights, CUDA/BF16, distributed execution or paper scores. Consult the PR for
the current reference comparison, mutation, whole-suite and CI results.

## Revised image and semantic task CLI

`python -m downstream.revised_tasks --config config.json --out runs/new-run`
provides a separate `b5x_lora_dpt_v2` profile. It accepts exactly these routes:

| `task` | `track` | Epochs | Effective batch | Head LR | Adapter LR | Warmup epochs |
| --- | --- | --- | --- | --- | --- | --- |
| `imagenet` | `lora` | 100 | 256 | 0.001 | 0.000125 | 5 |
| `ade20k` | `lora` | 20 | 8 | 0.001 | 0.0001 | 1 |
| `ade20k` | `dpt` | 20 | 8 | 0.0005 | None | 1 |
| `ssv2` | `lora` | 50 | 256 | 0.001 | 0.0005 | 5 |
| `coco` | `lora` | 12 | 16 | 0.02 | 0.02 | 0 |
| `nyuv2` | `lora` | 30 | 8 | 0.001 | 0.0001 | 1 |
| `nyuv2` | `dpt` | 30 | 8 | 0.0005 | None | 1 |

The portable `imagenet` task name corresponds to `imagenet1k` in the inspected
source protocol. All routes except COCO use AdamW `(0.9, 0.999)`, weight decay `0.05` and
global trainable-gradient clipping at `1.0`. Bias/one-dimensional parameters
and names containing `norm` have no decay. LoRA rank and alpha are both `16`;
DPT uses hidden width `512`. The head and adapters retain float32 parameters.
Each AdamW parameter group has its own absolute linear warmup from `1e-6` followed
by cosine decay to `1e-6`. Scheduling happens before each zero-based optimizer
update; the floor is reached at the reference horizon, not one update early.
Diagnostic truncation does not shorten that reference schedule.

COCO uses the inspected `B5X_LORA_LP_v1` optimizer, retained by the v2 source:
SGD, momentum `0.9`, weight decay `0.0001`, and the same scaled rate for head
and adapter parameters. Its first 500 updates use the factor
`0.001 + 0.999 * step / 500`; thereafter epochs 8 and 11 apply factors 0.1
and 0.01. Warmup takes precedence even in a tiny diagnostic run. The source
uses global gradient clipping at 1.0 here too. This is bounding-box detection,
not category-presence classification.

Save this **template** as `config.json`. Replace the provider placeholder and
local paths before running. The shown backbone fields illustrate a local
HF-style vision snapshot: `encoder` contains its configuration and weights;
`img_size` and `patch_size` describe that checkpoint's declared geometry,
not a configurable task crop. Other providers require their own existing
checkpoint fields. Pin the local checkpoint and recursive submodules, and
follow the provider's loading documentation. This template does not download
weights or grant redistribution rights; see the
[license inventory](THIRD_PARTY_LICENSES.md).

```json
{
  "profile": "b5x_lora_dpt_v2",
  "task": "imagenet",
  "track": "lora",
  "seed": 0,
  "device": "cuda",
  "workers": 4,
  "data_root": "/path/to/dataset",
  "backbone": {
    "kind": "<vision-provider>",
    "arch": "released",
    "encoder": "/path/to/pinned/local/vision-snapshot",
    "img_size": 384,
    "patch_size": 16
  }
}
```

For ADE20K LoRA, change `task` to `ade20k`; for ADE20K DPT, also change `track`
to `dpt`. Both readers produce fixed 224-pixel task inputs with exactly one
ImageNet normalization before the provider's own normalization conversion.
ImageNet expects `train/<class>/` and `val/<class>/` directories with matching
class mappings. Training uses bicubic random resized crops and horizontal
flips; validation resizes the short edge to 256 and center-crops 224.
ADE20K expects `images/training`, `images/validation`, `annotations/training`
and `annotations/validation`. It uses the inspected Python-RNG scale/crop/flip
sequence, bicubic images and nearest-neighbor masks; labels 1 through 150 map
to 0 through 149 and other labels are ignored. Validation is a square 224
resize. These readers do not inherit interpolation defaults from older task
routes. `workers` is an explicit nonnegative loader-worker count, defaulting
to zero.

For SSv2 use `task: ssv2`, `track: lora`: inputs are 16-frame clips from
`videos/<id>.webm` (or `.mp4`) and `labels/{labels,train,validation}.json`.
Sampling uses Python RNG; one bicubic crop is shared by all frames, without
horizontal flipping. Decoding uses Decord; the Linux downstream lock includes
a pinned, hashed package. The captured source uses this decoder but did not
provide its exact installed version, so decoder-version identity with the
historical environment is not certified. On platforms without a compatible
Decord installation, the video route cannot run; it never silently switches
to a different decoder. The real-codec test is selected in Linux downstream CI;
local macOS checks use injected decoded frames and do not certify real decoding. Missing videos are counted in `missing_populations`,
never replaced with a different clip. Full reference execution requires no
missing videos. `micro_batch_size` is an SSv2-only positive integer (default
8, capped to the per-rank effective batch for a reduced diagnostic). Effective
batch must divide exactly by `micro_batch_size * world_size`. Incomplete
accumulation groups are not updated; the first unused tail batch is read,
as in the source, to preserve its RNG effect.

COCO uses `task: coco`, `track: lora`, `images/{train,val}2017` and
`annotations/instances_{train,val}2017.json`. The source Faster R-CNN has
91 outputs, contiguous foreground labels, and provider-specific FPN and
normalization before image resizing/padding. Crowd and <=1-pixel boxes are
excluded from training; evaluation uses official bbox COCOeval and reports
AP, AP50, AP75 and size-specific AP in percent. `result.json` retains these
source percentages; contract `metrics.json` uses the existing `coco_map*`
fraction-valued keys (including the COCOeval undefined-size sentinel). Outputs 81 through 90 are
excluded from the 80-category result mapping. Raw RGB is normalized only by
the detector; it must not receive the other readers' pre-normalized tensors. The adapter-effect
diagnostic uses that same detector transform before inspecting the backbone
feature map, without depending on the newly initialized FPN. Arbitrary COCO
image dimensions are padded before patch-aligned providers are invoked.

NYUv2 uses `task: nyuv2`, with either `track: lora` or `track: dpt`, and
`labeled/nyu_depth_v2_labeled.mat` plus `labeled/splits.mat`. HDF5 handles are
worker-local. Upright RGB and depth are resized to 224; valid training pixels
are finite and strictly between 0.1 and 10 metres. LoRA uses a positive 1x1
readout before resize; DPT applies positivity after resize. The SILog loss
has no square root and uses coefficient 0.5. Primary metrics use per-image
median alignment and global valid-pixel aggregation; the [paper reporting guide](REVISED_DEPTH_REPORTING.md) selects unaligned RMSE
for the LoRA and DPT manuscript columns.

Without a `diagnostic` object, the entry point requires `arch: released`
(V-JEPA2.1 instead uses its native `vit_giant_xformers` constructor, a local
checkpoint, 384-pixel input, 16-pixel patches and no dimension overrides) and
the reference populations: ImageNet 1,281,167 training / 50,000 validation
images with 1,000 classes; ADE20K 20,210 / 2,000; SSv2 168,913 / 24,777
with 174 classes; COCO 118,287 / 5,000 with 80 categories; NYUv2 795 / 654. These checks are necessary
input checks, not proof of a paper result. For a reduced test, add an explicit
`diagnostic` object, for example
`"diagnostic": {"epochs": 1, "batch_size": 2, "max_steps": 3}`.
Optional diagnostic fields also include `max_train_samples`, `max_val_samples`
and `dpt_hidden`; COCO additionally accepts `detection_size` and
`detection_proposals` to reduce detector compute explicitly. Diagnostic batch changes linearly scale base learning rates
while preserving absolute warmup/floor rates and reference-batch metadata.
Top-level epoch/rate overrides are rejected rather than silently changing the
reference recipe. The output directory must not already exist.

ImageNet, SSv2 and COCO can be launched with
`torchrun --master-addr=127.0.0.1 --master-port=29517 --nproc-per-node=2 -m downstream.revised_tasks --config config.json --out runs/new-distributed-run`.
This is a **single-host** example; ensure port 29517 is unused or select another
unused port. The tested local launcher needs explicit IPv4 loopback: automatic
`--standalone` rendezvous stalled in that environment. Multi-node launch needs
administrator-approved reachable addresses and network configuration, not a
loopback address shared across hosts.
Its effective batch must divide evenly across ranks; training drops incomplete
batches, and validation partitions samples without duplication. ADE20K and NYUv2 follow
the inspected single-device trainer and rejects multi-rank execution. These
interfaces are not evidence of released-checkpoint or multi-GPU validation.

Successful execution writes `result.json`, contract metrics/manifest and
`trainable.pt` containing only trainable parameters. **`trainable.pt` is not a
resume checkpoint**. For exact portable completed-epoch continuation, set
`continuation: true` and optionally `stop_after_epoch: N`. Use `workers: 0`
or set a positive `workers` count together with
`"worker_lifecycle": "reseed_each_epoch"`. The latter recreates workers for
each training epoch from the checkpointed loader generator. The original
persistent-worker runs can therefore draw different augmentations. The CPU
integration check compares complete checkpoint contents after a stop and
restart with a real image loader and one worker.
Each completed epoch writes a separate atomic `resume.pt` before validation.
Resume into a **new** output directory with `resume: /path/to/resume.pt`,
keeping the original recipe and diagnostic horizon; remove or increase the
stop boundary. A checkpoint already at the target is evaluated/exported
without another optimizer update. `max_steps` is incompatible with continuation.
Loss logs explicitly identify `loss_scope: rank0`, as in the source; the leader
loss is broadcast before a common checkpoint is validated and published.

Continuation validates actual frozen state, checkpoint input bytes, ordered
train/validation membership and file bytes, recipe, optimizer groups, scheduler
position, precision, world size, runtime versions and loader policy. It restores
trainable parameters/buffers, optimizer state and each rank's Python/NumPy/Torch
and loader-generator states only after collective preflight. Hashing large
inputs costs an initial full read. Ordered metadata is streamed into the existing
SHA-256 encoding, bounding temporary metadata memory while preserving the
existing digest.
Private input paths appear only inside identity digests, not in saved metadata.
For a resumed Figure 3 seed, retain its source checkpoint. The cell audit checks
its bytes against the recorded source digest before pooling seeds with distinct
resume paths. Frozen weights are not copied into the checkpoint. Worker-prefetch state,
mid-epoch continuation, changed world size and native experimental checkpoint
imports are deliberately rejected. This is portable continuation, not a claim
of an identical historical native-checkpoint trajectory. `result.json` retains `record_value: false`
and `historical_run_verified: false` for every run. `checkpoint_sha256` binds
the local checkpoint input bytes through a digest without retaining their
filename or path; the digest identifies an input, not a verified correspondence
to a paper result.
`status: diagnostic` marks
an explicit diagnostic configuration; `status: component_complete` records a
completed non-diagnostic component run, not a certified paper score. A non-diagnostic
run stopped at an earlier epoch is `component_partial`. Inspect
`training_complete`, populations, update counts, precision and recipe fields
as well as the manifest; an `ok` manifest alone is insufficient for reproduction.

CPU execution uses float32. CUDA uses bfloat16 autocast with float32 adapters and heads. The
VGGT-Omega aggregator retains its loaded float32 parameters and buffers under
autocast. DINOv3-7B loads bfloat16 parameters directly while the loader
constructs float32 rotary frequencies. Other providers retain the existing
bfloat16 backbone conversion. The dated GPU checks below describe the validated scope;
they do not establish full-data execution or numerical score reproduction.
The inspected classification, detection and dense trainers explicitly enable
TF32 for CUDA float32 matrix multiplication and cuDNN operations. The revised
CLI now applies those switches only during its run and restores the caller's
previous settings on success or failure. On newer PyTorch it uses separate
operator precision settings; on older versions it preserves the full original
matmul mode, including `medium`. It does not mix the old and new APIs, as
required by [PyTorch's precision API](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

`result.json.runtime_precision` records the effective FP32 policy, inherited
determinism, cuDNN and attention-backend switches. Completed-epoch continuation
binds these settings and rejects changes before loading or publishing a
checkpoint. Unspecified source settings remain inherited and recorded; they
are not silently forced to new values. CPU execution leaves backend policy
unchanged. Backend-state tests on CPU validate scoping, restoration and
identity checks, not CUDA numerical equivalence or GPU reproducibility.

The implementation preserves RNG around diagnostic feature checks instead of
copying their historical random-stream side effects, so an identical historical
initialization trajectory is not claimed. The diagnostic temporarily disables
autocast's weight cache so its in-place adapter perturbation is actually observed,
then restores the caller's cache policy. It does not change the numerical effect
threshold, autocast dtype or training policy. CPU CLI checks exercise reduced local vision snapshots for ImageNet, ADE20K,
NYUv2 and COCO; video tensor execution and distributed accumulation are checked
separately from real decoding. Completed-epoch continuation is compared using
fresh CLI processes. The other four providers have bound-component checks,
not yet a complete released-model CLI validation matrix. Consult the PR's
executed-test evidence for the actual scope. Select the documented [short schedules](REVISED_SCHEDULES.md) explicitly.
Full-data score attribution and the tokenizer/generator distinction remain open.

### GPU validation checkpoint: 2026-11-19

The CUDA/BF16 diagnostic CLI matrix passed all 35 combinations of five reduced
providers and seven task/track routes, using real image/video/depth/bbox readers
with synthetic fixtures. Output contracts, trainable updates and unchanged
input checkpoints were checked. Geometry neighbors and agreement matched CPU
exactly for the non-tied test inputs, and the numerical policy was restored.
All three CUDA tests completed without skips; CPU skips are not this evidence.

On H200 with Python 3.12.13, Torch 2.5.1/cu124 and Transformers 5.16.1,
all five released-weight providers completed one ImageNet LoRA update and one
ADE20K DPT update using synthetic inputs and a reduced diagnostic head.
These ten component cases checked trainable updates, frozen-state preservation
and unchanged input checkpoint hashes. The RAEv2 case is the K7 tokenizer,
not the pending generator readout. These checks do not run the full datasets
or establish manuscript-score correspondence.

A reduced ImageNet CLI case also matched uninterrupted training exactly after
a completed-epoch stop/resume in fresh CUDA processes, including model,
optimizer, RNG, loss history and metrics. This covers single-GPU continuation
in the tested runtime, not distributed CUDA or cross-runtime continuation.
The local CPU and GPU dependency versions differ; their validation results
must not be treated as interchangeable environment certifications.

## Confirmed remaining work and boundaries

| Area | Observed difference and next action |
| --- | --- |
| LoRA task integration | The revised CLI connects all five inspected task profiles, with portable completed-epoch continuation. Full-data NYUv2 results appear below and the [ADE20K comparison](ADE20K_REPRODUCTION.md) covers all five models. ImageNet, SSv2 and COCO still require full-data released-weight validation; small component updates do not complete them. Earlier LoRA recipes remain historical. |
| DPT | The revised CLI connects ADE20K and NYUv2 to the frozen four-tap head and v2 schedule. NYUv2 full-data executions are reported below. The [ADE20K report](ADE20K_REPRODUCTION.md) compares all five released models over three seeds. Dense execution follows the inspected single-device source; distributed dense training is not claimed. |
| RAEv2 readout | The revised paper's LP/AP generator block 29 at time 0.1 is different from this package's existing tokenizer K7 readout. Keep these identities separate and port the matching generator path. |
| COCO | Basic5 uses Faster R-CNN bounding-box AP. The existing portable COCO runner already performs detection; historical category-presence scores must not be relabeled as detection AP. New correction metadata is not proof that all corresponding reruns completed. |
| NYUv2 | The LoRA/DPT run record stores median-aligned RMSE as primary. The author-supplied v2 descriptions select the unaligned field for manuscript reporting. Other tracks require their own provenance. |
| Geometry analysis | The [feature-array analysis CLI](REPRESENTATION_ANALYSIS.md) adds source-compared image/patch agreement and held-family task-direction fits. The original Figure 7b table statistics matched across five tasks. The reference-free correlations come from nested coordinate selection. Extraction identity requires separate validation. |
| SSv2 attentive probing | The adopted input and reader vary by model. See the [five source paths and original score audit](SSV2_EXECUTION.md#figure-3-attentive-probes). This does not block the separate revised LoRA route. |

The supplied manuscript also explicitly describes a category-presence analysis
separately from Basic5 detection. Preserve that distinction while reconciling
its intended scope; it must not be used to justify classification in the
Basic5 COCO runner. Source availability and component tests do not establish
the correspondence of any individual manuscript score to a run.

## Validation commands

The opt-in CUDA matrix is `tests/test_method_revised_cuda_matrix.py`. It exercises
five reduced synthetic providers across seven task/track CLI routes, geometry
CPU/CUDA agreement, and fresh-process ImageNet completed-epoch continuation.
It skips on CPU; such skips are not GPU evidence. On a CUDA host, execute
`OMP_NUM_THREADS=1 python -m unittest tests.test_method_revised_cuda_matrix` in the
complete downstream environment and inspect the exit status and skip count.
Even a successful run uses synthetic weights/data and does not verify released
checkpoints, full-data results, or distributed CUDA training.

Run the component and COCO regression checks with an installed downstream
environment:

`OMP_NUM_THREADS=1 .venvs/downstream/bin/python -m unittest tests.test_attention_lora tests.test_cvpr_adaptation_runtime tests.test_dpt_probe tests.test_downstream_coco tests.test_basic5_seg_detection`

Run deliberate mutations in an isolated source copy:

`OMP_NUM_THREADS=1 python3 bin/mutate.py --spec mutations/attention-lora.json --python "$PWD/.venvs/downstream/bin/python"`

`OMP_NUM_THREADS=1 python3 bin/mutate.py --spec mutations/cvpr-adaptation-runtime.json --python "$PWD/.venvs/downstream/bin/python"`

The downstream CI job explicitly executes the new component tests. Whole-suite
push validation remains required; focused checks do not replace it.

Run the bound-model contracts in the same downstream environment:

`OMP_NUM_THREADS=1 .venvs/downstream/bin/python -m unittest tests.test_cvpr_bound_runtime tests.test_cvpr_hf_bindings tests.test_method_cvpr_native_bindings tests.test_cvpr_rae_bindings`

The `cvpr-bound-runtime`, `cvpr-hf-bindings`, `cvpr-native-bindings` and
`cvpr-rae-bindings` mutation specifications cover the new behavior. Run each
with `bin/mutate.py --spec mutations/<name>.json --python <downstream-python>`.

## Author-supplied November v2 run descriptions

The LoRA and DPT v2 Markdown and JSON descriptions were written after the
experiments from their code and records. The runs consumed earlier task and
optimizer specifications. Use the v2 descriptions to interpret recorded
protocol IDs. The descriptions cover six backbones including Cosmos 3 Super.
The five-provider validation matrix above covers a subset of that population.

DINOv3-7B LoRA uses 50 epochs on ImageNet and 24 on SSv2. Each has three
original seed records. The COCO bbox LoRA records retain the v1 identifier
with the same detection recipe. The two-epoch resume diagnostic is excluded.
For the two revised tracks, report the final scheduled epoch and the mean and
sample standard deviation over seeds 0, 1 and 2. The source segmented jobs
reseeded random draws at restart. Portable exact continuation preserves RNG.

For NYUv2 LoRA and DPT, the recorded `rmse` is per-image median-aligned.
The manuscript reports `rmse_unaligned`. In the three original VGGT Omega
DPT records the means are 0.408087 and 0.424844 respectively. The latter
rounds to the manuscript's 0.425. This finding applies to these tracks.
The review copies of the [LoRA](current_protocols/BASIC5_LORA_v2.md) and [DPT](current_protocols/BASIC5_DPT_v2.md) protocols preserve the experimental settings. Private source locations are replaced with a placeholder.

Across the six models, seven task/track routes and three seeds, all 126
selected result records matched their completion markers and reached the
specified final epoch. Independent field-level checks of both evaluation
count and total updates were possible for 72 records. The ADE20K metrics
in the other 36 records expose evaluated pixels but no image count, and the
18 COCO recipes omit a total-step field. The DINOv3-7B short-run recipes
also record zero activation-checkpointed blocks on ImageNet and 40 on SSv2,
qualifying the protocol's general activation-checkpointing description.

Use the [transfer comparison guide](TRANSFER_COMPARISON.md) to calculate
LP/AP pair reversals and radial values from an explicitly selected score table.

A two-GPU NCCL check of the preceding implementation ran the ImageNet, SSv2
and COCO LoRA CLI on reduced inputs. Each completed uninterrupted training,
an epoch-boundary stop, and a fresh-process restart. ImageNet and SSv2 matched
across model, optimizer, RNG, loss and metrics. COCO matched evaluation
metrics, reported losses and update count, but the checkpoint tensor
comparison failed. Exact two-GPU detection restart remains unverified.

An additional two-GPU COCO control restored the same checkpoint with zero
updates and reproduced all saved tensors, metrics and losses exactly. Two
independent uninterrupted runs also produced identical metrics and losses
but different trained tensors (maximum absolute trainable-weight difference
3.73e-9). The strict comparison detected both differences. The particular
operation responsible for the training variation has not been identified.

For the preceding LP/AP/FT tracks see the [three v1 protocols and execution comparison](BASIC5_V1_AUDIT.md).

## Figure 3 LoRA and DPT seed evidence (2026-11-20)

Across the five displayed models and seven task/readout columns, the audit
read 105 original final results and their matching completion records. In
31 of 35 cells the three seeds shared a code digest, checkpoint identity and
recipe, and both the mean and sample SD reproduced the displayed values at
their printed precision. This includes all ten NYUv2 cells, which use
unaligned RMSE, and both explicitly shortened classification schedules.
The four exceptions were VGGT-Omega COCO LoRA and RAEv2 ImageNet-1K, COCO
and SSv2 LoRA. Each had two code digests across its three seeds. Their arithmetic also
matches at printed precision (35/35 overall). The affected classification
and detection entry points changed progress logging and wall-time stop
handling. An independent run across the two versions would be needed to
measure any effect on results. The report exposes this arithmetic separately
and retains the failed code-identity check. This
audit reads historical artifacts. Full-data retraining with the public
implementation is a separate validation step.

Run `python -m downstream.revised_reporting --config plan.json --out report.json`
with a new output path. For this audit supply a private plan with
`schema: native_figure3_v1` and a list named `cells`. Each cell specifies
`method_id`, `dataset`, `track`, `protocol_id`, `paper_mean`, `paper_sd`
(the two values as printed decimal strings), and `runs`. For each seed
0, 1 and 2 include the integer `seed` and two descriptors, `result` and
`completion`, each containing the file's `path` and full `sha256`.
Select runs by experiment identity and schedule before comparing scores.
COCO detection's native LoRA record uses the v1 identifier. ADE20K records
an evaluated pixel count. The audit checks that it is positive; image-level
membership cannot be reconstructed from that field. Output reports each
cell as match, mismatch, incomplete or invalid. Inspect these statuses even
when the command successfully writes a report.

## Auditing portable Figure 3 training outputs

For comparison with all 35 printed LoRA/DPT entries, use the
[Figure 3 panel reporting mode](LORA_FULL_REPRODUCTION.md#compare-public-outputs-with-figure-3).
For a single artifact audit, create a private plan for each model,
task and track with `schema: portable_figure3_v1`, `expected_seeds: [0, 1, 2]`
and a `runs` list. Each entry supplies the integer seed, config path, output
directory (`out`) and recorded process exit status (`exit_status`). Run
`python -m downstream.revised_reporting --config plan.json --out new-report.json`.
This mode reads existing public-CLI artifacts for all five LoRA routes and
both dense DPT routes. It also resolves the two provider-owned short schedules.
Use the training environment for named schedules.

The report checks the hash-checked output contract, resolved recipe,
final epoch, recorded train/validation populations, evaluated sample count,
checkpoint digest, final trainable artifact, per-epoch update counts and execution topology. The configuration apart from seed
must be identical across runs. Counts alone cannot establish sample identity.
ImageNet and SSv2 require the published top-5 accuracy to match the evaluator
in percentage points, between top-1 and 100. COCO checks the nonnegative
integer detection count and all six bbox AP values (overall, at IoU .50 and
.75, and for small, medium and large objects). The evaluator reports percentage
points and the published contract stores fractions. Empty predictions yield
six zeros.
Depth uses unaligned RMSE in meters, with the aligned value retained separately.
Three valid seeds are required for a complete cell. Partial summaries describe
only the valid seeds listed. No historical result or equivalence to a published
score is automatically certified. The native historical audit above uses a
different input schema.

### Full-data input checks for the remaining LoRA routes

The [full-data LoRA comparison targets](LORA_FULL_REPRODUCTION.md) give the
printed reference statistics and the original configuration provenance.

The public loader ran on both splits using the same dataset roots as the
revised training source. It checked class mappings and split sizes and
decoded the first item of each split.

| LoRA task | Training | Validation | Classes | Membership check |
| --- | ---: | ---: | ---: | --- |
| ImageNet-1k | 1,281,167 | 50,000 | 1,000 | ImageFolder enumerated the two class trees |
| SSv2 | 168,913 | 24,777 | 174 | Every listed clip resolved to a file |
| COCO detection | 118,287 | 5,000 | 80 | Every annotated image filename was present in its split |

The input check decoded six samples in total. Full-data training for these
three LoRA routes is underway; see the dated panel audit for completed seeds.

## Full-data NYUv2 reruns (2026-11-21)

The portable CLI ran the two revised tracks for the five Figure 3 models,
using released weights, the full official split (795 train / 654 test),
224-pixel input, batch 8, six data-loader workers, 30 epochs and seeds 0, 1 and 2 on individual H200
GPUs. Each of the 30 outputs completed 2,970 optimizer updates. Each result,
trainable checkpoint and metric contract passed the portable audit. The
comparison below reports unaligned RMSE in meters as mean +/- sample SD at
the figure's printed precision. The historical column was independently
checked against native seed records. The rerun uses the public training code
at commit `f7dfe83`; subsequent reporting checks read those outputs.

| Backbone | Portable DPT | Figure 3 DPT | Portable LoRA | Figure 3 LoRA |
| --- | ---: | ---: | ---: | ---: |
| DINOv3-7B | 0.401 +/- 0.002 | 0.401 +/- 0.001 | 0.396 +/- 0.003 | 0.396 +/- 0.002 |
| VGGT-Omega | 0.427 +/- 0.002 | 0.425 +/- 0.002 | 0.419 +/- 0.002 | 0.418 +/- 0.003 |
| RAEv2 K7 | 0.437 +/- 0.003 | 0.436 +/- 0.002 | 0.472 +/- 0.002 | 0.472 +/- 0.002 |
| V-JEPA 2.1-G | 0.454 +/- 0.001 | 0.454 +/- 0.001 | 0.463 +/- 0.006 | 0.461 +/- 0.003 |
| SigLIP 2-G | 0.481 +/- 0.003 | 0.480 +/- 0.001 | 0.480 +/- 0.005 | 0.479 +/- 0.005 |

At the printed precision, four means match. Two cells also match the sample SD:
RAEv2 LoRA and V-JEPA 2.1-G DPT. Numerical differences
at the printed precision remain visible here. Neither the source of such
differences nor equivalence of the implementations follows from these aggregate
numbers. The existing holds for other tasks and figures still apply.

### VGGT-Omega backbone precision

For one fixed NYUv2 validation image on H200, we compared the four DPT
feature tensors against the original trainer with the released aggregator.
The initial head parameters and depth target matched. Converting the backbone
to bfloat16 before autocast produced layer-wise maximum absolute differences
of 0.199, 0.548, 38.567 and 321.567. Preserving the loaded float32 backbone
produced zero difference at every tap, with either the native image or the
portable preprocessing. This verifies the storage correction on that input.
The table above records the earlier implementation.

### DINOv3 rotary frequencies

The same H200 comparison with released DINOv3-7B weights isolated another
precision difference. Loading the vision model with bfloat16 weights and
keeping the rotary-frequency buffer in float32 reproduced all four native
DPT taps exactly. Converting the entire model to bfloat16 after loading
rounded that buffer and produced maximum differences of 0.625, 4, 36 and
96. This loader policy applies to revised CUDA training only.

For the same NYUv2 input, the other three providers (RAEv2 K7, SigLIP 2-G
and V-JEPA 2.1-G) already matched the original four taps and initial DPT
prediction exactly. End-of-training parity remains a separate measurement.

### Full-data reruns after the two precision corrections

Each cell below includes three independent 30-epoch runs (seeds 0, 1 and 2)
with the same populations and schedules as the earlier NYUv2 table. All 12
training processes exited successfully. The final auditor verified each
checkpoint, epoch history, evaluation count and metric contract. For VGGT,
the first post-training reporting process used incompatible revisions of the
reporter and recipe and failed. The complete reporter subsequently read all
six retained training outputs.

| Backbone | Updated DPT | Figure 3 DPT | Updated LoRA | Figure 3 LoRA |
| --- | ---: | ---: | ---: | ---: |
| DINOv3-7B | 0.401 +/- 0.001 | 0.401 +/- 0.001 | 0.396 +/- 0.002 | 0.396 +/- 0.002 |
| VGGT-Omega | 0.428 +/- 0.002 | 0.425 +/- 0.002 | 0.417 +/- 0.003 | 0.418 +/- 0.003 |

All values are unaligned RMSE in meters (mean +/- sample SD). The two
changes were selected from original loader behavior and fixed-input tests;
these aggregate reruns assess their effect independently.
