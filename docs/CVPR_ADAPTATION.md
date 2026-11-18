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
components to three explicitly versioned task routes. Older task defaults are
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

The task CLI below adds inspected image/semantic readers and training schedules.
Released-weight execution, CUDA and continuation still need further validation.
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

The portable `imagenet` task name corresponds to `imagenet1k` in the inspected
source protocol. All routes use AdamW `(0.9, 0.999)`, weight decay `0.05` and
global trainable-gradient clipping at `1.0`. Bias/one-dimensional parameters
and names containing `norm` have no decay. LoRA rank and alpha are both `16`;
DPT uses hidden width `512`. The head and adapters retain float32 parameters.
Each parameter group has its own absolute linear warmup from `1e-6` followed
by cosine decay to `1e-6`. Scheduling happens before each zero-based optimizer
update; the floor is reached at the reference horizon, not one update early.
Diagnostic truncation does not shorten that reference schedule.

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

Without a `diagnostic` object, the entry point requires `arch: released` and
the reference populations: ImageNet 1,281,167 training / 50,000 validation
images with 1,000 classes; ADE20K 20,210 / 2,000. These checks are necessary
input checks, not proof of a paper result. For a reduced test, add an explicit
`diagnostic` object, for example
`"diagnostic": {"epochs": 1, "batch_size": 2, "max_steps": 3}`.
Optional diagnostic fields also include `max_train_samples`, `max_val_samples`
and `dpt_hidden`. Diagnostic batch changes linearly scale base learning rates
while preserving absolute warmup/floor rates and reference-batch metadata.
Top-level epoch/rate overrides are rejected rather than silently changing the
reference recipe. The output directory must not already exist.

ImageNet can be launched with
`torchrun --master-addr=127.0.0.1 --master-port=29517 --nproc-per-node=2 -m downstream.revised_tasks --config config.json --out runs/new-distributed-run`.
This is a **single-host** example; ensure port 29517 is unused or select another
unused port. The tested local launcher needs explicit IPv4 loopback: automatic
`--standalone` rendezvous stalled in that environment. Multi-node launch needs
administrator-approved reachable addresses and network configuration, not a
loopback address shared across hosts.
Its effective batch must divide evenly across ranks; training drops incomplete
batches, and validation partitions samples without duplication. ADE20K follows
the inspected single-device trainer and rejects multi-rank execution. These
interfaces are not evidence of released-checkpoint or multi-GPU validation.

Successful execution writes `result.json`, contract metrics/manifest and
`trainable.pt` containing only trainable parameters. **`trainable.pt` is not a
resume checkpoint**: optimizer, data-loader and random-stream continuation are
not implemented by this entry point. `result.json` retains `record_value: false`
and `historical_run_verified: false` for every run. `checkpoint_sha256` binds
the local checkpoint input bytes through a digest without retaining their
filename or path; the digest identifies an input, not a verified correspondence
to a paper result.
`status: diagnostic` marks
an explicit diagnostic configuration; `status: component_complete` records a
completed non-diagnostic component run, not a certified paper score. Inspect
`training_complete`, populations, update counts, precision and recipe fields
as well as the manifest; an `ok` manifest alone is insufficient for reproduction.

CPU execution uses float32. CUDA selects bfloat16 backbone/autocast with
float32 adapters/head, but this task path has not yet been validated on CUDA.
The implementation preserves RNG around diagnostic feature checks instead of
copying their historical random-stream side effects, so an identical historical
initialization trajectory is not claimed. Initial CLI tests exercise a reduced
local vision snapshot; the other four providers have bound-component checks,
not yet a complete released-model CLI validation matrix. Consult the PR's
executed-test evidence for the actual scope. Score attribution, shortened
model-specific schedules and the tokenizer/generator distinction remain pending.

## Confirmed remaining work and boundaries

| Area | Observed difference and next action |
| --- | --- |
| LoRA task integration | The revised CLI connects ImageNet and ADE20K with rank 16, alpha 16, an LP head and separate head/adapter rates. SSv2, COCO and NYUv2 revised routes, continuation, all-provider CLI verification and released-weight/GPU validation remain. Earlier LoRA recipes remain historical. |
| DPT | The revised CLI connects ADE20K to the frozen four-tap head and v2 schedule. NYUv2 task integration and released-weight/GPU validation remain. Dense execution follows the inspected single-device source; distributed dense training is not claimed. |
| RAEv2 readout | The revised paper's LP/AP generator block 29 at time 0.1 is different from this package's existing tokenizer K7 readout. Keep these identities separate and port the matching generator path. |
| COCO | Basic5 uses Faster R-CNN bounding-box AP. The existing portable COCO runner already performs detection; historical category-presence scores must not be relabeled as detection AP. New correction metadata is not proof that all corresponding reruns completed. |
| NYUv2 | The inspected revised source records median-aligned primary depth metrics and unaligned diagnostics. Cross-family alignment correspondence still needs reconciliation; do not silently choose a metric. |
| Geometry analysis | The revised manuscript adds image/patch nearest-neighbor agreement. Its full source/configuration correspondence has not been validated by this checkpoint. |

The supplied manuscript also explicitly describes a category-presence analysis
separately from Basic5 detection. Preserve that distinction while reconciling
its intended scope; it must not be used to justify classification in the
Basic5 COCO runner. Source availability and component tests do not establish
the correspondence of any individual manuscript score to a run.

## Validation commands

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
