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

**These components are not yet wired into the canonical task CLI.** The explicit
model bindings below expose block lists and gradient paths, but do not establish
complete task recipes. Selecting task-specific heads, parameter-group learning
rates and continuation remains integration work.

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

Canonical data readers, training CLI/configuration, distributed execution and
resume integration still need task-level validation. The model-specific tap
bindings below provide component access, not complete task execution.
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

## Confirmed remaining work and boundaries

| Area | Observed difference and next action |
| --- | --- |
| LoRA task integration | The inspected revised source uses rank 16, alpha 16 and an LP head, with separate head/adapter learning rates. Explicit model block/gradient bindings are available; task recipes, CLI routes and released-weight validation remain. Earlier LoRA recipes remain historical. |
| DPT | The four-tap head, model-specific feature bindings and frozen-backbone/head composition are available as components. Full task/CLI integration, scheduling and CUDA/distributed runs remain unvalidated. |
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
