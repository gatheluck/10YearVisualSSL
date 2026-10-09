# BasicFive activation checkpointing

See [submission scope](SUBMISSION_SCOPE.md) for original experiments versus
portable component verification.

Fine-tuning can explicitly trade additional backward computation for fewer saved
intermediate activations. Set `"activation_checkpointing": true` inside the
existing `backbone` mapping, with `adaptation: finetune` and the existing component
profile. Omission or `false` preserves previous execution. This option is
independent of checkpoint files, portable resume, accumulation and precision.

| Provider | Inspected recomputation mechanism |
| --- | --- |
| `clip_hf`, `siglip2_g`, `dinov3_hf`, `cosmos3_super_vm` | The vision model's native Transformers checkpointing API |
| `raev2_k7` | Non-reentrant checkpointing around each existing encoder block |
| `vjepa2_1` | The encoder's native activation-checkpointing flag |

The shared trainable builder applies these policies to ImageNet, SSv2, ADE20K,
NYUv2 and COCO components. Existing task/provider restrictions still apply.
A boolean is required. Frozen and attentive adaptation reject an enabled option,
as do providers without an inspected policy. C-RADIO and Omega are not enabled
by this change. Never infer recomputation support from trainability alone.

## Source comparison and intentional differences

The original BasicFive FT constructors enable the vision checkpointing API for
CLIP/SigLIP2/Cosmos3; the DINO helper enables the first available native API.
The K7 helper wraps individual blocks with `use_reentrant=False`; the video
helper enables the native encoder flag. Current original source and captured
snapshots were inspected read-only on 2026-10-09. Source identities and actual
run configuration evidence stay outside Git. Available V-JEPA2.1 dense resolved
configurations were re-read; their existence does not establish completed runs.
Other families' historical run attribution remains unverified.

The portable setting is explicit rather than automatically enabled for every
FT run. Transformers uses an explicit non-reentrant backend, allowing ordinary
image inputs without input gradients and keyword arguments. K7 preserves block
registration, state keys and default RNG preservation; repeated enablement does
not nest wrappers. Errors are propagated rather than silently swallowing an
unsupported backend. This is component support, not native FSDP integration.

## Verification and limits

`tests.test_method_activation_checkpointing` verifies actual backward recomputation on
six small real encoder implementations, equal outputs, parameter gradients,
optimizer updates, state keys and evaluation outputs. A private comparison also
executes the inspected K7/DINO/video helpers and the source-style native API
calls on the same small models; all six output/gradient comparisons pass in
the installed CPU environment. This does not recreate the original runtime. A stochastic block checks
RNG preservation and gradients. Configuration tests reject invalid selection and
verify dense/video validators. Fault injection is specified in
`mutations/activation-checkpointing.json`.

Released weights, GPU peak-memory reduction, BF16, distributed recomputation,
full-data execution and paper scores have not been verified by these CPU tests.
Do not equate these components with completed experiments. See the
[coverage ledger](FINAL_PAPER_COVERAGE.md) and [dense execution](DENSE_EXECUTION.md).
