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
| `cradiov4_h` | The native timm ViT checkpointing API, including CPE block execution |
| `vggt_omega` | Non-reentrant checkpointing of the aggregator while training |

The shared trainable builder applies these policies to ImageNet, SSv2, ADE20K,
NYUv2 and COCO components. Existing task/provider restrictions still apply.
A boolean is required. Frozen and attentive adaptation reject an enabled option,
as do providers without an inspected policy. Never infer recomputation support
from trainability alone.

## Source comparison and intentional differences

The original BasicFive FT constructors enable the vision checkpointing API for
CLIP/SigLIP2/Cosmos3; the DINO helper enables the first available native API.
The K7 helper wraps individual blocks with `use_reentrant=False`; the video
helper enables the native encoder flag. Current original source and captured
snapshots were inspected read-only on 2026-10-09. Source identities and actual
run configuration evidence stay outside Git. Available V-JEPA2.1 dense resolved
configurations were re-read; their existence does not establish completed runs.
Other families' historical run attribution remains unverified.

C-RADIO's inspected constructor enables its inner ViT checkpointing API.
Omega checkpoints the entire aggregator only in training mode. These two
policies also cover video and detection feature routes. Six original FT
result/metadata records were re-read across the two families; they establish
configuration provenance, not a complete mapping to paper results. The
portable Omega wrapper owns its flag and checkpoint call, so copying a model
does not retain a closure referencing another model's parameters.

The portable setting is explicit rather than automatically enabled for every
FT run. Transformers uses an explicit non-reentrant backend, allowing ordinary
image inputs without input gradients and keyword arguments. K7 preserves block
registration, state keys and default RNG preservation; repeated enablement does
not nest wrappers. Errors are propagated rather than silently swallowing an
unsupported backend. This is component support, not native FSDP integration.

## Verification and limits

Method locks can contain encoder dependencies without NYUv2's `h5py`. The
NYUv2 configuration contract is tested separately and skips explicitly only
when `h5py` is absent. Encoder recomputation and ADE20K/COCO/SSv2 configuration
checks remain active. Fresh-process regressions verify both this partial
environment and complete execution without skips; broken dependency imports
are propagated. The downstream CI job executes the full module.

`tests.test_method_activation_checkpointing` verifies actual backward recomputation on
six small real encoder implementations, equal outputs, parameter gradients,
optimizer updates, state keys and evaluation outputs. A private comparison also
executes the inspected K7/DINO/video helpers and the source-style native API
calls on the same small models; all six output/gradient comparisons pass in
the installed CPU environment. This does not recreate the original runtime. A stochastic block checks
RNG preservation and gradients. Configuration tests reject invalid selection and
verify dense/video validators. Fault injection is specified in
`mutations/activation-checkpointing.json`.

`tests.test_method_remaining_checkpointing` adds C-RADIO and Omega image,
video, detection, image-summary and video-summary routes, checking real
recomputation, outputs, gradients and optimizer updates. Omega checks also
cover evaluation bypass and independent gradients after model copying.
A private comparison executes the inspected experimental wrappers with the
same reduced models and two SGD updates per route. All ten comparisons pass;
the maximum recorded output difference is 4.77e-7. C-RADIO uses the actual
official custom CPE code in this comparison; all 27 retrieved official files
match the previously inspected copy. Public tests use native timm ViT blocks
and the pinned Omega aggregator without redistributing private source.

Released weights, GPU peak-memory reduction, BF16, distributed recomputation,
full-data execution and paper scores have not been verified by these CPU tests.
Do not equate these components with completed experiments. See the
[coverage ledger](FINAL_PAPER_COVERAGE.md) and [dense execution](DENSE_EXECUTION.md).
