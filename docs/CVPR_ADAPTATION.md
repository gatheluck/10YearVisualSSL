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

**This component is not yet wired into the canonical task CLI.** It does not
claim that all five models or their task recipes now support LoRA training.
In particular, selecting task-specific heads, parameter-group learning rates,
model block lists, gradient paths and continuation remains integration work.

## Confirmed remaining work and boundaries

| Area | Observed difference and next action |
| --- | --- |
| LoRA task integration | The inspected revised source uses rank 16, alpha 16 and an LP head, with separate head/adapter learning rates. Bind actual model blocks and task recipes, then compare forwards, gradients and updates before enabling a CLI route. Earlier LoRA recipes remain historical. |
| DPT | A revised four-tap dense head is present in the experimental source. Its source notes a third-party derivation; establish pinned provenance and redistribution terms before porting. Do not substitute an arbitrary dense head. |
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

`OMP_NUM_THREADS=1 .venvs/downstream/bin/python -m unittest tests.test_attention_lora tests.test_downstream_coco tests.test_basic5_seg_detection`

Run deliberate mutations in an isolated source copy:

`OMP_NUM_THREADS=1 python3 bin/mutate.py --spec mutations/attention-lora.json --python "$PWD/.venvs/downstream/bin/python"`

The downstream CI job explicitly executes the new component tests. Whole-suite
push validation remains required; focused checks do not replace it.
