# Local vision providers for Basic5 components

Status: 2026-09-24. This expands model integration, not the canonical experiment
or score-reproduction claim. Use the [downstream environment](DOWNSTREAM.md)
with its hashed lock; method-adapter environments remain separate.

| Provider | Released architecture | Classification readout | Spatial features |
| --- | --- | --- | --- |
| `clip_hf` | CLIP ViT-L/14 at native 336, 24 blocks | projected CLS, 768 channels | 1024 channels, CLS removed |
| `siglip2_g` | SigLIP2-G/16 at native 384, 40 blocks | official MAP pooler, 1536 channels | 1536 channels, no special token removal |
| `dinov3_hf` | DINOv3 ViT-7B/16, 40 blocks | L2-normalized CLS, 4096 channels | 4096 channels, CLS and four registers removed |

All three support ImageNet frozen LP and frozen/FT components for ADE20K,
NYUv2 and SSv2. DINOv3 additionally supports the existing COCO pyramid for
frozen/FT components. These are **23 task/model/adaptation execution paths**,
tested with reduced local models and synthetic data. They do not establish
complete family-specific recipes, released-weight scores or GPU feasibility.

The downstream CI environment runs all 23 integration paths with its complete
dependencies. Method-specific environments still run available model tests, but
skip this task-integration test explicitly when downstream dependencies are
absent. A subprocess regression removes SciPy and verifies that only integration
is skipped while a provider capability test continues to run.

The following are alternative `backbone` objects for the
[complete ImageNet example](DOWNSTREAM.md#extended-training-components-2026-09-22).
Replace the placeholder directory with a complete local HF snapshot containing
`config.json` and its weights. `img_size` describes checkpoint-native geometry;
the task's `probe.image_size` remains the task input size. For DINOv3, use the
image size recorded in the local configuration if it differs from this example.

```json
[
  {"kind": "clip_hf", "arch": "released", "encoder": "/path/to/local-snapshot", "img_size": 336, "patch_size": 14},
  {"kind": "siglip2_g", "arch": "released", "encoder": "/path/to/local-snapshot", "img_size": 384, "patch_size": 16},
  {"kind": "dinov3_hf", "arch": "released", "encoder": "/path/to/local-snapshot", "img_size": 224, "patch_size": 16}
]
```

The loader constructs only the vision tower. It accepts vision-only and full
multimodal snapshots, checks missing/unexpected vision keys and refuses a
different released architecture. Full CLIP snapshots carry projection size in
the outer configuration. No network fetch or random fallback is allowed.
`arch: fixture` explicitly permits reduced saved models for tests; it is not a
released-weight experiment. Checkpoint hash acquisition/verification remains
the caller's responsibility; shape compatibility does not authenticate weights.

The task supplies ImageNet-normalized pixels. The provider converts these once
to CLIP or SigLIP normalization; DINOv3 keeps them and pads to its patch grid.
The global classifier never substitutes spatial patch means for CLS/MAP.
CLIP and SigLIP LP normalize image features, but their video and FT readouts
do not. DINOv3 normalizes each frame before temporal averaging. CLIP/SigLIP
linear heads use normal initialization with standard deviation .01; DINOv3
uses zeros. FT groups follow the family: last-block endpoint for CLIP/SigLIP,
one additional head endpoint for DINOv3. Existing providers keep their behavior.

Select `capture_basic5_components` with `adaptation: frozen` and
`optimizer_profile: basic5_frozen_v1`, or `adaptation: finetune` and
`optimizer_profile: basic5_finetune_v1` on the four downstream tasks where
supported. Existing schedule/batch constraints apply. Results always report
`canonical_eligible: false` and `record_value: false`.

## Explicit limits

- AP is rejected for these three providers: query-reader and dense-adapter
  definitions differ across the source evidence. No common reader is silently
  substituted. Existing providers' supported AP paths remain available.
- ImageNet FT execution remains rejected pending augmentation reconciliation.
  The differentiable classifier composition alone does not complete its recipe.
- CLIP COCO is rejected because its patch-14 grid does not satisfy the shared
  stride-16 pyramid. SigLIP COCO is rejected because normalization relative to
  detection padding has not been reconciled with the shared transform.
- No distributed/FSDP, accumulation, new GPU run, released 7B-weight execution,
  full dataset score, seed aggregate, Extend or additional Step-4 implementation
  is certified by this change. A CPU fixture score is never a workbook value.
- Existing method adapters and feature-extraction artifacts are unchanged.

Validation includes actual CLI artifact-contract checks, local snapshot loading,
negative loading/recipe tests and existing-provider regressions. Private
comparisons executed unchanged captured wrappers with identical small FP32
models injected at the loader boundary: pooled/spatial outputs, input gradients
and all parameters matched through three SGD updates for each family. All
129 encoder parameter assignments in those fixtures matched captured layer and
decay policies. This verifies the component boundary, not the reference loader
on released checkpoints. Source revisions, file hashes and raw evidence stay
outside Git. Documentation examples are parsed through configuration validation.
