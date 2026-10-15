# Frozen evaluation of native DINO teacher exports

This connects the existing design/scaling teacher implementation to the five
BasicFive LINEAR task components. It is a portable integration, not evidence of
reproduced paper scores. See [submission scope](SUBMISSION_SCOPE.md).

Use `kind: dinov3_ablation` for a local teacher `encoder.pt` produced by the
DINO training adapter or the native importer below. Choose `arch: vit_base_patch16` or `vit_large_patch16`,
with `img_size: 224` and `patch_size: 16`. The architecture uses four register
tokens, axial RoPE with base 100 and coordinate rescaling 2, and the existing
native model defaults. Dimension overrides and incompatible state keys fail.
Supply exports from that configuration; bare state dictionaries do not authenticate
historical training configuration, epoch or paper-row identity. Native full
training checkpoints are not accepted by this loader. It does not select a
checkpoint, recover training state, or convert a released HF model.

## Importing a native teacher checkpoint

From the repository root with the downstream environment active, run
`python -m methods.31_dinov3.downstream_backbone_ablation --source native.pth --sha256 VERIFIED_SHA256 --arch vit_base_patch16 --epoch 300 --out new-teacher-export`.
Replace the hash with independently verified source provenance; use
`vit_large_patch16` only for a matching native large model. `--epoch` matches
the literal saved `epoch` field, not a number inferred from the filename.
The importer does not select a paper row or infer completed training from it.

The importer requires `cfg.model.arch` to agree with the explicit architecture.
Declared geometry and RoPE settings must match this evaluation preset; omitted
optional fields use the captured loader's defaults. Missing configuration or
conflicting settings are refused rather than overridden. Restricted
`weights_only=True` deserialization has no unrestricted pickle fallback.
Only `teacher_state_dict` entries under `backbone.` are exported. The captured
loader's nine known head namespaces are excluded; unknown namespaces, missing
or extra encoder keys, wrong shapes/dtypes and nonfinite weights are refused.
The source hash is checked before and after conversion. Existing output
directories are never reused.

Use `new-teacher-export/encoder.pt` in the downstream configuration. The adjacent
`export.json` records hashes, architecture, literal epoch and validation scope.
Neither file carries source paths, full configuration, experiment identities,
student weights, heads, optimizer or RNG state. This is an evaluation export,
not training continuation or proof that a checkpoint produced a paper score.
The bare encoder format remains compatible with existing adapter exports;
its sidecar is provenance for the conversion, not a signed score certificate.

Copy the [ImageNet example](examples/dino_ablation_linear.json), set local input
paths, then run `python -m downstream.imagenet --config resolved.json --out new-output`.
Use the downstream environment and its platform lock. The example is a recipe,
not a record of an executed experiment. The existing trained teacher adapter's
export format remains unchanged.

| Task | Readout and required configuration |
| --- | --- |
| ImageNet | Final global CLS, L2 normalization; `preprocessing_profile: captured_provider_v1` |
| SSv2 | Mean final patch features per frame, then temporal mean; `video_recipe: captured_provider_video_v1` |
| ADE20K / NYUv2 | Final register-stripped patches on the actual rectangular grid |
| COCO | Same grid with native trainable transposed pyramid; `detector_profile: captured_native_detection_v1` |

All five require `profile: capture_basic5_components` and `adaptation: frozen`.
The task pipeline normalizes inputs once with ImageNet statistics; the provider
receives already-normalized tensors. Detection owns its own normalization.
Image/video heads use normal initialization with standard deviation 0.01 and
zero bias. The teacher remains frozen and in evaluation mode. AP and FT are
refused. Image/video execution uses the existing single-process accumulation and
portable continuation components; distributed execution is refused. Use the
[task examples](DOWNSTREAM.md) for the remaining schemas.

## Evidence and limits

On 2026-10-14, all 19 inspected evaluation/model source files and both inspected manifest/protocol JSON files matched between
Capture and the current original workspace. The native model implementation
also matches the portable model. Reduced CPU comparisons against the captured
wrapper match image, spatial and video outputs exactly, including a rectangular
grid. Tests compare head outputs, gradients and two SGD updates and exercise all
five real task entrypoints with tiny exports. Meta-device construction checks
both full-size preset geometries without allocating or executing their weights.

The native importer additionally matches the captured backbone-key selection
exactly on a reduced teacher fixture, including all nine head namespaces.
Four refreshed loader/trainer/manifest/configuration files match the current
original workspace. Read-only archive-metadata inspection confirms epoch 300
and the supported configuration for one actual base-model checkpoint. It did
not deserialize full weights or execute that checkpoint during the initial audit.
On October 15, an actual base-model epoch-300 teacher passed strict full-weight
conversion (190 backbone tensors) and H200 readout comparisons against the
original backbone. Image CLS normalization, spatial patch grids and video
patch/time means matched exactly for 224-square and 224-by-320 inputs, in FP32
and BF16 (12 comparisons). The source hash was unchanged after the experiment.
Both presets have reduced conversion/readout tests; no actual large archive
was verified here.

This does not verify large-model full-weight loading, downstream head training,
native DDP, complete recipes, historical continuation or final scores. The existing
shared depth-loss and schedule attribution conflicts remain unresolved; this
integration does not silently select an original run's version. Portable results
remain noncanonical; fixture results cannot be recorded as paper measurements.
Private source copies and run evidence remain outside Git. Record final local,
mutation, Linux and CI results separately in the PR.
