# Frozen evaluation of native DINO teacher exports

This connects the existing design/scaling teacher implementation to the five
BasicFive LINEAR task components. It is a portable integration, not evidence of
reproduced paper scores. See [submission scope](SUBMISSION_SCOPE.md).

Use `kind: dinov3_ablation` for a local teacher `encoder.pt` produced by the
DINO training adapter. Choose `arch: vit_base_patch16` or `vit_large_patch16`,
with `img_size: 224` and `patch_size: 16`. The architecture uses four register
tokens, axial RoPE with base 100 and coordinate rescaling 2, and the existing
native model defaults. Dimension overrides and incompatible state keys fail.
Supply exports from that configuration; bare state dictionaries do not authenticate
historical training configuration, epoch or paper-row identity. Native full
training checkpoints are not accepted by this loader. It does not select a
checkpoint, recover training state, or convert a released HF model.

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

This does not verify full-size checkpoint loading, GPU/BF16 behavior, native
DDP, complete recipes, historical continuation or final scores. The existing
shared depth-loss and schedule attribution conflicts remain unresolved; this
integration does not silently select an original run's version. Portable results
remain noncanonical; fixture results cannot be recorded as paper measurements.
Private source copies and run evidence remain outside Git. Record final local,
mutation, Linux and CI results separately in the PR.
