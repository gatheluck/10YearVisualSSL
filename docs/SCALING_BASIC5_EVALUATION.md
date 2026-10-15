# Frozen evaluation of three scaling families

For manuscript labels see the [terminology guide](PAPER_TERMINOLOGY.md).
Support claims refer to the portable package; see [scope and terminology](SUBMISSION_SCOPE.md).

The submitted manuscript's Appendix C.4 Tables 20–22 use colorization,
rotation and split-brain ViT-B/16 and ViT-L/16 checkpoints. These dedicated
providers connect their source-specific representations to the five portable
BasicFive task components. Table 20 itself reports only ImageNet, COCO and
ADE20K; the additional colorization task connections are not additional paper
results.

| Family | `backbone.kind` | Encoder representation | RGB handling |
| --- | --- | --- | --- |
| Colorization | `colorization_scaling` | Custom ViT CLS / patch tokens, LayerNorm epsilon 1e-5 | sRGB to Lab L/100 |
| Rotation | `rotation_scaling` | timm ViT CLS / patch tokens, epsilon 1e-5; rotation head excluded | ImageNet mean/std |
| Split-brain | `split_brain_scaling` | Concatenated half-width branch CLS / patches, epsilon 1e-6 | Lab (L-50)/50 and ab/128 |

Both presets have total feature width 768 (base) or 1024 (large). Split-brain
uses two branches of width 384/512, with 6/8 attention heads each; it does not
reuse the existing Step-2 branch constructor, whose normalization differs.
Colorization reuses the existing custom encoder implementation. All three
encoders stay frozen and in evaluation mode. ImageNet uses L2-normalized CLS;
SSv2 averages raw frame CLS vectors. Dense tasks read final patch grids.
Rectangular stride-16 inputs use bicubic, antialiased position interpolation.
There is no silent fallback to another interpolation implementation.

## Import and run

Use a read-only copy or original native checkpoint and an independently checked
SHA-256. The importer requires the exact architecture, state mapping, literal
epoch field and literal value. It does not infer an epoch from a filename or
convert a zero-based `epoch` into `completed_epoch`. Verify the intended table
checkpoint identity separately. Example (substitute the actual hash and value):

```bash
PYTHONPATH=. python -m downstream.frozen_vit --kind rotation_scaling --source /models/native.pth --sha256 CHECKED_SHA256 --arch vit_base_patch16_224 --state-key state_dict --epoch-key completed_epoch --epoch-value 30 --out /models/rotation-export
```

Add `--module-prefix` only if all native parameter names have the `module.`
prefix. Supported state mappings are `state_dict` and `model`; epoch fields are
`epoch` and `completed_epoch`. Native encoder prefixes are fixed by the provider:
flat keys for colorization, `vit.` for rotation, and `net1.encoder.` /
`net2.encoder.` for split-brain. Only the known pretext decoder/head keys are
discarded. Missing or unexpected encoder keys, wrong tensor shapes/dtypes,
nonfinite values, wrong hashes and existing output directories are rejected.
Loading remains restricted (`weights_only=True`), with the same narrow NumPy
RNG constructor allowlist used by the existing scaling importer. Arbitrary
pickle classes are not accepted. Training state is not restored.

The output is a versioned `encoder.pt` and an `export.json` containing hashes,
explicit identity and tensor count. No native configuration, dataset path,
optimizer state or source filename is copied into either output.

Start from the validated [ImageNet example](examples/three_scaling_linear.json).
Choose one of the three kinds, the matching export and either
`vit_base_patch16_224` or `vit_large_patch16_224`. Keep `img_size: 224` and
`patch_size: 16`; architecture overrides are rejected. For the other tasks use
existing BasicFive component configurations with the same backbone mapping:

- ImageNet: `preprocessing_profile: captured_provider_v1`.
- SSv2: `video_recipe: captured_provider_video_v1`.
- COCO: `detector_profile: captured_native_detection_v1` (transposed pyramid).
- ADE20K and NYUv2: existing frozen component task heads.

Shared non-detection loaders produce ImageNet-normalized RGB. The two Lab
providers explicitly undo that normalization before the native conversion.
COCO uses identity normalization for Lab providers and ImageNet normalization
for rotation. Inputs must not be manually normalized a second time.

## Evidence and limits

The October 15 source refresh compared 52 source/configuration files against
Capture snapshot `d8e82adc094ca68624d0f98bd6425c9ec976966f` and current originals.
All three families' selected sources match; the two differing shared BasicFive
files are the already recorded historical depth-loss/clamp disagreement. This
change does not choose a historical depth recipe or change the public loss.

Behavioral tests first failed on missing provider discovery. Subsequent RED
checks exposed absent CI registration, unexpected encoder-key acceptance and
native NumPy-state deserialization. Tests now cover strict import/export,
preprocessing, token geometry, image/video readouts, frozen state and fifteen
real task-entry invocations with tiny models and synthetic datasets. Separate
private source comparisons use identical weights for all three families at
224x224 and 32x48, checking image/spatial/video features and two classifier-head
gradient/update steps against original functions. Evidence stays outside Git.

These are frozen **component integrations**, not a claim that all table scores
have been reproduced. AP/FT, full native pretraining/continuation, historical
checkpoint-to-row attribution, full-size native checkpoint loading, GPU and
multi-node parity, and full-data downstream scores remain unverified here.
Results remain noncanonical and do not become record values merely because a
provider or smoke test exists. See [overall coverage](FINAL_PAPER_COVERAGE.md).
