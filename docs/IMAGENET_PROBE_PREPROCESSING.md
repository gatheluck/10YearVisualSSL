# ImageNet LP/AP source preprocessing

See [package scope](SUBMISSION_SCOPE.md). Set
`preprocessing_profile: captured_provider_v1` in the online ImageNet component
configuration to select the inspected provider's LP/AP image geometry. This
ports original experimental preprocessing; it does not identify the exact
historical runs behind Tables 36 and 37 or certify their scores.

[The AP example](examples/imagenet_probe.json) selects a local released model,
224-pixel inputs and the reference schedule. Use the
[downstream environment](DOWNSTREAM.md) and replace the dataset/weight paths:

`python -m downstream.imagenet --config /path/to/config.json --out /path/to/new-output`

| Provider | Training crop and evaluation resize |
| --- | --- |
| `clip_hf`, `siglip2_g`, `cradiov4_h`, `cosmos3_super_vm`, `vggt_omega` | Bicubic |
| `dinov3_hf`, `raev2_k7`, `vjepa2_1` | Bilinear |

Training uses random resized crop (scale 0.08–1, ratio 0.75–4/3), followed by
horizontal flip with probability 0.5. Evaluation resizes the short side to 256
and center-crops to 224. LP/AP do not acquire FT augmentation, erasing, batch
mixing or soft targets. ImageNet normalization at the shared input interface
is unchanged; provider-owned normalization/readout remains separate. RGB image
loading and matching train/validation class vocabularies are unchanged.

Select `adaptation: frozen` for LP, or `adaptation: attentive` with the existing
provider-specific `reader_profile` for AP. The example's cross/self-attention
reader is not appropriate for every provider; consult the
[reader guide](BASIC5_NATIVE_PATHS.md), [K7](BASIC5_K7.md) and
[vision providers](BASIC5_VISION_PROVIDERS.md). The preprocessing selection only
changes geometry, not the reader, loss, optimizer, or frozen encoder boundary.

The runner rejects unknown profiles, providers without inspected LP/AP geometry,
this selection on FT, and released-model inputs other than 224. Reduced model
fixtures may use smaller crops for component tests. Existing configurations
without `preprocessing_profile` retain the older common-bilinear component;
absence must not be interpreted as verified provider-specific preprocessing.
For [FT](IMAGENET_FINETUNE.md), continue using `finetune_recipe` instead.

Selected runs record `preprocessing.profile`, `preprocessing.interpolation` and
`preprocessing.historical_run_verified: false` in `results.json`, while retaining
`canonical_eligible: false` and `record_value: false`. A successful component
run does not establish historical dataset/checkpoint identity or paper accuracy.

Tests cover all eight providers on both LP/AP routes, deterministic training and
evaluation pixels/RNG, frozen-backbone preservation, model-head updates, output
contracts and invalid selections. Private direct-source comparisons and mutation,
regression and CI outcomes are recorded separately in the PR. Full-size weights,
CUDA numerics, distributed/BF16 execution and full-data scores remain unverified.
