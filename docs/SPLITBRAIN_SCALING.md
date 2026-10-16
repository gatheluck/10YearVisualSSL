# Indexed SplitBrain scaling training

This separate entry integrates the original scaling implementation for Appendix
B.9 / Table 22. It does not replace the older AlexNet or unified Step-2 entry.
See [scope](SUBMISSION_SCOPE.md), [paper terminology](PAPER_TERMINOLOGY.md)
and [frozen evaluation](SCALING_BASIC5_EVALUATION.md).

## Recipe and execution

The two ViT branches use the source timm factory, including LayerNorm epsilon
1e-6: B has width/depth/heads 384/12/6 per branch, L has 512/24/8. L predicts
313 ab bins and ab predicts 50 L bins. Both classifiers receive equal-weight
cross entropy; targets are nearest-neighbor resized to the decoder grid.
The original contiguous decoder input/output layout is preserved.

RandomResizedCrop(224) and horizontal flip precede Lab conversion. Choose
`lab_backend` explicitly: `fallback` uses the source float32 D65 conversion;
`skimage` requires scikit-image in the selected environment and never silently
falls back. The original chose this backend according to installed packages;
the historical run's backend must be established before claiming recipe parity.
Targets are reconstructed from normalized float32 inputs, using the verified
313-point codebook and torch float32 squared-distance argmin in 16-row chunks.
This preserves the source deferred-quantization rounding order.

AdamW uses betas 0.9/0.999, weight decay 0.05 and no gradient clipping. The
learning rate is fixed at peak 0.0006, with 10,000 update warmup and minimum
1e-6 over 90 epochs; changing physical batch does not scale it. The shared
rotation schedule implements the same formula. Accumulation divides each loss
by `accum`; incomplete tails still execute forward/backward and contribute to
reported loss, but their gradients are discarded without an optimizer update.
FP32 and CUDA BF16 are explicit. Nonfinite inputs, loss or gradients fail closed.

Edit the [resolved configuration](examples/splitbrain_scaling.json) and run:

`PYTHONPATH=. python methods/08_split_brain/splitbrain_scaling.py --config /path/to/resolved.json --out /path/to/new-output`

The existing method lock supports `fallback`; `skimage` needs an explicitly
managed environment containing that optional package. For DDP use `torchrun`
inside a verified allocation. Global batch is microbatch times world size times
accumulation. Original configurations use effective batch 1,024, seed 42 and
BF16. Smaller diagnostic runs remain explicitly noncanonical.

The shared indexed reader freezes membership and excludes validation filenames.
It does not certify content-level overlap absence. Input roots are read-only;
existing outputs and overlapping input/output locations are refused. Use a fresh
personal run directory for logs, scheduler output, temporary files and artifacts.
No queue is selected or submitted by this entry.

## Outputs and limits

Outputs contain an atomic latest training checkpoint, per-rank RNG, membership,
configuration, optimizer state, metrics and a hashed contract manifest. Epochs
30/60/90 retain checkpoints and head-free exports. `encoder.pt` uses the existing
frozen provider's ModuleList keys and supports its BasicFive evaluation routes.

This is a fresh-run entry: native or portable checkpoint continuation is not
implemented, and `resume` is rejected. A stored optimizer/RNG record does not
imply a tested continuation loader. The portable loader uses a dedicated
sampler generator and restarts workers each epoch; historical persistent-worker
sample ordering has not been reproduced. All runs report
`canonical_eligible: false`.

The selected source/configuration files match the current originals and Capture
revision `d8e82adc094ca68624d0f98bd6425c9ec976966f`. Private reduced CPU comparisons
match initialization, fallback preprocessing, deferred targets, gradients and
three accumulated FP32 parameter updates exactly. Seven schedule points agree
within 1e-15 relative error. Scikit-image numerical parity has not been verified.
Tests additionally cover real indexed training/export loading, retained epochs,
partial tails, invalid inputs and two-rank gradient/read-error behavior. A
separate reserved-node CUDA BF16 diagnostic ran both full-size presets for four
updates on a four-image synthetic indexed fixture with finite model state.
These checks do not establish complete-data scores, multi-node behavior, historical
checkpoint identity or full-paper reproduction. Consult the PR for separately
recorded GPU and final test-gate results; private paths and evidence stay outside
Git.
