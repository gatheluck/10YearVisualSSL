# Rotation scaling training

See [submission scope](SUBMISSION_SCOPE.md) and [paper terminology](PAPER_TERMINOLOGY.md).
This is an explicit portable component for the indexed ImageNet-21K 90-epoch
rotation experiment in Appendix B.9 and Table 21. It does not replace the
historical AlexNet or Step-2 ViT paths and does not certify a reproduced score.

## Supported behavior

- ViT-B/16 and ViT-L/16, native four-class timm head initialization, LayerNorm
  epsilon 1e-5, dropout 0.1 and no pretrained download.
- Resize to 256, random 224 crop, horizontal flip, then all four counterclockwise
  right-angle rotations with ImageNet normalization. One source image produces
  four classified images in source-image-major order.
- AdamW at peak 0.0006, betas 0.9/0.999 and weight decay 0.05. Warmup uses 10,000
  optimizer updates with a 1e-8 floor, followed by cosine decay to 1e-6 over the
  fixed 90-epoch horizon. Stopping early does not shorten that horizon.
- Explicit FP32 or CUDA BF16, gradient accumulation and norm clipping at 1.0.
  Incomplete final accumulation groups execute the native forward/backward
  operations and consume RNG, but do not update weights. Reported epoch loss
  weights these groups by their microbatch count.
- Single-process and replicated DDP execution, distributed sampler tail removal,
  rank-local input-error propagation, and portable epoch-boundary continuation.
- Read-only indexed-tar inputs, frozen membership and validation-filename
  exclusions, shared with the existing indexed scaling reader. This is not a
  content-level duplicate audit. No gray-image fallback is used.

## Running

Use the method's locked environment and start from the repository root. Edit
[the resolved configuration](examples/rotation_scaling.json) to point to the
prepared indexed tree and ImageNet validation directory. `microbatch` counts
source images per rank; the effective source batch is
`WORLD_SIZE * microbatch * accum`, and the rotation-image batch is four times
that number. The original inspected recipe used seed 42 and 1,024 source images
across 32 ranks. Other explicit geometries are component configurations.

`PYTHONPATH=. python methods/06_rotation_prediction/rotation_scaling.py --config /path/to/resolved.json --out /path/to/new-output`

For a verified allocation, launch the same script with `torchrun`, preserving
`PYTHONPATH=.` and selecting `device: cuda`. The tool does not submit jobs or
choose a cluster queue. Use only an independently verified reservation.

Outputs include `checkpoint_latest.pth`, `encoder.pt`, membership records,
metrics, `results.json` and a contract manifest. Epochs 30/60/90 retain separate
checkpoint and encoder files. The head-free export loads directly through the
[frozen rotation provider](SCALING_BASIC5_EVALUATION.md); it is distinct from
legacy Step-2 exports. Existing output directories and overlapping input/output
roots are refused.

To continue, set `resume` to a previous portable `checkpoint_latest.pth`, increase
`epochs` within 90, and choose a new output directory. Membership, world size,
architecture, precision, batch geometry and the other configuration fields must
remain unchanged. The checkpoint restores weights, AdamW and every rank's RNG,
including the loader generator. Missing/corrupt states are rejected. Native
historical full-training checkpoint conversion is not implemented here.

## Verification and limits

Capture revision `d8e82adc094ca68624d0f98bd6425c9ec976966f` and freshly read originals
agree for the selected rotation model, dataset, trainer and configuration files.
An independent harness executes the original initialization, transform/rotation
methods, schedule and microbatch update statements. Reduced CPU FP32 tests match
initialization and augmented inputs exactly, plus gradients/weights through two
updates and an incomplete tail (atol 3e-7, rtol 3e-6).

Local tests cover actual task contracts, retained 30/60/90 encoder exports,
strict resume rejection, continuous
versus resumed model/AdamW equality, two-rank updates against a global-batch
oracle, and rank-local read failure propagation. Method-only locks need no
unrelated dense-task readers for these tests. CI explicitly registers them in
the downstream job as well.

The portable loader uses a dedicated saved generator and restarts workers at
epoch boundaries; the native loader uses persistent workers and process-global
sampling RNG. Historical sample order and multi-worker continuation parity are
therefore not established. CPU two-rank tests are not multi-node CUDA evidence.
Full-size training, BF16 numerical parity, 90-epoch/full-data results, native
checkpoint continuation and historical run-to-table attribution remain pending.
Colorization and SplitBrain native scaling training are separate remaining
ports. All outputs remain `canonical_eligible: false`.
