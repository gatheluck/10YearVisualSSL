# Extended first-mask video segmentation

See [package scope and terminology](SUBMISSION_SCOPE.md). This is the portable
DAVIS 2017 LP/AP component path for the five existing Extended providers,
not certification of historical runs or the final paper's scores.

## Inputs and execution

Keep original frames, annotations and weights read-only. The native converter
reads `ImageSets/2017/train.txt` and `val.txt`, matching sorted JPEG frames to
PNG object-ID masks. It selects `480p` only when both `JPEGImages/480p` and
`Annotations/480p` exist; otherwise it requires `Full-Resolution` in both trees.
Every sequence requires at least three aligned frames and objects in its first
mask. Sequence IDs must be unique across splits. Later masks cannot introduce
new objects or change dimensions. Palette masks retain numeric IDs; RGB masks
are rejected. Counts and successful parsing do not authenticate the release.

`python -m downstream.vos_data --data-root /data/DAVIS --out /data/DAVIS/samples.json`

The converter refuses an existing output. The manifest contains
`schema_version: 1`, `dataset: davis2017`, `split_evidence`, `train` and
`validation`. Each sequence has `name`, `frames` and `masks`; every asset is a
relative `{"path": "..."}` with an optional ZIP `member`. The loader validates
all assets before model construction, rejects escaping/missing/duplicate paths
and cross-split image-byte overlap, and hashes RGB and mask bytes for continuation.
Private data and generated memberships stay outside Git.

Copy [the execution example](examples/extended_vos.json), set actual local data
and encoder paths, and select the existing provider's explicit LP/AP reader.

`python -m downstream.extended_vos --config /work/run.json --out /work/run-001`

`torchrun --nproc-per-node=2 --module downstream.extended_vos --config /work/run.json --out /work/run-002`

The five providers are DINOv3-7B, SigLIP2 Giant, RAEv2-K7, VGGT-Omega and
V-JEPA 2.1. LP freezes spatial encoder features. AP applies the provider's
existing spatial adapter independently to the template and search features.
The encoder stays frozen and in evaluation mode. Paired feature normalization
is shared with [optical flow](EXTENDED_FLOW.md), preserving that runner's behavior.

## Captured behavior

- RGB uses bilinear fixed 224 resizing. Provider normalization is applied once.
- Training enumerates every first-frame object against every training-sequence
  frame, including the first frame. The initial object mask uses area occupancy;
  search targets use nearest interpolation. IDs other than the selected object
  are background; label 255 contributes no BCE gradient.
- The head projects both feature maps, normalizes channels, pools the masked
  template descriptor and predicts binary logits from channelwise correlation.
  Small first-frame objects retain occupancy at the token grid.
- Evaluation fixes templates and masks from the first frame. It does not read
  later ground truth for prediction, update templates or fit at evaluation time.
  Object logits are resized bilinearly to native frame resolution and compete
  against background logit zero. Ties select background, then the lowest ID.
- J and F use native masks. Boundary tolerance is `ceil(0.008 * image diagonal)`.
  Neither endpoint frame is scored. Scores are averaged over each object's
  frames, then over all objects, rather than equally weighting sequences.
  Missing/extra predictions, unknown IDs and incomplete sequence populations fail.

**Metric limitation:** the inspected evaluator treats void pixels as background
when forming evaluation object masks, even though training BCE ignores them.
The port preserves and discloses that behavior. Comparison with official void
handling and historical result attribution remain unverified; this path must
not be presented as independently certified benchmark scoring.

The source schedule is 30 epochs, seed 0, AdamW, effective-batch LR scaling,
1-epoch warmup from 1e-6 and cosine to zero. LP/AP weight decay differs.
[Shared execution and continuation](EXTENDED_EXECUTION.md) supplies accumulation,
provider tail policy, FP32/CUDA BF16 selection, distributed training and per-rank
RNG restoration. Evaluation runs on rank zero over complete sequences. Set
`resume` to an earlier `resume.pt`, raise the requested epoch count and use a new
output directory. Membership bytes, encoder state and configuration remain bound
to that checkpoint identity. Existing outputs cannot be overwritten.

Outputs include the run manifest, named J/F/J&F metrics in percentages,
`results.json`, `probe.pt` and `resume.pt`. Encoder weights are excluded from the
probe export. Every result retains `canonical_eligible: false` and
`record_value: false`.

## Verification boundaries

Tests exercise actual reduced models for all five LP/AP interfaces, frozen
encoder gradients, preprocessing, invalid inputs, exact-population evaluation,
CLI contracts and continuous-versus-resumed states. Reference comparisons cover
head initialization, outputs, gradients and three AdamW updates, plus mask metrics
and RGB resizing. Two-process CPU execution with input workers is tested.

Released full-size weights, GPU/BF16/NCCL, complete DAVIS measurements, native
checkpoint import and whole-builder random-number consumption parity remain
unverified. The current experimental catalog's runnable status is not evidence
of a completed run; its historical failure reasons must not be confused with the
current implementation. FT and other tracking benchmarks are outside this port.
See the [remaining coverage](FINAL_PAPER_COVERAGE.md).
