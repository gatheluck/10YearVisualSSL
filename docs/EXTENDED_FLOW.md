# Extended optical-flow components

This entry point ports the inspected experimental feature-difference head and
masked componentwise L1 loss for frozen LP and attentive AP. It supports the five
existing Extended spatial providers: DINOv3, RAEv2 K7, SigLIP2 G, V-JEPA2.1 and
VGGT Omega. It does **not** certify the final paper's table cells. See
[coverage](FINAL_PAPER_COVERAGE.md) and [submission scope](SUBMISSION_SCOPE.md).

## Source and protocol differences remain unresolved

The current experimental sources match the captured flow implementations.
However, the supplied Unified companion describes parameter-free correlation
and robust endpoint loss, whereas the inspected executable path uses projected
feature differences and componentwise L1. This runner explicitly retains the
experimental behavior; it does not silently reinterpret the companion. Its
`captured_flow_input_pixels_v2` transform and `capture_extended_components`
profile are mandatory. Every report records these conflicts and sets
`canonical_eligible=false` and `record_value=false`.

The final tables report MPI Sintel and Middlebury Flow. Spring is an additional
catalog component, not an additional reported paper row. The current rerun
catalog contains LP/AP entries for all five providers on Sintel and Spring;
readiness is not completion. Middlebury's exact historical run association is
not established by its available builder or older logical-job records.

## Inputs and split conversion

Provide a data root and an explicit JSON manifest with exactly `schema_version`
(integer 1), `dataset`, `split_evidence`, `train`, and `validation`. Each sample
has `a`, `b`, `flow`, `scene`, and `rendering`. Each input is either a relative file
path or `{"archive":"relative.zip","member":"path/in/archive"}`. No extraction
of archives is needed. Only exact, unique members are accepted. Paths, including
symlinks, must remain below the data root. Both splits must be nonempty, with
disjoint scenes, target identities and RGB identities. Sintel rendering is
`clean` or `final`; other datasets use the empty string. This checks manifest
consistency, not authenticity of the original dataset release.

Two native converters preserve the inspected local partitions:

- `mpi_sintel`: one `flow` root at the supplied root, `training`, or
  `raw/training`; matching `clean` and `final` scene/frame sets, consecutive
  frames and every selected `.flo` target are required. Sorted scenes are split
  80/20 before both renderings are included.
- `middlebury_flow`: `archives/other-gt-flow.zip` and
  `archives/other-color-allframes.zip`; match each `flow10.flo` to `frame10.png`
  and `frame11.png` by scene. Sorted GT scenes are split 80/20.

Both validate discovery in the complete population before partitioning. These
are **local held-out partitions**, not official benchmark evaluation splits.
To create a new manifest without overwriting an existing one:

`python -c 'import json; from downstream.extended_flow import build_manifest; json.dump(build_manifest("/data/sintel", "mpi_sintel"), open("/data/sintel-samples.json", "x"), indent=2)'`

Spring requires an explicit manifest; its native archive discovery and pair
budgets are not ported here. `.flo5` accepts only floating `flow` and an optional
binary `valid` dataset. Flow grids may equal RGB geometry or be twice as large;
displacements remain in **native RGB pixel units**, not flow-grid pixels.
BlinkVision's prepared-data revisions remain unsupported in this runner.

## Numerical behavior

RGB is resized bicubically to 224 without random augmentation. The provider
interface receives ImageNet normalization once. Targets use bilinear resizing
without antialiasing; horizontal/vertical displacements scale by 224 divided by
native RGB width/height. Validity intersects nearest-exact sampling with complete
bilinear support. Unknown `.flo` sentinels, nonfinite values and invalid support
never become valid zero targets. Each sample needs valid pixels; a stationary
sample is allowed, but an entirely zero supervision/evaluation split is refused.

The shared 1x1 projection has width 128, followed by a two-channel 1x1 head on
`project(b)-project(a)`. AP constructs its provider-specific spatial adapter
before the head. Default convolution initialization is retained. Outputs are
bilinearly upsampled without a second displacement scaling.

Metrics pool valid pixels globally: EPE, percentage with error strictly greater
than one 224-grid pixel, and separate clean/final metrics when applicable.
`epe_native_pixels_on_224_grid` converts vector units back to native RGB pixels
while keeping the 224 evaluation grid. It is **not native-grid benchmark EPE**.
Only rank zero evaluates the full validation population.

## Execution and continuation

Use [the example](examples/extended_flow.json), replacing data and encoder paths:

`python -m downstream.extended_flow --config /work/flow.json --out /work/flow-run`

The 30-epoch AdamW horizon uses reference batch 16, linear LR scaling from
0.001, weight decay 0.0001 (LP) or 0.05 (AP), one-epoch warmup from 1e-6 and cosine
to zero. Short component runs preserve that horizon. Accumulation, precision,
tail handling and schedule-clock limits follow [shared execution](EXTENDED_EXECUTION.md).
Existing output directories are protected. Input file/archive hashes,
membership and encoder identity participate in epoch continuation; frozen
encoder weights are excluded from probe/resume payloads. Set `resume` to an
earlier `resume.pt` and choose a new output directory to continue.

`torchrun --standalone --nproc-per-node=2 --module downstream.extended_flow --config /work/flow.json --out /work/flow-distributed`

CPU fixture checks cover ten actual provider LP/AP routes, single-process and
two-rank continuation, worker reads, and source output/gradient/update comparisons.
They do not validate released checkpoints, CUDA/BF16/NCCL, full-data scores,
historical row attribution, FT or native checkpoint import. Preserve the source
and companion conflicts until the original run identity is reconciled.
