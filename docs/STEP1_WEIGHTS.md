# Reproducing Step-1 features from identified checkpoints

The extraction driver already discovers feature providers, but an original training checkpoint is not necessarily the exported `encoder.pt` a provider expects. Never choose the newest filename or substitute publisher weights for a lab-trained run without recording a distinct target.

## Weight acquisition

Artifact entries in method provenance specify a filename and SHA-256. A verified public URL can be fetched using the existing tool:

`python3 bin/fetch-weights.py --provenance methods/28_dinov2/provenance.json --artifact backbone_artifact --out .weights/28_dinov2`

When no public URL has been verified, an entry records `availability: user_supplied` and acquisition instructions. The command fails with those instructions until the exact file is supplied. It never invents a URL or silently uses different weights:

`python3 bin/fetch-weights.py --provenance methods/14_simclrv1/provenance.json --artifact step1_native_artifact --source /path/to/checkpoint_epoch_1000.pth --out .weights/14_simclrv1-step1`

This copies the input into a separate destination after verifying its bytes. It never modifies the source. Public release/access to lab-trained weights is not established by adding a provenance entry. Retraining with the same seed is not a guarantee of the same checkpoint bytes.

## Native export: first verified checkpoint

SimCLR v1's recorded Step-1 result names its 1000-epoch ResNet-50 checkpoint. The native file records zero-based epoch 999 and seed 42; its SHA-256 is pinned under `step1_native_artifact`.

Run the exporter in the method's dependency environment. It verifies the source hash, unwraps only the explicit state key, removes only a leading prefix, refuses key collisions, delegates encoder selection and loading to the existing method adapter, and writes to a new output directory:

`python3 bin/export-native-encoder.py --source .weights/14_simclrv1-step1/checkpoint_epoch_1000.pth --sha256 e945c6bd144640f174e90ef924890fd47280f522a10d20ee01054be234d83b13 --method-dir methods/14_simclrv1 --config methods/14_simclrv1/configs/linear_eval.yaml --state-key state_dict --strip-prefix module. --out .weights/14_simclrv1-step1-export`

`export.json` records the source and exported hashes, wrapper mapping, configuration and torch version. A successful load is not sufficient evidence of feature parity: compare the exported encoder with the recorded native model before full extraction. Different torch versions may serialize identical tensors into different file bytes; the original checkpoint hash and tensor/feature checks remain essential.

## Extract with the existing driver

Use a fresh output root for this Step-1 checkpoint; preserve previous runs. This driver records other providers without supplied weights as skipped:

`python3 bin/extract-features.py --data-root /path/to/imagenet --split val --out /path/to/new-step1-features --device cuda --batch-size 128 --num-workers 8 --representation l2 --encoder 14_simclrv1=.weights/14_simclrv1-step1-export/encoder.pt --allow-missing`

For the SimCLR target, evaluation uses bicubic resize of the shorter side to 256, center crop 224, ToTensor and no channel normalization. This matches the inspected native evaluation transform. L2 normalization is applied by the common driver, as in the existing feature sweep. `raw` is a separate explicit representation.

Verify 50,000 rows, 2,048 dimensions, float32 finite features, 1,000 classes with 50 images each, consistent class ordering, source/export hashes and unit norms for L2 output. Runtime execution evidence is recorded in STEP1_EXTRACTION_PROGRESS.md. Do not interpret a skipped model as extracted.

## Cluster execution and privacy

Use the locally supplied reserved-node configuration only. Follow the cluster's documented reserved-node submission form with placeholders: `qsub -P <group> -q <reservation> -l select=1 -l walltime=01:00:00 -v RTYPE=rt_HF <script>`.

Concrete account/group/reservation identifiers, original private paths and job logs live in `$HOME/.local/state/10YearVisualSSL/abci/`, outside the repository. Never force-add that directory. All original experiment files and weights must be mounted read-only; only a separate task workspace is writable. There is no fallback to a standard queue.

## Batch preparation

`bin/prepare-native-encoders.py` reads a local JSON object mapping method names
to native checkpoint paths. It preflights source existence, recipe fields and
output paths before launching any worker (tensor and checksum validation run
in each worker), reads `step1_native_artifact` in each method's provenance, and invokes
`export-native-encoder.py` in a separate process per model. Use an environment
compatible with the selected methods. The method's `configs/linear_eval.yaml`
is used for both export and extraction; select only checkpoints whose backbone
settings and evaluation preprocessing have been verified against that config.

Example local sources file (replace the placeholder paths):

```json
{
  "10_inst_disc": "/path/to/instdisc/checkpoint_epoch_200.pth",
  "13_mocov1": "/path/to/mocov1/checkpoint_epoch_200.pth",
  "15_mocov2": "/path/to/mocov2/checkpoint_epoch_200.pth",
  "16_simclrv2": "/path/to/simclrv2/checkpoint_epoch_800.pth",
  "20_simsiam": "/path/to/simsiam/checkpoint_epoch_100.pth"
}
```

`python3 bin/prepare-native-encoders.py --sources /path/to/sources.json --out /path/to/new-encoders`

Each successful model produces `<out>/<method>/encoder.pt` and `export.json`.
The batch exits nonzero on failure, reports per-model exit codes in `batch.json`,
and retains separate logs. Interrupted batches remain `incomplete`. The output
root must be new: partial outputs are not silently treated as reusable results.
To retry failures, supply only those models and choose a new output root.
Sources files, output metadata and logs may contain private paths; keep them
outside the repository.

The checkpoints above are lab-trained, selected from the recorded Step-1
linear evaluations at their configured training endpoints. Their exact hashes
are pinned in provenance. No public URL for those exact bytes is verified;
obtain them from the lab and use the checksum-verified local acquisition path.
Once native/exported feature parity is verified, feed the resulting encoders
to the existing `extract-features.py` driver. Keep L2 representation and the
same ImageNet val class ordering as the preceding extraction.

SwAV and SeLa are also pinned and validated by the same batch path; see
`STEP1_EXTRACTION_PROGRESS.md` and `STEP1_BATCH_RESULTS_20260916.json` for the
seven-model results.


## Additional native layouts and evaluation transforms

BYOL, visual CPC 2018 and Colorization now have pinned native checkpoint
recipes. Their exact lab-trained bytes have no verified public download URL;
obtain the matching file from the lab and use the existing checksum-verified
local acquisition and batch preparation commands. A public checkpoint from
the same model family is not an interchangeable source for these recipes.

An optional `module_map` in a recipe maps exact top-level native module names
to port module paths. For example, `{"conv1": "encoder.0"}` renames
`conv1.weight` but preserves `conv1_extra.weight`. Unused names and key
collisions are rejected. Batch preparation forwards this mapping automatically;
direct exports accept it as JSON with `--module-map`. The export record saves
it alongside the wrapper and prefix mapping.

Colorization maps the native named convolution/batch-normalization layers to
the port's sequential encoder and preserves the native evaluation's float32
sRGB gamma conversion and float64 Lab arithmetic. Training Lab conversion is
unchanged. BYOL validation uses bicubic resize to `ceil(image_size / 0.875)`,
then center crop and ImageNet normalization. Visual CPC 2018 validation resizes
to a source-size square with bilinear interpolation, center crops, applies
ImageNet normalization and averages patch-encoder features over the grid.

## Checkpoint-specific extraction protocols

MoCo v3, DINO, MAE and SimMIM have pinned lab-checkpoint recipes. The exact
files are user-supplied: no public URL for these bytes has been verified.
The existing acquisition command checks SHA-256 before accepting a local file.
Batch preparation now forwards the recipe's optional `feature_options` to
`export-native-encoder.py` and records it in `export.json`:

| Method | Recorded native feature | Validation override |
|---|---|---|
| MoCo v3 | Base ViT-B CLS, 768 dimensions | Bilinear Resize(256) |
| DINO | Teacher last four normalized CLS tokens, concatenated; 1,536 dimensions | Existing bicubic Resize(256) |
| MAE | ViT-L CLS, 1,024 dimensions | Existing bicubic Resize(256) |
| SimMIM | Swin-B mean pooled tokens, 1,024 dimensions | Bilinear Resize(219), crop 192 |

**Transfer `encoder.pt` and `export.json` together.** Providers validate the
method name, encoder SHA-256 and supported option names before applying the
profile. A mismatched file or unknown option fails explicitly. Without this
sidecar, or with an older record containing no `feature_options`, the provider
uses its existing defaults (including final-layer DINO CLS and MAE average
pooling). Removing the sidecar therefore loses the native protocol selection.
The sidecar is an auditable configuration record, not a signed certificate.

Images retain deterministic center cropping and ImageNet normalization. The
provider emits raw backbone features before the linear classifier or its
train-set standardization; use `--representation l2` in the extraction driver
for the visualization collection. GPU parity and full-val completion are
recorded separately in `STEP1_EXTRACTION_PROGRESS.md`.

## Additional seven native recipes

VAE, DeepCluster, Split-Brain, BEiT, iBOT, I-JEPA and NEPA now have exact
checkpoint hashes and native export recipes. These are lab-trained files with
no verified public download URLs; the user-supplied acquisition flow above
refuses any other bytes. Checkpoint epoch values are recorded as stored, since
not every trainer uses zero-based epochs.

| Method | Native representation | Explicit protocol |
|---|---|---|
| VAE | Posterior mean, 50 dimensions | Input 224; bilinear resize 256; no ImageNet normalization |
| DeepCluster | AlexNet fc7, 4,096 dimensions | Preserve actual frozen Sobel weights |
| Split-Brain | Two averaged branch features, 512 dimensions | Native NumPy float32 Lab conversion |
| BEiT | Mean of patch tokens, 768 dimensions | Bilinear resize 256 |
| iBOT | Last four teacher CLS tokens, 1,536 dimensions | Four-block concatenation |
| I-JEPA | Target encoder mean patch token, 1,280 dimensions | Explicit target_encoder state |
| NEPA | EMA causal predictor mean, 768 dimensions | Explicit ema_state_dict state |

`feature_options.config_overrides` merges into the shipped evaluation config
at both export and inference. Only existing keys with matching types are
accepted; nested defaults are preserved and input configs are not mutated.
The VAE shape and iBOT layer count therefore agree at both stages.
An optional recipe `add_prefix` (direct CLI: `--add-prefix`) places explicitly
selected bare state keys under an adapter namespace after module remapping.
It does not guess between online and target weights.

DeepCluster's old exports omitted the Sobel front-end even though the native
model's initializer overwrites its initial filter values. Those exports cannot
reproduce the original inputs; regenerate from the full checkpoint. The loader
now rejects missing Sobel tensors. The reset classification head remains
excluded. Split-Brain's native Lab mode is explicit and leaves ordinary
training/default conversion unchanged; it corresponds to the original
NumPy fallback, not a scikit-image conversion.


### Additional native evaluated variants

Context Prediction, Jigsaw, Rotation, Jigsaw++ and Barlow Twins now have exact
`step1_native_artifact` records, consumed by the existing verified acquisition
and batch-export commands. Supply independently obtained original files; no
unverified download URL is substituted. Context Prediction selects the evaluated
official-style million-step checkpoint. Barlow's bare backbone uses an explicit
`backbone.` namespace and has no verified pretraining epoch. Rotation maps exact
native block names to the port's encoder and rejects unknown or mixed namespaces.

The hash-bound native profiles select normalized ImageNet input and full-image
CFN pool4 for Jigsaw, pre-pool5 conv5 global averaging for Rotation, and the
knowledge-transfer AlexNet for Jigsaw++. Their ordinary port paths are unchanged
when no native profile is supplied. Preserve each export's paired `export.json`.
The actual five-output hashes and four-image GPU parity results are recorded in
`STEP1_REMAINING_RESULTS_20260916.json`.


### Official architectures: Context Encoder and DINOv3

These two evaluated checkpoints require different architectures from the port's
training defaults. Their `step1_native_artifact` records work with the same
`fetch-weights.py --source`, `prepare-native-encoders.py` and extraction driver.
Initialize pinned submodules with `git submodule update --init --recursive`.
Supply the exact evaluated checkpoints; no public URL has been verified for
these exact artifacts, so neither tool substitutes a similarly named download.

| Method | Explicit profile | Input and representation |
|---|---|---|
| Context Encoder | `official_caffe_pool5` | 227 crop, BGR 0-255 mean subtraction, flattened pool5 (9216) |
| DINOv3 | `official_vitb16_cls512` | Resize585/crop512 bicubic, ImageNet normalization, normalized CLS (768), CUDA float16 autocast |

DINOv3 imports only the backbone from the pinned official submodule, under its
upstream DINOv3 License. The original evaluation did not record its upstream
commit; the new pin makes this implementation reproducible but does not establish
which historical source revision produced the original linear-probe score.
Checkpoints load strictly. The profile and checkpoint must agree, and the paired
export sidecar's encoder hash is checked before selecting the native path.
