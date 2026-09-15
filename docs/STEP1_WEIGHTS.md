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
