# Supplementary code: reviewer guide

This archive accompanies the paper's visual representation learning experiments.
It contains model implementations, feature extraction tools, evaluation and
training components, tests, **16 protocol and scope documents**, and **three Extend JSON
registries**. This guide helps you locate the material relevant to an experiment
and check the package without downloading datasets or model weights.

## Start here

1. **Read the evaluation specifications:** the [protocol index](docs/submission_protocols/README.md)
   links Basic5 and Extended linear probing, attentive probing/pooling and
   fine-tuning, along with scope notes and frontier-model prompts.
2. **Check the earlier experimental conditions:** the [initial protocol index](docs/initial_protocols/README.md)
   covers original-setting and category-specific comparisons, plus controlled pre-training. Each document distinguishes historical evidence from later
   proposed unifications. Use this distinction when interpreting a reported result.
3. **Inspect the corresponding implementation:** use the table below. If you
   want to try the package first, the optional CPU check after the table needs
   only Python 3.12.

## Find an experiment or protocol

| Material to inspect | Where to start | Code or implementation guide |
| --- | --- | --- |
| ASIS: original-setting evaluation (Section 3) | [Historical ASIS conditions](docs/initial_protocols/INITIAL_STEP1_v1.md) | [Checkpoint preparation and feature extraction](docs/STEP1_WEIGHTS.md); per-model guides in [methods/](methods/) |
| CTRL: controlled VSSL pre-training (Section 5.1) | [Historical controlled pre-training conditions](docs/initial_protocols/INITIAL_STEP2_v1.md) | Per-model training configurations and adapters in [methods/](methods/); [method naming](docs/METHOD_NAMING.md) |
| Category-specific model comparisons (Section 3) | [Initial protocols by family](docs/initial_protocols/README.md) | [Vision-provider coverage](docs/BASIC5_VISION_PROVIDERS.md) and [SAM3/Cosmos3 components](docs/BASIC5_PATCH_PROVIDERS.md) |
| Unified Basic Five LP/AP/FT evaluation (Section 4; Appendix E) | [Linear](docs/submission_protocols/BASIC5_LINEAR.md), [attentive](docs/submission_protocols/BASIC5_ATTENTIVE.md), [fine-tuning](docs/submission_protocols/BASIC5_FINETUNE.md) | [Downstream execution](docs/DOWNSTREAM.md); [implemented components and recipe boundaries](docs/BASIC5_PROTOCOL.md) |
| Extended evaluation (Section 4; Appendix E) | [Specifications and three JSON registries](docs/submission_protocols/README.md) | Per-dataset settings are in the registries; full Extended trainer/task integration is not included in this package |
| Learning-design diagnostic studies (Section 5.2) | [DINOv3 heads, Gram stage and continuation](docs/DINOV3_STEP4.md); [IDv2 objectives and banks](docs/IDV2_COMPONENTS.md) | [DINOv3 adapter](methods/31_dinov3/README.md); [Instance Discrimination adapter](methods/10_inst_disc/README.md) |
| Frontier-model prompting | [Supplied system prompts](docs/submission_protocols/BASIC5_FRONTIER_SYSTEM_PROMPTS.md) | Prompt specifications; full response records are not included |

These are navigation links, not a claim that every model–task combination is
executable. Each implementation guide states its supported paths and validation
limits. The [scope guide](docs/SUBMISSION_SCOPE.md) explains the distinction
between original experimental implementations, their integration here, and
verification of reported results.

## Legacy labels in files and code

Older paths and comments use `Step 1` through `Step 4`. These are internal
experiment-group labels, not manuscript section numbers or a required sequence
of four runs. See the [manuscript terminology and legacy-label map](docs/PAPER_TERMINOLOGY.md)
for their scope and exceptions. Filenames and configuration identifiers remain
unchanged so existing commands still resolve.

## Optional check: no datasets, weights or GPU

Extract the archive and open a terminal in the **`code/` directory containing
this README**. With Python 3.12 installed, run:

```bash
python3.12 -m unittest discover -s tests -p test_end_to_end.py -v
```

Expected outcome: **19 tests, ending in `OK`**. No additional Python packages are
needed for this check. It exercises configuration resolution, a synthetic
reference adapter and output validation, using temporary files. It checks that
the core tools work together; it does not train a model or reproduce a paper score.

## Running an experiment

1. Choose the model and evaluation track using the table above. Read its guide
   before selecting a configuration: architecture, feature readout, preprocessing
   and metric can differ across tracks.
2. Create an isolated Python 3.12 environment and follow that method's dependency
   and lock-file instructions. Select the documented CPU or CUDA environment;
   the optional check above does not install training dependencies. See the
   [device guide](docs/GPU.md) for device selection.
3. Obtain the required dataset and checkpoint separately. Follow the documented
   data layout and replace local placeholders. In the Extend JSON registries,
   `${DATA_ROOT}/<dataset-id>` requires explicit substitution by a consuming
   loader; JSON does not expand environment variables automatically.
4. Use the method guide or [downstream execution guide](docs/DOWNSTREAM.md) for
   the invocation and expected outputs. Check the chosen path's support and
   validation status before starting a full run.

## Contents and how to interpret them

- **Code:** `methods/` contains model adapters and configurations; `adapterlib/`
  contains shared execution utilities; `downstream/` contains evaluation/training
  components; `bin/` contains command-line tools; `tests/` contains validation tests.
- **Evaluation specifications:** `docs/submission_protocols/` contains eight
  supplied documents and the matching 81-configuration Extend LINEAR registry
  with ATTENTIVE and FINETUNE overrides.
- **Initial experiment companions:** `docs/initial_protocols/` contains eight
  edited reconstructions. Historical conditions and proposed new settings are
  separated. Proposed changes, including RAEv2 readout, 4DFM resolution and
  VideoSSL schedules/seeds, must not be treated as completed historical reruns.
- **Scope of the code release:** original experimental implementations include
  pipelines whose full model/task recipes are only partly ported or validated
  here. Documentation delivery, executable coverage and reproduction of an
  individual reported score are separate. The guides identify these boundaries.
- **External inputs:** datasets, pretrained weights, full experiment outputs and
  result tables are not bundled. Relative references inside reconstructed
  protocols may name files in the original experimental archive rather than
  runnable files supplied here.

## Bundled sources and licenses

Selected upstream source is already in `third_party/`; Git initialization and
cloning a private repository are not needed. Some method guides retain the word
“submodule” from the development repository. In this archive, use the bundled
files. Development CI, agent instructions and handoff logs are excluded; references
to them in historical guides concern repository maintenance.

First-party code uses the [MIT license](LICENSE), with an anonymous holder label
for review. Upstream licenses and copyright notices are retained; follow each
component's terms, including any research-use restrictions. The [MAR guide](methods/mar/README.md)
describes its bundled device-preserving source patch.
