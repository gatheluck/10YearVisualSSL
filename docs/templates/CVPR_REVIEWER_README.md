# Supplementary code for review

Run commands from the extracted `code/` directory. The package contains evaluation and training entry points, model adapters, protocol references, figure analysis, and focused tests. Supply your own datasets and appropriately licensed checkpoints for training.

Open the [current evaluation protocols](#evaluation-protocols) or jump to the [files cited in Appendix E](#files-cited-in-appendix-e).

## Quick verification

From the extracted `code/` directory, activate a clean Python 3.12 environment. On Linux x86_64, install the [locked CPU dependencies](downstream/requirements.lock.txt):

```sh
python -m pip install --require-hashes --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r downstream/requirements.lock.txt
```

On other platforms with compatible PyTorch wheels, use `python -m pip install -r downstream/requirements.txt`. Run the following 19 tests with local fixtures, then check the three command interfaces:

```sh
python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe.DenseProbe.test_missing_submodule_reports_actionable_error
python -m downstream.revised_tasks --help
python -m downstream.revised_reporting --help
python -m downstream.representation_analysis --help
```

Expect 19 tests to pass without skips and each help command to print its options. If the archive also contains the pinned DPT implementation at `third_party/geosubprobe/evals/models/probes.py`, run the full 23-test check:

```sh
python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe
```

Expect 23 tests to pass without skips. Both checks use fixture data and require no model weights.

## Evaluation protocols

The following are the primary Basic Five protocols for the experiments in this paper. Each row links to the full specification in Markdown and JSON. LP, AP and FT use v1. The LoRA and DPT v2 documents describe the executed settings reconstructed from the training code and run records. The scheduled seeds are 0, 1 and 2. See [COCO metrics and completed FT repeats](docs/BASIC5_V1_AUDIT.md#coco-evaluation-and-completed-repeats) for the reported populations and final-epoch results. In Table 26, 18 of 25 full fine-tuning entries have three runs, VGGT-Omega on SSv2 has one run, and six entries are blank.

| Method | Protocol | Implementation and adopted run details |
| --- | --- | --- |
| Linear probing (LP) | [Markdown](docs/current_protocols/BASIC5_LINEAR_v1.md) · [JSON](docs/current_protocols/BASIC5_LINEAR_v1.json) | Frozen encoder and task head. See the [optimization and evaluation audit](docs/BASIC5_V1_AUDIT.md#what-was-compared). |
| Attentive probing (AP) | [Markdown](docs/current_protocols/BASIC5_ATTENTIVE_v1.md) · [JSON](docs/current_protocols/BASIC5_ATTENTIVE_v1.json) | For Figure 3 SSv2, see the [adopted reader and input tokens for each encoder](docs/SSV2_EXECUTION.md#figure-3-attentive-probes). |
| Full fine-tuning (FT) | [Markdown](docs/current_protocols/BASIC5_FINETUNE_v1.md) · [JSON](docs/current_protocols/BASIC5_FINETUNE_v1.json) | End-to-end training. The [execution audit](docs/BASIC5_V1_AUDIT.md#what-was-compared) records deviations, including ImageNet label smoothing. |
| LoRA | [Markdown](docs/current_protocols/BASIC5_LORA_v2.md) · [JSON](docs/current_protocols/BASIC5_LORA_v2.json) | [Task-specific schedules](docs/CVPR_ADAPTATION.md#revised-image-and-semantic-task-cli) and [DINOv3 shorter runs](docs/CVPR_ADAPTATION.md#author-supplied-november-v2-run-descriptions). |
| DPT dense probing | [Markdown](docs/current_protocols/BASIC5_DPT_v2.md) · [JSON](docs/current_protocols/BASIC5_DPT_v2.json) | [Four-tap decoder](docs/CVPR_ADAPTATION.md#training-facing-checks-and-four-tap-dense-probe) for ADE20K and NYUv2. For NYUv2 the paper uses unaligned RMSE. |

For the 40 Extended transfer settings, see the [Figure 5 dataset and metric ledger](docs/CVPR_EXTENDED_COVERAGE.md). The executed semantic metrics for SpaceNet 2 and SUN RGB-D are documented there.

## Figures and analyses

Start with the [reproduction guide for Figures 1–7 and Appendices A–E](docs/PAPER_REPRODUCTION_GUIDE.md). It gives the input and verification procedure for each figure.

| Paper experiment | Where to start |
| --- | --- |
| Figure 3, five encoders and Basic Five tasks | [LoRA and DPT training](docs/CVPR_ADAPTATION.md), [three-seed comparisons](docs/LORA_FULL_REPRODUCTION.md), [SSv2 attentive probes](docs/SSV2_EXECUTION.md#figure-3-attentive-probes) |
| Figure 5, 40 transfer settings | [Dataset and metric ledger](docs/CVPR_EXTENDED_COVERAGE.md), [LP/AP comparisons](docs/TRANSFER_COMPARISON.md) |
| Representation geometry and Figure 7 | [Analysis, plotting and coordinate selection](docs/REPRESENTATION_ANALYSIS.md) |

### Methods

Each method directory contains its adapter and experiment configurations. For example, the [Context Prediction pre-training configuration](methods/context_prediction/configs/pretrain.yaml) selects its original-setting recipe. Use the [ASIS protocol](docs/initial_protocols/INITIAL_STEP1_v1.md) to identify the evaluation settings.

## Reading the results

A completed training run, an archived score, and a portable rerun are distinguished throughout the guides. For Figure 3, all 35 LoRA/DPT target cells are available for comparison. The recorded audit covers 20 complete three-seed dense cells; five matched both printed statistics. SSv2 attentive probing uses model-specific reader inputs and blocks. The original mean for VGGT-Omega could not be independently checked across all seeds.

For Figure 5, SpaceNet 2 uses background/building mean IoU and SUN RGB-D uses 37-class semantic mIoU with a learned RGB-depth-validity stem. All twenty original seed-0 LP/AP scores for these two datasets match the source table at printed precision. Portable input and evaluation validation for both datasets remains open. The Figure 7 reference-free result uses nested selection over twelve settings; every selected fold used three or four coordinates.

## Files cited in Appendix E

Use this table to open a filename cited in the supplement. The Evaluation protocols table above links the current Basic Five specifications and recorded model-specific settings.

| Appendix section | Companion files |
| --- | --- |
| E.2 · Initial comparisons | [INITIAL_STEP1_v1.md](docs/initial_protocols/INITIAL_STEP1_v1.md) · [INITIAL_STEP2_v1.md](docs/initial_protocols/INITIAL_STEP2_v1.md) |
| E.3 · Initial comparisons by family | [INITIAL_STEP3_3DFM_v1.md](docs/initial_protocols/INITIAL_STEP3_3DFM_v1.md) · [INITIAL_STEP3_4DFM_v1.md](docs/initial_protocols/INITIAL_STEP3_4DFM_v1.md) · [INITIAL_STEP3_VISION_GENERATIVE_v1.md](docs/initial_protocols/INITIAL_STEP3_VISION_GENERATIVE_v1.md) · [INITIAL_STEP3_VIDEO_SSL_v1.md](docs/initial_protocols/INITIAL_STEP3_VIDEO_SSL_v1.md) · [INITIAL_STEP3_VIDEO_WORLD_MODELS_v1.md](docs/initial_protocols/INITIAL_STEP3_VIDEO_WORLD_MODELS_v1.md) · [INITIAL_STEP3_VLM_v1.md](docs/initial_protocols/INITIAL_STEP3_VLM_v1.md) |
| E.4 · Basic Five baseline filenames | [BASIC5_LINEAR.md](docs/submission_protocols/BASIC5_LINEAR.md) · [BASIC5_ATTENTIVE.md](docs/submission_protocols/BASIC5_ATTENTIVE.md) · [BASIC5_FINETUNE.md](docs/submission_protocols/BASIC5_FINETUNE.md) |
| E.4 · LoRA and DPT | [BASIC5_LORA_v2.md](docs/current_protocols/BASIC5_LORA_v2.md) · [BASIC5_LORA_v2.json](docs/current_protocols/BASIC5_LORA_v2.json) · [BASIC5_DPT_v2.md](docs/current_protocols/BASIC5_DPT_v2.md) · [BASIC5_DPT_v2.json](docs/current_protocols/BASIC5_DPT_v2.json) |
| E.5 · Extended LP | [EXTEND_LINEAR.md](docs/submission_protocols/EXTEND_LINEAR.md) · [EXTEND_LINEAR_v1.json](docs/submission_protocols/EXTEND_LINEAR_v1.json) |
| E.5 · Extended AP | [EXTEND_ATTENTIVE.md](docs/submission_protocols/EXTEND_ATTENTIVE.md) · [EXTEND_ATTENTIVE_v1.json](docs/submission_protocols/EXTEND_ATTENTIVE_v1.json) |
| E.5 · Extended FT | [EXTEND_FINETUNE.md](docs/submission_protocols/EXTEND_FINETUNE.md) · [EXTEND_FINETUNE_v1.json](docs/submission_protocols/EXTEND_FINETUNE_v1.json) |
| E.6 · Frontier-system prompts | [BASIC5_FRONTIER_SYSTEM_PROMPTS.md](docs/submission_protocols/BASIC5_FRONTIER_SYSTEM_PROMPTS.md) |

## Data, dependencies and notices

Model weights and datasets are external inputs. For the required upstream dependency locations and revisions, see [upstream setup](docs/THIRD_PARTY_LICENSES.md). The DPT probe source and its notices are included. Some historical method paths require additional upstream code; the modified MAR dependency is outside this anonymous distribution. The [root license](LICENSE) covers the authors' contributions. Third-party components retain their respective notices.
