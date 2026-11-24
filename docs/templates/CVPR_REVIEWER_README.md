# Supplementary code for review

Run commands from the extracted `code/` directory. The package contains evaluation and training entry points, model adapters, protocol references, figure analysis, and focused tests. Supply your own datasets and appropriately licensed checkpoints for training. The quick checks below use local fixtures.

## Evaluation protocols

The following are the primary Basic Five protocols for the experiments in this paper. Each row links to the full specification in Markdown and JSON. LP, AP and FT use v1. The LoRA and DPT v2 documents describe the executed settings reconstructed from the training code and run records. Results use seeds 0, 1 and 2 and the final scheduled epoch.

| Method | Protocol | Implementation and adopted run details |
| --- | --- | --- |
| Linear probing (LP) | [Markdown](docs/current_protocols/BASIC5_LINEAR_v1.md) · [JSON](docs/current_protocols/BASIC5_LINEAR_v1.json) | Frozen encoder and task head. See the [optimization and evaluation audit](docs/BASIC5_V1_AUDIT.md#what-was-compared). |
| Attentive probing (AP) | [Markdown](docs/current_protocols/BASIC5_ATTENTIVE_v1.md) · [JSON](docs/current_protocols/BASIC5_ATTENTIVE_v1.json) | For Figure 3 SSv2, see the [adopted reader and input tokens for each encoder](docs/SSV2_EXECUTION.md#figure-3-attentive-probes). |
| Full fine-tuning (FT) | [Markdown](docs/current_protocols/BASIC5_FINETUNE_v1.md) · [JSON](docs/current_protocols/BASIC5_FINETUNE_v1.json) | End-to-end training. The [execution audit](docs/BASIC5_V1_AUDIT.md#what-was-compared) records deviations, including ImageNet label smoothing. |
| LoRA | [Markdown](docs/current_protocols/BASIC5_LORA_v2.md) · [JSON](docs/current_protocols/BASIC5_LORA_v2.json) | [Task-specific schedules](docs/CVPR_ADAPTATION.md#revised-image-and-semantic-task-cli) and [DINOv3 shorter runs](docs/CVPR_ADAPTATION.md#author-supplied-november-v2-run-descriptions). |
| DPT dense probing | [Markdown](docs/current_protocols/BASIC5_DPT_v2.md) · [JSON](docs/current_protocols/BASIC5_DPT_v2.json) | [Four-tap decoder](docs/CVPR_ADAPTATION.md#training-facing-checks-and-four-tap-dense-probe) for ADE20K and NYUv2. For NYUv2 the paper uses unaligned RMSE. |

For the 40 Extended transfer settings, see the [Figure 5 dataset and metric ledger](docs/CVPR_EXTENDED_COVERAGE.md). The executed semantic metrics for SpaceNet 2 and SUN RGB-D are documented there.

## Figures and analyses

| Paper experiment | Where to start |
| --- | --- |
| Figure 3, five encoders and Basic Five tasks | [LoRA and DPT training](docs/CVPR_ADAPTATION.md), [three-seed comparisons](docs/LORA_FULL_REPRODUCTION.md), [SSv2 attentive probes](docs/SSV2_EXECUTION.md#figure-3-attentive-probes) |
| Figure 5, 40 transfer settings | [Dataset and metric ledger](docs/CVPR_EXTENDED_COVERAGE.md), [LP/AP comparisons](docs/TRANSFER_COMPARISON.md) |
| Representation geometry and Figure 7 | [Analysis, plotting and coordinate selection](docs/REPRESENTATION_ANALYSIS.md) |

## Quick verification

Use Python 3.12 in a clean environment. The shared downstream dependencies are listed in [the requirements file](downstream/requirements.txt). The [hashed CPU lock](downstream/requirements.lock.txt) targets Linux x86_64; on other platforms use the requirements file with compatible PyTorch wheels. Model-specific environments are under `methods/`.

```sh
python -m pip install --require-hashes --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r downstream/requirements.lock.txt
python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe
python -m downstream.revised_tasks --help
python -m downstream.revised_reporting --help
python -m downstream.representation_analysis --help
```

## Reading the results

A completed training run, an archived score, and a portable rerun are distinguished throughout the guides. For Figure 3, all 35 LoRA/DPT target cells are available for comparison. The recorded audit covers 20 complete three-seed dense cells; five matched both printed statistics. SSv2 attentive probing uses model-specific reader inputs and blocks. The original mean for VGGT-Omega could not be independently checked across all seeds.

For Figure 5, SpaceNet 2 uses background/building mean IoU and SUN RGB-D uses 37-class semantic mIoU with a learned RGB-depth-validity stem. All twenty original seed-0 LP/AP scores for these two datasets match the source table at printed precision. Portable input and evaluation validation for both datasets remains open. The Figure 7 reference-free result uses nested selection over twelve settings; every selected fold used three or four coordinates.

## Data, dependencies and notices

Model weights and datasets are external inputs. For the required upstream dependency locations and revisions, see [upstream setup](docs/THIRD_PARTY_LICENSES.md). The DPT probe source and its notices are included. Some historical method paths require additional upstream code; the modified MAR dependency is outside this anonymous distribution. The [root license](LICENSE) covers the authors' contributions. Third-party components retain their respective notices.
