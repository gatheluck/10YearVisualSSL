# Revisiting Vision Model Baselines in the Deep Learning Era

Which advances in visual representation learning transfer across tasks, and how do the conclusions change when models share a training or evaluation protocol? This repository contains the training recipes, evaluation code and analysis tools that accompany the study.

**Paper** (arXiv forthcoming) · [Project Page](https://gatheluck.github.io/10YearVisualSSL/) · [Citation](#citation) · [![Tests on main](https://github.com/gatheluck/10YearVisualSSL/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/gatheluck/10YearVisualSSL/actions/workflows/tests.yml?query=branch%3Amain)

## Find a figure or table

Section 3 defines three evaluation settings. **ASIS** evaluates released or reproduced models with family-specific protocols (Section 4). **CTRL** compares 32 visual self-supervised learning (VSSL) methods under controlled pre-training (Section 5). **Unified** evaluates selected checkpoints under shared downstream protocols (Section 6).

Explore the [interactive rankings](https://gatheluck.github.io/10YearVisualSSL/leaderboard.html#leaderboard). For each figure, the instructions identify the inputs, the checks performed and any unresolved differences.

| Figure | Question and companion tables | Instructions | Code or protocol |
| --- | --- | --- | --- |
| Figure 1 | Which configurations rank well across five tasks? | [Five-task ranks](docs/PAPER_REPRODUCTION_GUIDE.md#figure-1--five-task-ranks) | [Historical VSSL protocols (ASIS subset)](docs/initial_protocols/INITIAL_STEP1_v1.md) |
| Figure 2 | How has historical transfer changed? Tables 10, 12–18 | [Historical transfer](docs/PAPER_REPRODUCTION_GUIDE.md#figure-2--historical-transfer) | [Native detection](downstream/native_detection.py) |
| Figure 3 | How do readouts on frozen backbones (LP, AP and DPT) and LoRA compare? Tables 22–26 | [Unified readout and adaptation](docs/PAPER_REPRODUCTION_GUIDE.md#figure-3--unified-readout-and-adaptation) | [LoRA configurations](downstream/figure3_lora_configs.py) |
| Figure 4 | How do scores change between ASIS and CTRL when architecture, pre-training and downstream evaluation differ? Tables 19–21 | [Controlled pre-training](docs/PAPER_REPRODUCTION_GUIDE.md#figure-4--controlled-pre-training) | [Paired comparison](downstream/controlled_comparison.py) |
| Figure 5 | How do models transfer across 40 settings? Tables 27–28 | [Extended transfer](docs/PAPER_REPRODUCTION_GUIDE.md#figure-5--extended-transfer) | [Transfer comparison](downstream/transfer_comparison.py) |
| Figure 6 | How do frontier systems compare on matched examples? Table 29 | [Frontier matched subsets](docs/PAPER_REPRODUCTION_GUIDE.md#figure-6--frontier-matched-subsets) | [Prediction scorer](downstream/frontier.py) |
| Figure 7 | How does representation geometry relate to task scores? Tables 31–32 | [Representation geometry](docs/PAPER_REPRODUCTION_GUIDE.md#figure-7--representation-geometry) | [Nested analysis](downstream/representation_nested.py) |

For Table 30 (model scaling) and the supplement's separately numbered figures, use the [supplement reading order](docs/PAPER_REPRODUCTION_GUIDE.md#supplement-reading-order).

## Quick verification

Generate the Figure 3 LoRA training configurations from the repository root with Python 3.12. This command uses the standard library and example paths. Choose a new output directory.

```sh
python3 -m downstream.figure3_lora_configs --paths configs/figure3-lora-paths.example.json --out figure3-configs
```

The command finishes silently and writes **45 JSON files** in `figure3-configs/`, one for each of three tasks, five encoders and three seeds. Open one to inspect the dataset path, encoder and seed. For training, supply real paths and follow the [LoRA run instructions](docs/LORA_FULL_REPRODUCTION.md). To exercise the evaluators on local fixtures, follow the [environment setup and smoke tests](OPERATING_GUIDE.md#quick-verification).

## Methods

Use the [method index](OPERATING_GUIDE.md#methods) to select an adapter and see which training or evaluation operations it supports. The operating guide also covers [resolving a configuration](OPERATING_GUIDE.md#reproducibility-the-resolved-config), [training](OPERATING_GUIDE.md#training-a-method) and [checking the outputs](OPERATING_GUIDE.md#checking-an-adapters-output-against-the-contract).

## Evaluation protocols

The following are the primary Basic Five protocols for the experiments in this paper. Each row links to the full specification in Markdown and JSON. LP, AP and FT use v1. The LoRA and DPT v2 documents describe the executed settings reconstructed from the training code and run records. The scheduled seeds are 0, 1 and 2. See [COCO metrics and completed FT repeats](docs/BASIC5_V1_AUDIT.md#coco-evaluation-and-completed-repeats) for the reported populations and final-epoch results. In Table 26, 18 of 25 full fine-tuning entries have three runs, VGGT-Omega on SSv2 has one run, and six entries are blank.

| Method | Protocol | Implementation and adopted run details |
| --- | --- | --- |
| Linear probing (LP) | [Markdown](docs/current_protocols/BASIC5_LINEAR_v1.md) · [JSON](docs/current_protocols/BASIC5_LINEAR_v1.json) | Frozen encoder with a trained task-specific head, including Faster R-CNN for COCO object detection. See the [optimization and evaluation audit](docs/BASIC5_V1_AUDIT.md#what-was-compared). |
| Attentive probing (AP) | [Markdown](docs/current_protocols/BASIC5_ATTENTIVE_v1.md) · [JSON](docs/current_protocols/BASIC5_ATTENTIVE_v1.json) | For Figure 3 SSv2, see the [adopted reader and input tokens for each encoder](docs/SSV2_EXECUTION.md#figure-3-attentive-probes). |
| Full fine-tuning (FT) | [Markdown](docs/current_protocols/BASIC5_FINETUNE_v1.md) · [JSON](docs/current_protocols/BASIC5_FINETUNE_v1.json) | End-to-end training. The [execution audit](docs/BASIC5_V1_AUDIT.md#what-was-compared) records deviations, including ImageNet label smoothing. |
| LoRA | [Markdown](docs/current_protocols/BASIC5_LORA_v2.md) · [JSON](docs/current_protocols/BASIC5_LORA_v2.json) | [Task-specific schedules](docs/CVPR_ADAPTATION.md#revised-image-and-semantic-task-cli) and [DINOv3 shorter runs](docs/CVPR_ADAPTATION.md#author-supplied-november-v2-run-descriptions). |
| DPT dense probing | [Markdown](docs/current_protocols/BASIC5_DPT_v2.md) · [JSON](docs/current_protocols/BASIC5_DPT_v2.json) | [Four-tap decoder](docs/CVPR_ADAPTATION.md#training-facing-checks-and-four-tap-dense-probe) for ADE20K and NYUv2. For NYUv2 the paper uses unaligned RMSE. |

For the 40 Extended transfer settings, see the [Figure 5 dataset and metric ledger](docs/CVPR_EXTENDED_COVERAGE.md). The executed semantic metrics for SpaceNet 2 and SUN RGB-D are documented there.

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


## Reproduction scope

The [figure guide](docs/PAPER_REPRODUCTION_GUIDE.md) distinguishes three starting points. Original run records can recover reported aggregates. Saved predictions can exercise a scorer, while checkpoint replay also exercises inference. Independent training starts with a new run and requires the corresponding data, model and recipe. Each figure's guide identifies which comparisons have been verified.

Use the [package scope](docs/SUBMISSION_SCOPE.md) and [manuscript terminology](docs/PAPER_TERMINOLOGY.md) to match ASIS, CTRL and Unified evaluation. See the [downstream task guide](docs/DOWNSTREAM.md) for execution and the Extended guides for [image classification](docs/EXTENDED_CLASSIFICATION.md), [semantic segmentation](docs/EXTENDED_SEGMENTATION.md) and [video](docs/EXTENDED_VIDEO.md).

## Citation

Hirokatsu Kataoka, Kohsuke Ide, Yoshihiro Fukuhara, Shumpei Takezaki, Nakamasa Inoue, Yuki M Asano and Christian Rupprecht. *Revisiting Vision Model Baselines in the Deep Learning Era*. Manuscript, 2026. [BibTeX](https://gatheluck.github.io/10YearVisualSSL/#citation)

## License

The authors provide their contributions under the MIT terms in [LICENSE](LICENSE). Upstream implementations, checkpoints and datasets carry their own terms. Consult the [third-party license inventory](docs/THIRD_PARTY_LICENSES.md) for provenance, notices and outstanding rights questions.
