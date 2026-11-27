# Reproducing the submitted CVPR figures and supplement

The [submission scope](SUBMISSION_SCOPE.md) describes the experiments and inputs covered by the portable package. Choose the input you have. **Original run records** let you recompute a printed mean, sample standard deviation (SD) or ranking. **Saved weights** let you run inference again with the adopted encoder, trained task head, preprocessing and evaluation split. Saved predictions allow a shorter check that starts at the scorer. **Independent training** starts a new run with the public implementation and compares its final scores across the reported seeds. The November 25 audit below used existing records unless a checkpoint evaluation or training run is identified explicitly.

Run the examples from the code root in the [downstream environment](../README.md#quick-verification). Filenames in the commands stand for inputs you supply. Follow the linked formats and choose fresh output paths. For a local check with fixture inputs and no model downloads, run:

```sh
python -m unittest tests.test_transfer_comparison tests.test_frontier
```

## Figure 1 — Five-task ranks

Start with Supplement C.1 and Appendix D Figure 1. Use the 109 representative configurations, omitting DINOv3 ViT-B/16, MAGE, JiT and Wan2.2 from the 113 in Figure 2. Compare each task's printed ranks with the plotting table before averaging the five ranks per configuration. All 545 ranks and 109 means agreed. Preserve the adopted COCO tie at rank 97, which includes HighwayNet and twelve printed zeros. No general rank recalculation from rounded scores or missing-value substitution was verified.

The checkout provides [ASIS protocols](initial_protocols/INITIAL_STEP1_v1.md) and [model-specific execution guides](../README.md#methods). The adopted rank table and an end-to-end Figure 1 plotting command are outside this checkout. Checkpoint replay and public retraining of all 109 configurations remain unverified.

## Figure 2 — Historical transfer

Use the 113 **original-setting (ASIS)** configurations and their own readers, checkpoints and task recipes. Supplement C.1 identifies the rows. Of 565 plotted score slots, 563 agreed with the printed values or blanks in Tables 10 and 12–18. HighwayNet has numeric ImageNet and COCO entries in Table 12 and blanks in the plotting data. AlayaWorld's recorded NYUv2 value was excluded from the graph.

For a native COCO run, use the [bounding-box evaluation path](BASIC5_NATIVE_PATHS.md). Independent scoring of saved boxes on all 5,000 val2017 images recovered all 12 saved statistics for 15 targets. Eleven targets matched adopted supplement values. MAGE and JiT have no corresponding Table 15 value, and the 224-square DINOv3-7B and V-JEPA 2.1-L runs represent different configurations. This check starts after model inference. Full-panel checkpoint evaluation, retraining and a public command that rebuilds the historical plot remain unverified.

## Figure 3 — Unified readout and adaptation

Rebuild the README's ImageNet-1K and SSv2 chart from the [reported LP and AP means](figures/figure3-classification.csv) with `python3 bin/render-readme-figure3.py > docs/figures/figure3-classification.svg` from the repository root.

Figure 3 compares three frozen-backbone readouts (linear probing, attentive probing and DPT) with low-rank adaptation (LoRA). Supplement C.3 Tables 22–26 also report full fine-tuning (FT). Original seed files reproduced 91 of the 104 reported cells. Other contemporaneous per-seed records reproduced 12 more at printed precision. VGGT-Ω COCO LP has only aggregate support. Preserve six unreported FT cells and the single reported VGGT-Ω SSv2 FT seed. For RAEv2 LP/AP, manuscript Section 6 names diffusion-generator block 29 at timestep 0.1. The [run attribution audit](CVPR_ADAPTATION.md) identifies a seven-layer tokenizer readout in the recorded Unified runs. These feature tensors come from different components. Identify the target run before comparing scores. The ASIS detection run used a separate frozen generator readout.

To train or compare, use the [LoRA and DPT run instructions](LORA_FULL_REPRODUCTION.md), the [LP/AP/FT protocols](../README.md#evaluation-protocols) and the [SSv2 encoder-specific AP readers](SSV2_EXECUTION.md#figure-3-attentive-probes). Generate 45 LoRA configs for three tasks, five encoders and three seeds from the [example path mapping](../configs/figure3-lora-paths.example.json):

```sh
python -m downstream.figure3_lora_configs --paths paths.json --out configs-local
```

The linked training instructions give the GPU topology and per-task command. To compare completed public LoRA/DPT runs to the 35 printed targets, supply the final configs, exit statuses and output directories:

```sh
python -m downstream.revised_reporting --config comparison-plan.json --out new-comparison.json
```

The November 24 full-data comparison matched both mean and SD in 1/10 ADE20K cells and 4/10 NYUv2 cells. Fifteen dense cells differed. The November 25 snapshot recorded 15 completed seeds out of 45 planned classification/detection LoRA runs. A seed completion count alone gives no three-seed score comparison. Original record checks cover more cells than independent training. Full-panel replay from historical task weights remains unverified.

## Figure 4 — Controlled pre-training

Use Supplement C.2 Tables 19–21 and the [paired comparison input format](TRANSFER_COMPARISON.md#figure-4-controlled-pre-training). Each of 32 methods has ASIS scores and one controlled (CTRL) trajectory at epochs 100, 200 and 300. The command consumes the four sets of scores:

```sh
python -m downstream.transfer_comparison --figure4 --input comparison-data.json --out controlled-report.json
```

The reported 480 CTRL values agree within printed rounding. For ASIS, Context Prediction COCO bbox AP is 0.30 in Figure 4 and 1.47 in Figure 2 and Table 10. The recovered three-seed final mean is 0.2590 and the mean of per-run best scores is 0.3487. A separate Context Encoder run yielded 1.47. The adopted Context Prediction run remains unresolved. For new training, start with the [CTRL protocol](initial_protocols/INITIAL_STEP2_v1.md) and the applicable method adapter. Complete checkpoint replay and independent training across the panel remain unverified.

## Figure 5 — Extended transfer

Use the [40-setting ledger](CVPR_EXTENDED_COVERAGE.md) to select the task runner, input and metric. Tables 27–28 contain five models under LP and AP. All 400 worksheet entries agree with the supplement. Original records support 399 scores numerically, with multiple candidate runs for some cells. VGGT-Ω Kuzushiji-Kanji LP rounds to 0.30% from the final seed-0 record, while the worksheet and Table 27 give 0.29%. For normalized radial values supply selected LP/AP means in the [comparison format](TRANSFER_COMPARISON.md):

```sh
python -m downstream.transfer_comparison --input selected-means.json --out comparison.json
```

For SpaceNet 2 Vegas and SUN RGB-D, the audit ran native inference from the 20 final seed-0 checkpoints, then used the [public saved-map scorer](EXTENDED_SEMANTIC_INPUTS.md#figure-5-native-semantic-inputs-and-evaluation). To repeat the scoring step, supply predicted class maps, the official split and the documented config:

```sh
python -m downstream.extended_segmentation --config eval.json --out new-result-directory
```

All 20 original scores matched at printed precision. Weight replay matched 10/10 SUN RGB-D cells (37-class semantic mIoU) and 5/10 SpaceNet cells (mean background/building IoU). The largest difference was V-JEPA 2.1-G SpaceNet AP, 65.44 reported and 71.04 replayed. The original scorer gave 71.04 on the same maps. The cause remains unresolved. The public command starts with maps produced by inference. Independent public training of the full 400-cell panel remains unverified.

## Figure 6 — Frontier matched subsets

Use Supplement C.5 Table 29 and E.6. For each task, supply a labelled 500-sample manifest and five ordered batches of 100 predictions in the [Frontier scoring format](FRONTIER_EVALUATION.md):

```sh
python -m downstream.frontier --manifest manifest.json --responses batch1.json batch2.json batch3.json batch4.json batch5.json --out frontier-score.json
```

All 20 plotted values agree with Table 29. The recovered DINO outputs gave 429/500 and 437/500 for ImageNet LP and AP, and 217/500 for SSv2 AP. SSv2 LP gave 77/500 against the printed score implying 78/500. The adopted batch remains unresolved. DINO dense predictions and GPT answers were unavailable in the audited run. COCO here measures category-presence Micro-F1 on the matched subset. Native COCO in Figures 1–4 uses bbox AP. This scorer accepts existing responses. The audit made no new model calls and establishes neither weight replay nor independent training for the matched panel.

## Figure 7 — Representation geometry

Use the [feature, plot and nested-analysis formats](REPRESENTATION_ANALYSIS.md). From the original coordinate and score tables, all 10 pooled Spearman correlations in the Figure 7 agreement analysis matched at four decimals. Each of the five tasks contributes two correlations, one using image agreement alone and one using both image and patch agreement (104 fitted models, 94 with COCO). To start earlier, provide aligned frozen image and patch feature arrays:

```sh
python -m downstream.representation_analysis --config analysis-config.json --out new-analysis
```

To render the plane, supply the family labels in fitted-model order. The plot command writes an SVG, a PNG and a display record:

```sh
python -m downstream.representation_plot --analysis new-analysis/analysis.json --families families.json --out new-plot
```

For the main text's reference-free result, use aligned agreement matrices and scores. For Supplement D.1 Tables 31–32 use image agreement and the 141 candidate patch references. These are distinct inputs:

```sh
python -m downstream.representation_nested --mode reference-free --input pcoa.npz --out pcoa.json
python -m downstream.representation_nested --mode patch-reference --input candidates.npz --out candidates.json
```

The matrix-level checks recovered 0.692/0.738 for ADE20K/NYUv2 in reference-free mode and 0.597/0.595 with patch-reference selection. Nested selection chose three or four coordinates in every reference-free fold, although the text describes two fixed coordinates. End-to-end extraction from the original weights and regeneration of all task scores from public training remain unverified.

## Supplement reading order

| Section | How to use it |
| --- | --- |
| A and B | A covers discussion and limitations. B identifies the models, training and evaluation settings. Match the configuration and checkpoint before selecting a [method adapter](../README.md#methods). |
| C.1–C.5 | Follow Figures 1–6 above for ASIS, CTRL, Unified Basic Five, Extended and Frontier respectively. Keep task definitions, missing cells and reported seed counts with each comparison. |
| C.6, Table 30 | DINOv3 scaling across four sizes. Means and sample SDs from 60 native seed result files match all 36 printed metric cells. See the [linear protocol](current_protocols/BASIC5_LINEAR_v1.md) for the downstream track. A four-size saved-weight replay and independent retraining remain unverified. |
| D, Figures 1–4 | The supplement numbers its own figures. Use main Figure 1 for per-task ranking, original ASIS paired measurements for the pairwise plots, main Figure 4 for CTRL trajectories and C.6 for DINOv3 scaling. The CTRL comparison command reports ranks and changes. A single public command regenerating all four supplement plots remains unverified. |
| D.1, Tables 31–32 | Follow the patch-reference command under main Figure 7 for the nested dense-task results. |
| E.1–E.6 | Start at the [scope and replication notes](submission_protocols/00_scope_and_replication.md) and [index of protocol files](../README.md#files-cited-in-appendix-e). It links ASIS/CTRL, family-specific protocols, Basic Five, Extended registries and Frontier prompts. |
