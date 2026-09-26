# Manuscript terminology and legacy labels

The manuscript organizes its comparisons by scientific question, not numbered
implementation steps. Use the terminology below when reading this package.
This guide supersedes numbered-step wording for reviewer navigation; it does
not change historical records, executable recipes, checkpoints or scores.

| Legacy internal label | Corresponding manuscript context | Important boundary |
| --- | --- | --- |
| Step 1 / step1 | Original-setting evaluation (ASIS), Section 3; Appendix E.2 | Released weights or reproduced original architectures and learning procedures. The legacy VSSL files cover only part of the section. ASIS does not mean download-only. |
| Step 2 / step2 | CTRL: Revisiting VSSL under Controlled Conditions, Section 5.1; Appendix E.2 | Controlled pre-training with ViT-B/16 on unlabeled ImageNet-1K, 300 epochs and checkpoints at 100/200/300. This is not the unified downstream LP/AP/FT protocol. |
| Step 3 / step3 | Category-specific comparisons in Section 3 and model/component records in Appendices B and C | Includes geometric, generative and Video SSL models, plus language-aligned and multimodal reference systems. Some families reappear as representatives in Section 4; the specific checkpoint, component and protocol determine the correspondence. |
| Step 4 / step4 | Learning-design diagnostic experiments in Section 5.2 and related appendix records | In this package the label appears around Instance Discrimination and DINOv3 components. It is not a one-to-one mapping to every experiment in Section 5.2, nor evidence that a component reproduced a reported result. |

## Separate comparison axes

Section 4, *Present: Unified and Extended Evaluation*, compares selected
representative models under downstream protocols. Appendix E defines LP
(frozen-feature probing), AP (attentive probing), and FT (fine-tuning).
These are not replacement names for the four legacy steps. In particular,
`BASIC5_FAIR_v1` is a legacy downstream-probing identifier, not CTRL pre-training.
LP names an evaluation track; detection and dense task heads need not be linear
classifiers. Historical category-specific probes retain their own metric,
resolution, feature readout and seed conventions.

Section 5.2 also discusses scaling and a frontier-model reference. The
[frontier prompts](submission_protocols/BASIC5_FRONTIER_SYSTEM_PROMPTS.md)
relate to that reference and Appendix C.14, rather than a numbered code stage.
Section 6 uses representation features from ImageNet-1K validation for its map;
a path containing `step1` identifies a legacy artifact group, not the manuscript
section in which that artifact is analyzed.

## Reading retained names safely

Executable identifiers, checkpoint keys, filenames such as `INITIAL_STEP2_v1.md`,
and historical code comments are retained for compatibility and provenance.
A generic training step, optimizer step or adapter stage (`pretrain`,
`linear_eval`) is a computational operation, not this manuscript crosswalk.
The [initial protocol index](initial_protocols/README.md) and
[supplied protocol index](submission_protocols/README.md) preserve the separate
historical and unified specifications. Reconstructed proposals must not be
retroactively substituted for measured settings. Match each claim through its
checkpoint/component and protocol to Appendices B, C and E; the old label alone
cannot establish that match. Unresolved recipe differences remain unresolved.

This is a documentation-only terminology update. It adds no executable support,
changes no experimental result, and does not resolve previously identified
feature-extraction numerical issues. Source availability, porting and score
verification remain distinct as explained in the [scope guide](SUBMISSION_SCOPE.md).
