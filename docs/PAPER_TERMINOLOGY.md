# Manuscript terminology and legacy labels

Section 3, *Models and Evaluation Framework*, introduces three evaluation settings.
The numbered labels found in earlier recipes are repository identifiers. To find
an experiment in the current manuscript, start with its setting and figure.

| Manuscript section | What is compared |
| --- | --- |
| Section 4: Historical Transfer of Vision Models (ASIS) | Released or reproduced models evaluated on shared benchmarks using family-specific features and downstream protocols. Figure 2 shows 113 configurations. Figure 1 ranks 109 on the Basic Five. |
| Section 5: Controlled VSSL Pre-training (CTRL) | A comparison of 32 methods trained on unlabeled ImageNet-1K with ViT-B/16 for 300 epochs. Figure 4 compares ASIS with CTRL; changing the architecture, pre-training and downstream recipe can all affect the result. |
| Section 6: Unified and Extended Evaluation | Five representative checkpoints evaluated on shared tasks, splits and metrics, with model-specific feature interfaces. Figure 3 covers Basic Five readouts and adaptation; Figure 5 covers 40 extended settings. Figure 6 is a frontier-system reference on matched subsets with adapted task formulations. |
| Section 7: The Basic Five in a Representation Plane | Figure 7 relates task performance to image- and patch-level agreement. Its planes and held-out prediction analyses use frozen representations. |

## Reading older identifiers

| Earlier repository label | How to interpret it |
| --- | --- |
| Step 1 / step1 | For the initial VSSL comparisons, the original-setting recipes contribute to ASIS. Check the method: the adapter can train a model, extract features or evaluate released weights. |
| Step 2 / step2 | The initial controlled VSSL recipes are relevant to CTRL. Check the method and run against the 32-method comparison; additional adapter training options also exist. |
| Step 3 / step3 | Family-specific configurations and component comparisons, including geometric, generative, video and language-aligned models. Some models also appear in Unified with a different readout or checkpoint. |
| Step 4 / step4 | Repository diagnostics for components such as Instance Discrimination objectives and DINOv3 training components. There is not a one-to-one correspondence with a manuscript section or figure. |

Use the [figure guide](PAPER_REPRODUCTION_GUIDE.md) for the input, metric and verification
status of each comparison, and the [package scope](SUBMISSION_SCOPE.md) for the limits of public reproduction. LP (linear probing), AP (attentive probing), DPT (dense
prediction transformer probing), LoRA (low-rank adaptation) and FT (full fine-tuning)
describe downstream protocols. The trained task head used with a frozen backbone in an LP track depends on the task;
for COCO detection use its documented bounding-box evaluator.

Appendix E and the [initial](initial_protocols/README.md) and
[supplementary protocol indices](submission_protocols/README.md) preserve names
such as `INITIAL_STEP2_v1.md`. Keep these exact names when opening a file or using
a saved configuration. An optimizer step counts parameter updates; the adapter
stages `pretrain` and `linear_eval` select executable operations.
