# INITIAL_STEP3_VISION_GENERATIVE_v1: historical evidence and candidate reconstruction

This is an edited, anonymized companion to the initial experiments, separate
from the later Unified LP/AP/FT specifications. It is not a certification that
all manuscript results follow a single recipe. Original experimental code and
this portable package's integration/validation are separate matters.

## Historical evidence and its limits

The inspected RAEv2 wrapper defaults to K=7 and sums the last seven CLS features. The supplied reconstruction attributes initial completed results to ImageNet probes and distinguishes generator-internal features from semantic encoder features.

The following rows are historical attributions in the supplied reconstruction,
not independent confirmation of every run:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| RAE encoder | `dinov2_vitl14` / `facebook/dinov2-large` | Default in `build_rae_backbone` and `qsub_imagenet1k_multinode.sh` | Observed main-run setting |
| Epochs and head batch | 100 epochs, head batch 1024, SGD Nesterov, WD 0, warmup 10 | `GenSSL/_common/linear_probe.py` `LinearProbeConfig` | Observed main-run setting |
| Tasks | ImageNet-1k only | Completed `results.json` files are ImageNet-1k and ImageNet-100. ImageNet-100 is a pilot | Observed main-run setting |

The supplied evidence table below distinguishes source claims from newly chosen
settings. Unless explicitly described above as inspected, those historical claims
are attributed to the supplied reconstruction and have not been independently
verified for every run. No score or completion claim is newly established here.

## Proposed changes and unresolved choices

Replacing K=7 aggregation with final-layer CLS is a new representation proposal requiring a new probe. Removing the LR sweep is also a new choice. The reconstruction does not establish that these candidate reruns were performed.

These rows describe new choices, mixed evidence or scope boundaries; do not
interpret them as a uniform completed historical campaign:

| Setting | Attributed value | Supplied evidence | Source classification |
|---|---|---|---|
| RAEv2 stage | Final class token, not the sum of the last 7 | Stored file `results/raev2_imagenet1k_K7/results.json` matches `forward_features` summing last K=7 class tokens, width 1024 | Newly selected unification |
| Learning rate | `0.01` at head batch 1024 | `rae_imagenet1k` and `raev2_imagenet1k_K7` both store `best_lr: 0.01`. The driver swept seven rates | Newly selected unification (sweep removed) |
| DiT and JiT | Excluded | DiT `feat_dim` 1152 at `t=0`; JiT README fallback | Compatibility gap |

Candidate settings and execution instructions below are not evidence of completed runs.
They must not retroactively redefine the conditions of published result cells.
No experiment, seed assignment, preprocessing or model implementation is changed
by including this document.

## Supplied candidate specification (attributed reconstruction)

The following detailed reconstruction is retained for review, including its
own evidence/decision table. Its imperative language describes a candidate
procedure, not a command to execute or a claim that each setting was used.
New unifications remain proposals unless matched to a completed run. Statements
about stored files are source attributions, not a fresh audit of those files.

Internal paths and job identifiers have been removed. `${LOCAL_REFERENCE_PATH}`
is an intentionally unresolved placeholder, not a runnable dataset path.
Relative source paths name the original experimental archive; their presence
here does not claim those scripts are delivered or runnable in this package.
Public upstream model names identify comparison methods, not submission authors.

This document is a candidate specification reconstructed from the initial campaign under `methods_step3/GenSSL/`. It is not a claim that every stored score already follows this file. The stored RAEv2 ImageNet-1k file sums the last 7 class tokens. This protocol uses the final-layer class token only. DiT timestep features and JiT fallback features are a different representation and do not enter this table.

### Purpose and category scope

Evaluate the semantic encoder of a vision generative model. The evaluated tensor is that encoder’s final class token. It is not a VAE latent, not a diffusion denoiser state, and not a timestep-conditioned hidden vector.

The completed initial results are ImageNet linear probes. ImageNet-100 jobs are pilots. ADE20K, NYUv2, COCO, and SSv2 launchers exist and did not leave a completed `results.json` in this campaign, so they are not tasks in this protocol.

### Included models and datasets

| Directory | What satisfies the interface | Published encoder | Stored ImageNet-1k width |
|---|---|---|---|
| `03_rae` | Yes. Final class token | `facebook/dinov2-large` (`dinov2_vitl14`) | 1024 |
| `05_raev2` | Yes, after the reconstruction below | DINOv2-L/14 via `facebookresearch/dinov2` (`dinov2_vitl14`), K recorded as 7 in the historical file | 1024 |
| `01_mage` | Yes. Encoder class token of the MAGE ViT | MAGE ViT-B encoder, checkpoint `mage-vitb-1600ep.pth` | 768 |

| Gap | Why it is outside this result definition |
|---|---|
| `02_dit` | The completed probe is DiT-XL/2 hidden states, global average at timestep 0, width 1152 (`02_dit` README and `feat_dim` 1152). That is a denoiser state. |
| `04_jit` | The backbone file is a placeholder with an HART fallback, not the RAE semantic-encoder contract. |

Dataset: ILSVRC2012 ImageNet-1k, official train and validation. The GenSSL loader root is the path in `GenSSL/_common/data_imagenet.py` (the campaign also documents `${LOCAL_REFERENCE_PATH}`).

### Global rules

- Load the released semantic-encoder checkpoint. Freeze it. `eval()` mode. No encoder gradient.
- Use the encoder’s published square resolution and its published channel normalization. Record both numbers in the result. DINOv2-L/14 uses 224 and ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- One seed, `0`.
- One learning rate. Do not sweep.
- The candidate score is the best validation Top-1 inside the 100 scheduled epochs. Report that epoch.

### Common representation interface

- **Checkpoint.** RAE v1: `facebook/dinov2-large`. RAE v2: the DINOv2-L/14 hub checkpoint used by `05_raev2/backbone/raev2_backbone.py`. MAGE: the ViT-B encoder weights, not the decoder.
- **Included components.** Patch embedding, blocks, and the final encoder norm. Exclude the diffusion transformer, the VAE, the REPA projector, and the MAGE decoder.
- **Frozen scope.** The whole encoder.
- **Feature stage.** Final block, class token at index 0, after the final norm. Do not sum earlier blocks.
- **Tensor interface.** A single global vector `(B, D)`. This protocol has no dense task, so a spatial grid is not required.
- **Width.** `D` is the encoder width. Record it. Do not pad or truncate.
- **Special tokens.** The class token is the feature. Patch tokens and register tokens are discarded.
- **Pooling.** No second pool. The class token is already global.
- **Normalization.** The encoder’s published normalization, once. No L2 normalization before the linear head.
- **Head boundary.** `Linear(D, 1000)` is the first trainable layer.

RAEv2’s historical `forward_features` sums class tokens from the last `K=7` layers (`raev2_backbone.py`). That sum is not this interface. Recompute the probe from the final class token.

### candidate task summary

| Task | Input | Head | Optimizer | Schedule | Score epoch | Primary metric |
|---|---|---|---|---|---|---|
| ImageNet-1k | encoder’s published square size | linear, 1000-way | SGD, Nesterov | 100 epochs | best validation Top-1 | Top-1 (%) |

### Detailed task recipes

#### ImageNet-1k

- **Driver.** `GenSSL/_common/linear_probe.py`.
- **Split.** Official train for the head, official validation for the score.
- **Input.** Square resize to the encoder’s published resolution. Published channel normalization. Random resized crop and horizontal flip on the train extraction; center crop on the validation extraction, using the transforms in `GenSSL/_common/data_imagenet.py`.
- **Feature.** Final class token, extracted once and cached.
- **Head.** `Linear(D, 1000)`, weight std `0.01`, zero bias. Head batch 1024.
- **Loss.** Cross-entropy. Label smoothing `0`.
- **Optimizer.** SGD, momentum `0.9`, Nesterov enabled, weight decay `0`.
- **Learning rate.** `0.01`, absolute, at head batch 1024.
- **Schedule.** 100 epochs. Linear warmup for 10 epochs, then cosine (`WarmupCosineLR`).
- **Augmentation.** The ImageNet train crop and flip used during feature extraction only. The head sees frozen features.
- **Checkpoint rule.** Highest validation Top-1 during the 100 epochs.
- **Metrics.** Top-1 (%) primary, Top-5 (%) at that epoch secondary.

### Execution and seed policy

Launch `03_rae/scripts/qsub_imagenet1k_multinode.sh` with encoder `dinov2_vitl14`, and the RAEv2 ImageNet-1k probe against `raev2_backbone.py` modified only so that the feature is the final class token. MAGE uses `01_mage`’s ImageNet-1k linear entry point with the encoder class token. Seed `0`. One trial.

Do not launch `eval_generator_downstream.py`. That script reads generator-internal layers (`k7_tokenizer`, `gen_middle`, decoder states) and is a later generator study.

### Evaluation and reporting

One ImageNet-1k table. Columns: checkpoint id, resolution, normalization, `D`, learning rate `0.01`, epoch of the best Top-1, Top-1 (%), Top-5 (%). Do not place a DiT `t=0` number or a K=7 sum in the same column.

### reconstruction decisions

| Setting | candidate value | Evidence | Kind |
|---|---|---|---|
| Evaluated component | Semantic-encoder final class token | `03_rae/backbone/rae_backbone.py` `feature_key=cls`; GenSSL README “ONLY use the pretrained encoder” | Observed RAE main-run setting |
| RAE encoder | `dinov2_vitl14` / `facebook/dinov2-large` | Default in `build_rae_backbone` and `qsub_imagenet1k_multinode.sh` | Observed main-run setting |
| RAEv2 stage | Final class token, not the sum of the last 7 | Stored file `results/raev2_imagenet1k_K7/results.json` matches `forward_features` summing last K=7 class tokens, width 1024 | Newly selected unification |
| Learning rate | `0.01` at head batch 1024 | `rae_imagenet1k` and `raev2_imagenet1k_K7` both store `best_lr: 0.01`. The driver swept seven rates | Newly selected unification (sweep removed) |
| Epochs and head batch | 100 epochs, head batch 1024, SGD Nesterov, WD 0, warmup 10 | `GenSSL/_common/linear_probe.py` `LinearProbeConfig` | Observed main-run setting |
| Tasks | ImageNet-1k only | Completed `results.json` files are ImageNet-1k and ImageNet-100. ImageNet-100 is a pilot | Observed main-run setting |
| DiT and JiT | Excluded | DiT `feat_dim` 1152 at `t=0`; JiT README fallback | Compatibility gap |

### Evidence references

- `methods_step3/GenSSL/README.md`
- `methods_step3/GenSSL/_common/linear_probe.py`
- `methods_step3/GenSSL/03_rae/backbone/rae_backbone.py`
- `methods_step3/GenSSL/03_rae/scripts/qsub_imagenet1k_multinode.sh`
- `methods_step3/GenSSL/05_raev2/backbone/raev2_backbone.py`
- `methods_step3/GenSSL/05_raev2/backbone/raev2_generator_backbone.py` (later generator extractor; not this protocol)
- `methods_step3/GenSSL/03_rae/results/rae_imagenet1k/results.json`
- `methods_step3/GenSSL/05_raev2/results/raev2_imagenet1k_K7/results.json`
