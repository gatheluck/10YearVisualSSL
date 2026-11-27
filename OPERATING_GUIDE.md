# Operating guide

See the [research overview](README.md), [current evaluation protocols](README.md#evaluation-protocols) and [files cited in Appendix E](README.md#files-cited-in-appendix-e).

## Quick verification

From the repository root, activate a clean Python 3.12 environment. On Linux x86_64, install the shared dependencies from the [hashed CPU lock](downstream/requirements.lock.txt).

```sh
python -m pip install --require-hashes --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r downstream/requirements.lock.txt
```

On other platforms with compatible PyTorch wheels, use `python -m pip install -r downstream/requirements.txt`.

**Source-only checkout.** Run these 19 tests and three help commands. They use local fixtures and need no model weights, datasets or submodule files.

```sh
python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe.DenseProbe.test_missing_submodule_reports_actionable_error
python -m downstream.revised_tasks --help
python -m downstream.revised_reporting --help
python -m downstream.representation_analysis --help
```

Expect 19 tests to pass with no skips. Each help command prints its available options.

**Full check with the DPT implementation.** To run all 23 tests, the checkout must also contain the geosubprobe source at the revision pinned by this repository, including `third_party/geosubprobe/evals/models/probes.py`. Four additional tests construct the dense probe using that code.

From the root of a Git clone at the revision you want to verify, run the following commands. The first fetches geosubprobe if needed and checks out the commit recorded by the parent repository. A source archive lacks the Git metadata for this step. See the [Git submodule documentation](https://git-scm.com/docs/git-submodule).

```sh
git submodule update --init --checkout -- third_party/geosubprobe
python -m unittest tests.test_figure3_comparison tests.test_cvpr_extended_coverage tests.test_transfer_comparison tests.test_dpt_probe
```

Expect 23 tests to pass with no skips.

## Methods

Each row describes the operations available in a method's adapter or downstream provider.
Pre-training learns the encoder; linear evaluation trains a probe on frozen features.
These operations can serve different comparisons. Section 4 evaluates released or
reproduced models with family-specific protocols (ASIS). Section 5 compares 32 VSSL
methods trained from scratch with ViT-B/16 on unlabeled ImageNet-1K for 300 epochs
(CTRL). Section 6 evaluates representative checkpoints with shared downstream
protocols (Unified). The supervised CLIP label-text adaptation is a separate reference.

The table follows the [historical method order](docs/METHOD_NAMING.md).
The Basic Five tasks are ImageNet-1K image classification, COCO object detection,
ADE20K semantic segmentation, NYUv2 depth estimation and SSv2 video classification.
See the [downstream guide](docs/DOWNSTREAM.md) for the task-specific evaluators.

Basic Five components also support provider-owned FT and online ImageNet
probing, including [provider-specific ImageNet LP/AP preprocessing](docs/IMAGENET_PROBE_PREPROCESSING.md)
and [explicit ImageNet FT source profiles](docs/IMAGENET_FINETUNE.md)
for eight families. [ImageNet execution and continuation](docs/IMAGENET_EXECUTION.md)
add explicit accumulation, CUDA BF16 and portable epoch checkpoints; replicated
distributed execution excludes the unresolved Omega source path.
Source/protocol conflicts remain disclosed. The [paper reproduction gap ledger](docs/PAPER_REPRODUCTION_GAPS.md)
records which learning-design diagnostics have been ported and how they were checked.
The [vision-provider matrix](docs/BASIC5_VISION_PROVIDERS.md) lists the
additional local vision models, task coverage and unresolved recipe boundaries;
these integrations do not certify full-scale paper or workbook scores.

[Basic Five dense-task execution](docs/DENSE_EXECUTION.md) adds explicit LP/AP/FT accumulation
and training precision to ADE20K, NYUv2 and COCO for eight inspected families,
with replicated DDP for four source-supported families and accumulated gradient
reduction for seven V-JEPA2.1 task/adaptation routes.
Portable epoch continuation now preserves trained state and input identity.
The [shared continuation checks](docs/CONTINUATION_INTEGRITY.md) validate
optimizer recipes, state tensors and RNG records before live restoration.
Native schedule/loss disagreements, the remaining dense distributed/FSDP paths and
native continuation remain separate verification/integration work.

[Basic Five SSv2 execution and source recipes](docs/SSV2_EXECUTION.md) cover eight families
across LP/AP/FT with explicit clip augmentation, accumulation and portable resume.
The shared classification engine preserves the ImageNet execution contract.
These components do not certify native FSDP, decoder equivalence or paper scores.

| Directory | Method | Stages | Notes |
|---|---|---|---|
| `context_prediction` | Context Prediction — Doersch, Gupta & Efros, ICCV 2015 | pre-training + linear eval | the first pilot; verified on a CPU end to end |
| `vae` | VAE — Kingma & Welling, 2013 | pre-training | pretext-only; the one method on MNIST, so it trains to completion on a CPU |
| `colorization` | Colorization — Zhang, Isola & Efros, ECCV 2016 | pre-training + linear eval | a self-contained re-implementation (the lab's own VGG-style CNN, no submodule); predicts quantised **ab** colour bins from the **L** channel (313-way per-pixel classification); `encoder.pt` is the CNN encoder trunk (decoder and 313-bin head excluded); `linear_eval` probes its 512-d feature. RGB→Lab and ab-quantisation are numpy (no opencv/skimage); the 313 ab-bin constant is vendored |
| `context_encoder` | Context Encoder — Pathak et al., 2016 | pre-training + linear eval | the one GAN (two models, two optimisers); `encoder.pt` is the conv encoder + bottleneck |
| `jigsaw_puzzle` | Jigsaw Puzzles — Noroozi & Favaro, ECCV 2016 | pre-training + linear eval | a self-contained re-implementation (the lab's own CFN AlexNet, no submodule); predicts which permutation reordered the 3×3 tiles; `encoder.pt` is the shared CFN encoder |
| `rotation_prediction` | Rotation Prediction (RotNet) — Gidaris et al., ICLR 2018 | pre-training + linear eval | a self-contained re-implementation (the lab's own AlexNet-BN, no submodule); predicts which of {0°, 90°, 180°, 270°} was applied; `encoder.pt` is the AlexNet-BN encoder |
| `deepcluster` | DeepCluster — Caron et al., ECCV 2018 | pre-training + linear eval | a self-contained re-implementation (the lab's own AlexNet-BN, no submodule); each epoch, fc7 features are PCA-whitened and **k-means** clustered into pseudo-labels the network predicts; `encoder.pt` is the backbone (top_layer excluded); `linear_eval` probes fc7 (4096-d). **A GPU / x86_64-linux-only method** (it clusters features every training epoch): clustering uses **faiss** (the paper-target backend), whose wheel has no cross-platform build, so faiss is in the CUDA lock only (`jigsaw_puzzle_pp` shares this, but only for its knowledge-transfer stage) |
| `split_brain` | Split-Brain Autoencoders — Zhang, Isola & Efros, CVPR 2017 | pre-training + linear eval | a self-contained re-implementation (the lab's own two cross-channel AlexNets, no submodule); net1 predicts quantised **ab** (313 bins) from **L**, net2 predicts quantised **L** (50 bins) from **ab**; `encoder.pt` is both branch encoders (decoders excluded); `linear_eval` probes their concatenated 512-d feature. Lab + quantisation are numpy (no scipy/skimage) |
| `jigsaw_puzzle_pp` | Jigsaw++ — Noroozi et al., CVPR 2018 | pre-training + knowledge transfer + linear eval | a self-contained re-implementation (the lab's own VGG16 + AlexNet, no submodule); the **VGG16 jigsaw++ pretext** (occlusions + grayscale, 701 permutations), then the paper's **knowledge transfer** (cluster the VGG16 conv4 features with **faiss** k-means into pseudo-labels and train a standard AlexNet on them); `encoder.pt` is the VGG16 encoder (pre-training) or the AlexNet conv trunk (knowledge transfer); `linear_eval` probes either — `arch=vgg16` (default) or `arch=alexnet_cluster_cls`. The knowledge-transfer stage uses faiss, so it is GPU / x86_64-linux only (faiss in the CUDA lock); pre-training and the default probe stay torch-only |
| `inst_disc` | Instance Discrimination — Wu et al., CVPR 2018 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50, no submodule); NCE over a momentum **memory bank** (every image its own class); `encoder.pt` is the ResNet-50 backbone (the 128-d head and the bank are excluded) |
| `cpc` | CPC (visual CPC 2018) — van den Oord, Li & Vinyals, 2018 | pre-training + linear eval | a self-contained re-implementation (the lab's own `visual_cpc2018`, no submodule); a patch encoder + **PixelCNN context** predict future rows of the patch grid under **InfoNCE**; `encoder.pt` is the patch encoder (the context and predictors are excluded); `linear_eval` probes the grid-averaged z (`avg_z`). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively — the ViT's patch tokens as the CPC z-grid, probed at the CLS token |
| `cmc` | CMC (visual CMC 2019) — Tian, Krishnan & Isola, 2019 | pre-training + linear eval | a self-contained re-implementation (the lab's own AlexNet CMC, no submodule); an image becomes CIE **Lab** and its L/ab views feed a two-branch AlexNet trained by **NCE over two memory banks** (cross-view); `encoder.pt` is the two-branch encoder (the banks are excluded); `linear_eval` probes the layer-6 features of both branches concatenated. RGB→Lab is reimplemented in numpy, so the closure stays torch-only |
| `mocov1` | MoCo v1 — He et al., 2019 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50 MoCo, no submodule); a query encoder + a **momentum key encoder** contrast a positive against a FIFO **queue** of K negatives under **InfoNCE** (no MLP — that is v2); `encoder.pt` is the query ResNet-50 backbone (the head, key encoder and queue are excluded); `linear_eval` probes the backbone (2048-d) |
| `simclrv1` | SimCLR v1 — Chen et al., ICML 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50 SimCLR, no submodule); two augmented views feed a shared ResNet-50 + 2-layer MLP projection head, trained by **NT-Xent** (in-batch negatives) with the **LARS** optimiser and a cosine schedule; `encoder.pt` is the ResNet-50 backbone (the projection head is excluded); `linear_eval` probes the backbone (2048-d). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively |
| `mocov2` | MoCo v2 — Chen et al., 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50 MoCo v2, no submodule); MoCo v1 + a **2-layer MLP projection head**, **Gaussian-blur** augmentation and a **cosine** LR schedule (τ=0.2); a query encoder + a **momentum key encoder** contrast against a FIFO **queue** of K negatives under **InfoNCE**; `encoder.pt` is the query ResNet-50 backbone (the MLP head, key encoder and queue are excluded); `linear_eval` probes the backbone (2048-d). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively |
| `simclrv2` | SimCLR v2 — Chen et al., NeurIPS 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50 SimCLR v2, no submodule); SimCLR v1 + a **3-layer MLP projection head** (τ=0.1) with an optional `width_multiplier`; two views feed a shared ResNet-50, trained by **NT-Xent** (in-batch negatives) with **LARS** and a cosine schedule; `encoder.pt` is the ResNet-50 backbone (the projection head is excluded); `linear_eval` probes the backbone (2048-d). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively; the semi-supervised distillation is excluded |
| `swav` | SwAV — Caron et al., 2020 | pre-training + linear eval | its loader could not run on one process; the sampler is now conditional |
| `sela` | SeLa — Asano et al., ICLR 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet + torch **Sinkhorn-Knopp** optimal transport, no submodule); a ResNet backbone with multiple linear **prototype heads** is trained on pseudo-labels that Sinkhorn-Knopp assigns under an **equipartition** constraint, cross-entropy averaged over the heads (heads never reset, no Sobel); `encoder.pt` is the backbone (`top_layer` heads excluded); `linear_eval` probes it (2048-d). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively |
| `byol` | BYOL — Grill et al., 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own ResNet-50 BYOL, no submodule); an **online** net (backbone + projector + predictor) is trained so its prediction of one view matches an EMA **target** net's projection of the other, under a **symmetric negative-cosine** loss — **no negatives, no queue** (τ 0.996→1.0, LARS); `encoder.pt` is the online ResNet-50 backbone (the projector, predictor and target are excluded); `linear_eval` probes the backbone (2048-d). The controlled ViT-B/16 pre-training recipe (`arch: vit`, which needs `timm`) is ported additively |
| `simsiam` | SimSiam — Chen & He, 2020 | pre-training + linear eval | the ImageNet probe evaluates the trained encoder |
| `barlow_twins` | Barlow Twins — Zbontar et al., 2021 | pre-training + linear eval | refuses fp16 on a CPU rather than downgrading quietly |
| `mocov3` | MoCo v3 — Chen, Xie & He, 2021 | pre-training + linear eval | a self-contained re-implementation (the lab's own ViT MoCo v3, no submodule); two views feed a **base** ViT (+ 3-layer MLP projector) and a **momentum** ViT (an EMA copy), with a 2-layer **predictor** on the base, trained by a **symmetric InfoNCE** loss (EMA momentum cosine-scheduled to 1.0); `encoder.pt` is the base ViT trunk (the projector head, predictor and momentum encoder are excluded); `linear_eval` probes its CLS feature (embed_dim). **The first ported method that needs `timm`** (it supplies the VisionTransformer base class), but the ViT is built from scratch (no pretrained download), so the run stays hermetic. The controlled ViT-B/16 pre-training recipe (`recipe: unified`) is ported additively |
| `dino` | DINO — Caron et al., 2021 | pre-training + linear eval | a self-contained re-implementation (the lab's own DINO, no submodule); **self-distillation with no labels** — a **student** (ViT + 3-layer MLP head) sees all multi-crop views and a **teacher** (an EMA copy) sees the 2 global views, trained by a **centred + sharpened cross-entropy** (an online centre EMA prevents collapse; teacher momentum cosine 0.996→1.0); `encoder.pt` is the **teacher** ViT backbone (`teacher.backbone.*`; the head, centre and student are excluded — the teacher is the representation DINO is known for); `linear_eval` probes its CLS feature (embed_dim). DINO ships **its own ViT** (not `timm`), so this port is **torch-only**. The controlled ViT-B/16 pre-training recipe (`recipe: unified`) is ported additively |
| `beit` | BEiT — Bao et al., 2021 | pre-training + linear eval | a self-contained re-implementation (the lab's own BEiT, no submodule); **masked image modeling with discrete targets** — a **dVAE tokenizer** turns each image into discrete **visual tokens**, a random block of patches is replaced by a shared learned **mask token** in the ViT input, and the ViT (LayerScale blocks) predicts the visual tokens at the masked positions under a **cross-entropy over the dVAE vocabulary**; `encoder.pt` is the BEiT backbone trunk (`patch_embed`/`cls_token`/`pos_embed`/`blocks`/`norm`; the mask token and MIM head excluded); `linear_eval` probes its mean-pooled patch tokens (embed_dim, CLS excluded). The ViT is BEiT's **own** (not `timm`), so it is **torch-only**; the MIM targets come from the frozen **OpenAI DALL-E dVAE** — a hash-pinned `encoder.pkl` download imported lazily via `dall_e` for a real run (CI's smoke uses a random tokenizer, so it downloads nothing). The controlled ViT-B/16 pre-training recipe (`recipe: unified`, the same MIM objective under the unified schedule) is ported additively |
| `mae` | MAE — He et al., CVPR 2022 | pre-training + linear eval | a self-contained re-implementation (the lab's own MAE, no submodule); masked-autoencoder pretraining; `linear_eval` probes the trained encoder (avg-pooled patch tokens) |
| `simmim` | SimMIM — Xie et al., 2022 | pre-training + linear eval | a self-contained re-implementation (the lab's own SimMIM, no submodule); **masked image modeling** — a random block of Swin patch tokens (60% of 32×32 mask units) is replaced by a learned **mask token**, the grid is encoded, and a Conv + PixelShuffle decoder reconstructs pixels under an **L1 loss on the masked pixels only**; `encoder.pt` is the bare **Swin encoder** (`encoder.*`; the mask token and decoder excluded); `linear_eval` probes its mean-pooled features (encoder_dim). The repo's **first Swin backbone** — genuinely Swin-based, so it needs `timm` (built from scratch, hermetic); the `transformers` dep in the capture was docstring-only (measured). The controlled ViT-B/16 pre-training recipe (`recipe: unified`) is ported additively |
| `ibot` | iBOT — Zhou et al., 2021 | pre-training + linear eval | first exercised on an A100 as written; `encoder.pt` is the teacher ViT |
| `dinov2` | DINOv2 — Oquab et al., 2023 | pre-training + linear eval | two comparisons, the DINOv2 sibling of `franca`. **ASIS:** probes the frozen official pretrained **ViT-g/14** (LVD-142M) CLS token, a frozen representation (LVD-142M is not public, so the as-is row reuses the official backbone). **Controlled pre-training:** a from-scratch ViT-B/16 (DINO + iBOT + KoLeo) on ImageNet-1k, probed frozen — ported additively. Backbone code is the pinned `third_party/dinov2` submodule (imported, xformers disabled → torch-only; the giant's SwiGLU keys match the fused checkpoint); weights are a **hash-pinned 4.5 GB download** (CI never downloads — the smoke uses a random ViT-S/14). [checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md) |
| `ijepa` | I-JEPA — Assran et al., 2023 | pre-training + linear eval | a self-contained re-implementation (the lab's own I-JEPA, no submodule); **latent-space prediction** — a **context encoder** (ViT) sees a context block, a narrow **predictor** predicts the representations of masked **target blocks**, and the targets come from an **EMA target encoder** under a **smooth-L1 loss in latent space** (no pixels, no augmentation invariances); `encoder.pt` is the **target** ViT encoder (`target_encoder.*`; the context encoder, predictor and mask token excluded); `linear_eval` probes its mean-pooled patch tokens (embed_dim, no CLS token). I-JEPA ships **its own ViT** (not `timm`) and trains from scratch on ImageNet, so this port is **torch-only**. The controlled ViT-B/16 pre-training recipe (`recipe: unified`) is ported additively |
| `aim` | AIM — El-Nouby et al., 2024 | pre-training + linear eval | two comparisons, the AIM sibling of `dinov2` / `franca`. **ASIS:** probes the frozen official pretrained **AIM-600M** (ViT-H/14, autoregressively pretrained on DFN-2B+) — the average of its last 6 transformer blocks, patch mean-pooled — a frozen representation (DFN-2B+ is not public, so the as-is row reuses the official backbone). **Controlled pre-training:** a from-scratch ViT-B/16 (prefix-LM next-patch MSE) on ImageNet-1k, probed frozen — ported additively (the lab's own torch-only AIM re-impl, not bound by apple-amlr). Backbone code is the pinned `third_party/ml-aim` submodule (Apple's `aim`, imported not copied); weights are a **hash-pinned download** (CI never downloads — the smoke builds a random tiny AIM). **Licence: `apple-amlr`, non-commercial research use only** — nothing under it is copied here (submodule + download); academic-research use only (see `provenance.json`). [checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md) |
| `dinov3` | DINOv3 — Siméoni et al., 2025 | pre-training + linear eval | a self-contained re-implementation (the lab's own DINOv3, no submodule); **self-distillation** — a **student** ViT (with **register tokens** + **axial RoPE**) and an EMA **teacher** see multi-crop views (2 global + 8 local), trained by the DINOv3 **core** objective `L_DINO + L_iBOT + 0.1·L_KoLeo` (DINO with **Sinkhorn-Knopp** centring, iBOT masked-patch distillation); `encoder.pt` is the **teacher** ViT backbone (`backbone.*` from `teacher_state_dict`, prefix stripped; the DINO/iBOT heads + student excluded); `linear_eval` probes its CLS feature (embed_dim). This ports the from-scratch CTRL recipe; the ASIS checkpoint (HF-**gated** official DINOv3 weights) and the released **Gram anchoring** stage (`gram.mode: core_only`) are excluded. The ViT is the lab's **own** (not `timm`) so this port is **torch-only**; a from-scratch re-implementation, so **no Meta code/weights** are used |
| `nepa` | NEPA — Xu et al., 2025 | pre-training + linear eval | a self-contained re-implementation (the lab's own NEPA, no submodule); **next-embedding predictive autoregression** — patch embeddings run through a **causal** transformer predictor, trained by a **negative-cosine** loss against a **stop-gradient** shifted (next-patch) target; an EMA copy is kept for eval; `encoder.pt` is the **EMA** model (`ema_model.*`; the online model excluded); `linear_eval` probes its avg-pooled predictor output (embed_dim). The repo's **first 2D-RoPE ViT** (with QK-norm + LayerScale) and first autoregressive latent objective. NEPA ships **its own ViT** (not `timm`) and trains from scratch on ImageNet, so this port is **torch-only**. The controlled ViT-B/16 pre-training recipe is ported additively — the same `NEPAModel` at `patch_size 16` with the unified recipe + milestones, probed at its raw patch embeddings (`pool: embed`) |
| `pirl` | PIRL — Misra & van der Maaten, CVPR 2020 | pre-training + linear eval | a self-contained re-implementation (the lab's own PIRL, no submodule); **pretext-invariant learning** — a ResNet-50 trunk encodes an image and a **jigsaw-shuffled** view, both contrasted against a momentum **memory bank** (one row per image) with an NCE cross-entropy; the loss is a convex combination of the image-NCE and the jigsaw-NCE, and the bank is updated with the image feature each step; `encoder.pt` is the ResNet-50 trunk (`encoder.*`; the projection heads and the bank excluded — the bank lives in the loss module); `linear_eval` probes the backbone (2048-d). The repo's **third memory-bank method** (with `inst_disc`, `cmc`); torch-only. The controlled ViT-B/16 pre-training recipe (`arch: vit`) is ported additively |
| `msn` | MSN — Assran et al., 2022 | pre-training + linear eval | a **submodule-import** port of the official `facebookresearch/msn` (pinned `third_party/msn`, imported not copied); **masked siamese networks** — an anchor ViT sees patch-dropped multi-crop views and an EMA **target** ViT sees one un-dropped view, both matched to learnable **prototypes** via a soft-nearest-neighbour classifier under an MSN cross-entropy + **me-max** regulariser; `encoder.pt` is the anchor ViT trunk (`fc.*` projection head excluded); `linear_eval` probes its CLS feature (embed_dim). The ViT + loss are imported from the submodule (torch-only in the imported parts); the multi-view aug is reimplemented (the upstream one trips the pinned Pillow), the trainer is single-process (DDP/submitit/cyanure dropped), and the probe is the shared ARSSL probe. **Licence: `CC BY-NC 4.0`, non-commercial research use only** — nothing under it is copied here (submodule); academic-research use only (see `provenance.json`). The controlled ViT-B/16 pre-training recipe (`recipe: unified`) is ported additively |
| `vjepa` | V-JEPA — Bardes et al., 2024 | pre-training + linear eval | a **submodule-import** port of the official `facebookresearch/jepa` (pinned `third_party/jepa`, imported not copied); **latent-space prediction** — a context encoder sees the visible tokens, an EMA **target** encoder encodes the whole input, and a predictor predicts the target's representations at masked positions (3D multi-block masks, smooth-L1 latent loss). This ports the controlled **image adaptation** (`num_frames=1`, an image ViT-B/16 trained from scratch on ImageNet); the released video model uses a separate protocol; `encoder.pt` is the EMA target encoder; `linear_eval` probes its mean-pooled tokens (embed_dim). The ViT + predictor + mask collator are imported from the submodule (torch-only); trainer single-process (DDP/TensorBoard dropped), shared ARSSL probe. **Licence: `CC BY-NC 4.0`, non-commercial research use only** — nothing under it is copied here (submodule); academic-research use only (see `provenance.json`) |
| `franca` | Franca — arXiv:2507.14137 | pre-training + linear eval | two comparisons, the sibling of `dinov2` / `aim`. **ASIS:** probes the frozen pretrained Franca ViT-B/14 CLS token (a hash-pinned download), a frozen representation. **Controlled pre-training:** a from-scratch ViT-B/16 (Franca's objective) on ImageNet-1k, probed frozen — ported additively ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `lejepa` | LeJEPA — Balestriero & LeCun, 2025 | pre-training + linear eval | a self-contained re-implementation (the lab's own LeJEPA, no submodule); each image is seen as several augmented **views**; a timm ViT backbone + a projection MLP maps each view to a projected feature, trained by **SIGReg** (an Epps-Pulley Gaussian regularizer over random 1-D **slices** of the batch — matching the empirical characteristic function to `exp(−t²/2)` on a quadrature grid) plus a **cross-view invariance** loss, combined as `SIGReg·λ + invariance·(1−λ)`; `encoder.pt` is the bare backbone (`backbone.*`, prefix stripped so it loads into a plain timm model; the projector excluded); `linear_eval` probes its num_features feature. Needs `timm` (ViT built from scratch, hermetic); **SIGReg is reimplemented locally** (no external package), so it ports **self-contained** — no submodule, no downloaded weights. The controlled ViT-B/16 pre-training recipe (`recipe: unified`, the same objective under the unified schedule) is ported additively |
| `clip` | CLIP — Radford et al., 2021 (arXiv:2103.00020) | pre-training + linear eval | two comparisons, the CLIP sibling of `dinov2` / `aim` / `franca`. **ASIS:** freezes the released official **ViT-B/32** image tower (a hash-pinned download) and probes its pooled image embedding — CLIP's 400M image-text data is not public, so the as-is row reuses the released checkpoint. **Supervised label-text adaptation:** a from-scratch CLIP **ViT-B/16** on ImageNet-1k, pairing each labeled image with an official ImageNet class-name prompt (symmetric image-text contrastive loss), probed frozen at milestones 100/200/300. **This is a supervised label-text adaptation, disclosed as `supervised_label_text_adaptation=true` / `main_vssl_comparability=false` — a CLIP-adaptation reference, not a comparable unlabeled self-supervised number.** Model constructor + BPE tokenizer + ImageNet prompts imported from the pinned `third_party/CLIP` submodule (**MIT**, imported not copied). [checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md) |
| `aimv2` | AIMv2 — Fini et al., 2024 (arXiv:2411.14402) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes the official multimodal-autoregressively pretrained backbone (a hash-pinned download) and probes its pooled feature — a genuine learned representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `franca`). **No author submodule:** the model class is timm's (a pinned pip dependency), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `beitv2` | BEiT v2 — Peng et al., 2022 (arXiv:2208.06366) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes the official self-supervised pretrained backbone (a hash-pinned download) and probes its pooled patch-token feature — a genuine learned SSL representation (the "pretrained-backbone reuse" row, the autoregressive/MIM sibling of `eva02` / `franca`). **Pinned author submodule:** the model class is the author's `modeling_finetune.VisionTransformer`, imported from `third_party/unilm` (**MIT**, imported not copied), so an `upstream` is recorded; the weights are a sha256-pinned download (`backbone_artifact`). Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `cae` | CAE — Chen et al., 2022 (arXiv:2202.04200) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes a pretrained CAE ViT-B backbone (a hash-pinned download) and probes its CLS token — a genuine learned SSL representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `aimv2` / `data2vec2` / `franca`). **No author submodule and no model-carrying pip dependency:** neither timm nor transformers carries a CAE model class, so the encoder is a small self-contained BEiT-style ViT in `evaluate_linear_cae.py` and the weights are a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. The pinned checkpoint is **OpenMMLab's mmselfsup reproduction** of CAE ViT-B, not the authors' released weights (the capture logged the official weights as unavailable, DEF-02; `provenance.json` says so), read without mmcv via a tolerant unpickler. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `cosmos3_super` | Cosmos3-Super — NVIDIA, 2026 (huggingface.co/nvidia/Cosmos3-Super) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining), the `transformers`-sourced sibling of `data2vec2` / `sam3`: freezes the released Cosmos3-Super Qwen3-VL vision encoder (a **public** OpenMDW-1.1, hash-pinned download; the 64B MoT / DiT / VAE are never loaded) and probes its mean-pooled patch tokens — a genuine learned representation (the multimodal "pretrained-backbone reuse" row). **No author submodule:** the model class is `transformers`' `Qwen3VLVisionModel` (a pinned pip dependency, `transformers==5.16.1`), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Unlike `sam3`, the released checkpoint is **directly loadable** by the HF class (`from_pretrained` on the `vision_encoder/` directory, no trunk conversion; a `save_pretrained`→`from_pretrained` round-trip is exact, so the loading path is unit-tested without the ~1.1 GB weights). Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `cradiov4_h` | C-RADIOv4-H | downstream components only | Local official summary/spatial readouts, video CPE and 14 LP/AP/FT fixture routes. Full-size checkpoint and numerical reproduction remain unverified. See [scope](methods/cradiov4_h/README.md). |
| `data2vec2` | data2vec 2.0 — Baevski et al., 2023 (arXiv:2212.07525) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining) and the first `transformers`-sourced one: freezes the official pretrained vision backbone `facebook/data2vec-vision-base` (a hash-pinned download) and probes its CLS token — a genuine learned SSL representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `franca`). **No author submodule:** the model class is `transformers`' `Data2VecVisionModel` (a pinned pip dependency new to the fleet), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `eva02` | EVA-02 — Fang et al., 2023 (arXiv:2303.11331) | linear eval | the first **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes the official masked-image-modelling pretrained backbone (a hash-pinned download) and probes its pooled feature — a genuine SSL representation (the "pretrained-backbone reuse" row, analogous to `franca`). **No author submodule:** the model class is timm's (a pinned pip dependency), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `image_gpt` | iGPT — Chen et al., ICML 2020 | pre-training + linear eval | a self-contained re-implementation (no submodule) ported from the lab's inline model; pretrains a GPT on colour-cluster tokens; `linear_eval` probes the trained model |
| `mar` | MAR — Li et al., NeurIPS 2024 | pre-training | the first `submodule+patch` port: the model is the pinned `third_party/mar` fork, imported not copied; `linear_eval` deferred — its captured eval path is unrecoverable ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `raev2` | RAEv2 K7 tokenizer | downstream components only | Pinned DINOv3-L encoder; seven selected layers plus the final-layer global term; 14 LP/AP/FT task routes. Historical RAE readouts and full result reproduction remain separate. See [scope](methods/raev2/README.md). |
| `sam3` | SAM 3 — Meta AI, 2025 (ai.meta.com/research/sam3) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining), the `transformers`-sourced sibling of `data2vec2`: freezes the official SAM 3 vision encoder (a **gated**, hash-pinned download) and probes its mean-pooled patch tokens — a genuine learned representation (the multimodal "pretrained-backbone reuse" row). **No author submodule:** the model class is `transformers`' `Sam3ViTModel` (a pinned pip dependency, `transformers==5.16.1`), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Because the official `sam3.pt` uses ViTDet-style trunk keys that do not match the HF class, a real run converts them (`sam3_trunk.py`, unit-tested on synthetic tensors so the gated-weight path is covered hermetically). Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `siglip` | SigLIP — Zhai et al., 2023 (arXiv:2303.15343) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining) and the first multimodal one: freezes the official sigmoid-loss image-text pretrained image tower (a hash-pinned download) and probes its pooled image embedding — a genuine learned representation (the multimodal "pretrained-backbone reuse" row, the SigLIP sibling of `clip`). **No author submodule:** the model class is timm's (a pinned pip dependency), the weights a sha256-pinned download (`backbone_artifact`), so no upstream is recorded. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `var` | VAR — Tian et al., NeurIPS 2024 | pre-training + linear eval | the first `submodule+adapter` port: `third_party/var` pinned directly (no fork). Next-scale autoregressive generation; `linear_eval` probes the pretrained VQVAE **tokeniser** (a hash-pinned download), which measures the fixed tokeniser rather than VAR's learned representation ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `vggt_omega` | VGGT-Omega visual aggregator | downstream components only | Pinned native multi-frame aggregator; 14 LP/AP/FT task routes and source-specific COCO labels. No full method adapter or paper-score reproduction is claimed. See [scope](methods/vggt_omega/README.md). |
| `videomae` | VideoMAE — Tong et al., NeurIPS 2022 (arXiv:2203.12602) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes the official masked-video pretrained encoder (a hash-pinned download) and probes its **temporally-averaged patch tokens** — a genuine learned SSL representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `aimv2` / `data2vec2` / `cae`). Following the capture's ImageNet eval, each still image is replicated `num_frames` times to form the clip VideoMAE expects (no PyAV / video dataset). **No author submodule and no model-carrying pip dependency:** neither timm nor transformers carries a VideoMAE model class, so the encoder is a small self-contained Conv3d-patch-embed ViT in `evaluate_linear_videomae.py` whose module names mirror the checkpoint's `videomae.*` keys, and the weights are a sha256-pinned download (`backbone_artifact`, the authors' official public `MCG-NJU/videomae-base`, **CC-BY-NC-4.0**), so no upstream is recorded. The `videomae.*` encoder tensors load directly (prefix stripped); the `decoder.*` pretext-decoder keys are dropped. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `vjepa2` | V-JEPA 2 — Assran et al., Meta FAIR 2025 (arXiv:2506.09985) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining): freezes the official latent-prediction pretrained video encoder (a hash-pinned download) and probes its **mean-pooled patch tokens** — a genuine learned SSL representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `data2vec2` / `cae` / `videomae`). Following the capture's ImageNet eval, each still image is replicated `num_frames` times to form the clip V-JEPA 2 expects (no PyAV / video dataset). **No author submodule and no model-carrying pip dependency:** neither timm nor transformers carries this V-JEPA 2 model class, so the encoder is a small self-contained Conv3d-patch-embed ViT in `evaluate_linear_vjepa2.py` whose module names mirror the checkpoint's `encoder.*` keys, and the weights are a sha256-pinned download (`backbone_artifact`, the authors' official public `facebook/vjepa2-vitl-fpc64-256`, **MIT** — measured from the model card; the capture's stale `CC BY-NC 4.0` label is corrected in `provenance.json`), so no upstream is recorded. The `encoder.*` tensors load directly (prefix stripped); the `predictor.*` JEPA-predictor keys are dropped. The encoder runs with **plain (non-rotary) attention**, mirroring the capture's wrapper (a stated approximation of V-JEPA 2's rotary forward). Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |
| `vjepa2_1` | V-JEPA 2.1 | downstream components only | Pinned ViT-L/ViT-g encoder; image grids, native-video tokens, frozen/AP/FT integration for four downstream CLIs. **No complete method adapter or paper-score reproduction yet.** See [scope and validation](methods/vjepa2_1/README.md). |
| `vjepa2_ac` | V-JEPA 2-AC — Assran et al., Meta FAIR 2025 (arXiv:2506.09985) | linear eval | a **pure eval-only** port (a single `linear_eval` stage, no pretraining), the **action-conditioned** V-JEPA 2 world model: freezes the official latent-prediction pretrained video encoder (a hash-pinned download) and probes its **mean-pooled patch tokens** — a genuine learned SSL representation (the "pretrained-backbone reuse" row, analogous to `eva02` / `data2vec2` / `cae` / `videomae` / `vjepa2`). Following the capture's ImageNet eval, each still image is replicated `tubelet_size` times to form the one-temporal-token clip V-JEPA 2 expects (no PyAV / video dataset). **Faithful, where the sibling `vjepa2` approximates:** the ViT is the **pinned `facebookresearch/vjepa2` submodule** (`third_party/vjepa2`, imported not copied), and the forward runs V-JEPA 2's **real rotary-position attention** (`use_rope=True`) — reproducing the capture's number rather than the plain-attention approximation — so an `upstream` **is** recorded (adapter `UPSTREAM` + `provenance.json`, which must agree). The weights are a sha256-pinned download (`backbone_artifact`, the authors' official public `vjepa2-ac-vitg.pt`, the provenance records **MIT** from the source repository license; redistribution of the specific weight artifact still requires artifact-level verification (see the public license inventory)). The checkpoint is a `.pt` training-state dict; its `encoder.*` tensors (a `module.` prefix stripped) load into the submodule's `vit_giant_xformers` with a **measured-exact** match (484 == 484, any missing/unexpected key a hard error) and the `predictor`/`opt`/`scaler`/`target_encoder` entries are dropped. The submodule loads `einops` and `timm` at import, so both are lock-only pins. Produces no `encoder.pt`; the manifest carries `encoder_absent_reason` ([checkpoint evaluation protocol](docs/EVAL_DOWNLOAD.md)) |

A `linear_probe` score reports classification accuracy for a particular dataset and readout.
ASIS uses family-specific protocols; its rankings describe those settings.
Section 3 explains how feature extraction and downstream training differ across tracks. `vae` is pretext-only and `mar` has no linear eval; `var`'s
`linear_eval` probes a fixed pretrained tokeniser rather than its own learned
representation, so the score describes the tokeniser
(`docs/EVAL_DOWNLOAD.md`). `methods/_reference/` is not a method under study but
the known-good adapter the contract tests run against.
Deferred, not dropped: `VideoGen` (LTX-2), which needs CUDA > 12.7 and a 22B
checkpoint.

## Repository layout

`exists` is in the tree and under test. `planned` is not written yet, and is
shown so the shape is visible before it is built.

```
10YearVisualSSL/
├── bin/                            command-line tools                exists
│   ├── resolve-config.py             authoring config -> canonical resolved JSON
│   ├── launch.py                     resolve, submit, verify, record
│   ├── matrix-run.py                  drive the method x stage grid; one verdict
│   ├── matrix-audit.py                judge a produced grid: did everything land?
│   ├── verify-environment.py         is this the locked environment?
│   ├── run-ci-locally.py            run the workflow here, by reading it
│   ├── validate-push.py             validate offline, then check evidence and push
│   ├── mutate.py                     break the code, check the tests notice
│   ├── submission-archive.py        audit and package an anonymous source ZIP
│   ├── build-lock.py                 render a resolved set into a CPU lock
│   ├── fetch-weights.py              download a pinned, hash-checked backbone
│   ├── contract-test.py              decides by machine that a port is finished
│   ├── export-native-encoder.py    verify and export native weights through an adapter
│   ├── prepare-native-encoders.py  export a batch using pinned method provenance
│   ├── extract-features.py          one run: save every method's val features
│   ├── extract-reference.py         extract validation through an audited reference evaluator
│   ├── import-reference-cache.py    validate pinned caches and restore sample order
│   ├── prepare-imagenet-val.py      ImageNet-1k val parquet -> val/<wnid>/*.JPEG
│   ├── render-readme-figure3.py      draw the public README chart from the reported means
│   └── aggregate-seeds.py           seeds 0,1,2 -> mean +/- std (BASIC5 rule seed)
├── adapterlib/                     the one place a run_manifest.json is written
│   └── __init__.py                                                    exists
├── downstream/                     cross-method downstream tasks      exists
│   ├── contract.py                   the downstream contract: write + verify
│   ├── arssl.py                      ARSSL harness: drive one backbone through them all
│   ├── spatial_backbones.py          frozen backbone -> [B,C,h,w] + out_channels
│   ├── ade20k.py                     ADE20K segmentation (the pilot)
│   ├── coco.py                       COCO detection (Faster R-CNN + mAP)
│   ├── nyuv2.py                      NYUv2 depth (DPT head + RMSE/AbsRel)
│   ├── ssv2.py                       SSv2 video (frame-average/native tokens + top-1/5)
│   ├── imagenet.py                   online LP/AP components; explicit provider readout
│   ├── requirements.txt              the downstream env's direct deps
│   └── requirements.lock.txt         its pinned, hashed closure (CI runs it)
├── methods/                        one directory per method
│   ├── _reference/                   known-good adapter; trains nothing exists
│   │   └── adapter/
│   │       ├── __init__.py             the body: what this method does
│   │       └── __main__.py             python -m adapter --config ... --out ...
│   ├── context_prediction/         context-prediction pre-training              exists
│   │   ├── Dockerfile                  the locked environment as an image
│   │   ├── adapter/                    translates the config, calls the original
│   │   ├── train_pretrain_alexnet_official.py   the original loop, extracted
│   │   ├── models/ data/               untouched; digests pinned by tests
│   │   ├── configs/pretrain.yaml          the settings the capture used
│   │   ├── configs/linear_eval.yaml    linear probe on frozen features
│   │   ├── requirements.txt            which packages, checked against imports
│   │   ├── requirements.lock.txt       exact versions, to rebuild a run
│   │   ├── provenance.json             what came across, and what changed
│   │   └── README.md                   the science, and the port's deviations
│   ├── vae/                        VAE pre-training            exists
│   │   ├── adapter/  configs/  models/  data/
│   │   ├── Dockerfile  requirements.lock.txt  provenance.json
│   │   └── README.md                   MNIST; trains to completion on CPU
│   └── VideoGen/                     deferred: needs a GPU            planned
│       ├── adapter/
│       └── configs/
├── third_party/                    pinned upstreams and forks        exists
│   ├── mar/                          pinned submodule (fork), used by methods/mar
│   ├── var/                          pinned submodule, used by methods/var
│   ├── franca/                       pinned submodule, used by methods/franca
│   ├── dinov2/                       pinned submodule, used by methods/dinov2
│   ├── dall_e/                       pinned submodule, used by methods/beit
│   ├── ml-aim/                       pinned submodule, used by methods/aim
│   ├── msn/                          pinned submodule, used by methods/msn
│   └── jepa/                         pinned submodule, used by methods/vjepa
├── mutations/                      mutation specs, with their measured results
├── platforms/                      where a job runs. Loosely coupled  exists
│   ├── base.py                       the shared interface, free of platform terms
│   ├── local/backend.py              this machine. The default, self-contained
│   └── abci/backend.py               optional

├── configs/                        submission policy example; shared bases planned
│   └── submission-policy.example.json  private-policy schema example
├── runs/                           run outputs. Not tracked           exists
│   └── <method>-<config sha>/        named after the config, not the clock
│       ├── launch.json               what was asked, and how it turned out
│       ├── resolved.json             the exact config that ran
│       └── out/
│           ├── encoder.pt
│           ├── metrics.json
│           └── run_manifest.json
├── docs/
│   ├── PLATFORMS.md                  the platform separation          exists
│   ├── EVAL_DOWNLOAD.md              generative-method probes + weights
│   ├── EVALUATION.md                 the 5-task eval axis, and coverage
│   ├── DOWNSTREAM.md                 the cross-method downstream tasks
│   ├── SUBMISSION_ARCHIVE.md         anonymous supplementary ZIP workflow
│   ├── PORTING_ROADMAP.md            historical method inventory and implementation status
│   └── GPU.md                        GPU env + the device invariant   exists
├── tests/                          one file per unit, plus the chain  exists
│   ├── test_resolve_config.py
│   ├── test_adapterlib.py
│   ├── test_contract_test.py
│   ├── test_end_to_end.py            resolve -> adapt -> verify, for real
│   ├── test_method_requirements.py   declarations match the imports
│   ├── test_launch.py                resolve -> submit -> verify -> record
│   ├── test_ci.py                    the workflow runs what it claims
│   ├── test_mutate.py                the mutation tool cannot lie
│   ├── test_method_swav.py         the fifth port
│   ├── test_method_barlow_twins.py the fourth port
│   ├── test_metric_vocabulary.py     one vocabulary, and what may be compared
│   ├── test_encoder_convention.py    every port loads back what it wrote
│   ├── test_method_simsiam.py     the third port
│   ├── test_no_hard_coded_methods.py shared machinery discovers methods
│   ├── test_repository_scan.py       one scan, and it works without git
│   ├── _repo_files.py                which files belong to the repository
│   ├── test_language.py              everything here is in English
│   └── test_repo_hygiene.py          nothing generated is tracked
├── .github/workflows/tests.yml     CI: the suite on linux x86_64      exists
├── CLAUDE.md                       the working rules                  exists
├── README.md
└── LICENSE                         MIT, for our code only             exists
```

The design of record is not here — see *Where the design lives* below.

**`methods/_reference` is not an example that might rot.** It is exercised by
`tests/test_end_to_end.py` through the real chain, so a later adapter can copy
something known to pass.

### Where third-party code goes

**`third_party/<name>/`, one pinned submodule per source repository.**

Keep integration code outside upstream trees, usually in
`methods/<name>/adapter/`. Where a pinned public fork contains modifications,
record that distinction and preserve its original license; see the
[license inventory](docs/THIRD_PARTY_LICENSES.md). A typical adapter imports
from a submodule:

```
third_party/deepcontext/          <- the authors' repository, at a pinned commit
methods/context_prediction/
    adapter/__init__.py           <- ours: imports deepcontext, writes the outputs
```

**One shared directory, not one per method,** because the originals already
share: measured across the 31 recorded upstream repositories, `cosmos`
(`444d86120a57`) and `vggt-omega` (`39a0cb8af885`) each appear under two
different methods **at the same commit**. Per-method placement would make each
of those two submodules, two clones, and two pins that can drift apart without
anything noticing.

It also settles a naming question the originals leave open. They spell the
same idea four ways — `third_party/` (15), `external/` (7), `repo/` (5),
`repos/` (4). One spelling here.

Where a method needs the upstream code *changed*, the change is a branch on
our fork and the submodule pins a commit on that branch, so this repository
still contains no modified copy of anyone's work. See the [third-party inventory](docs/THIRD_PARTY_LICENSES.md) for the sources used by this repository.

## Requirements

Python 3.10 or newer. **The core needs nothing installed**; it is standard
library only, so it also runs on a login node that forbids extra packages.

```bash
python3 --version
```

Writing configs in YAML is optional and needs one package. JSON configs need
nothing, and the resolved artifact is JSON either way.

```bash
python3 -m pip install pyyaml     # only if you want to author in YAML
```

`./tests/run-tests.sh` prints whether PyYAML is present, so a skipped test is
never mistaken for a passing one.

**Each method declares its own training dependencies**, in its
`requirements.txt` (which packages) and `requirements.lock.txt` (**the full
closure**, every version exact and every wheel hashed). The interpreter is
pinned in `.python-version`. `tests/test_method_requirements.py` checks a
method's declarations against what it actually imports, in both directions,
refuses a lock that is not a closure, and refuses an entry without a hash. The
core never imports any of them.

## Reproducibility: the resolved config

**`config_sha256` is the hinge.** `run_manifest.json` claims that one
configuration produced one result, and that claim is worth something only if
the same configuration always hashes the same way. `bin/resolve-config.py`
produces that canonical form.

Write the authoring configs — `include` lets a method reuse a shared base.
**These keys are illustrative**; each method defines its own, and
`methods/context_prediction/configs/pretrain.yaml` is a real one:

```bash
mkdir -p configs && printf '{"seed":0,"optimizer":{"name":"sgd","lr":0.1,"momentum":0.9}}\n' > configs/base.json
printf '{"include":["base.json"],"optimizer":{"lr":0.03},"data_root":"${DATA_ROOT}"}\n' > configs/example.json
```

Resolve. Values come from `--set`, never from the environment:

```bash
python3 bin/resolve-config.py --config configs/example.json --out runs/demo/resolved.json --set DATA_ROOT=/mnt/data
```

```
  wrote runs/demo/resolved.json
  config_sha256 0639d99a22108b2548335912300c2905e1b05767feab17091a78ad4f0c47d813
```

The resolved file is one line, keys sorted, `include` gone, `${DATA_ROOT}`
gone, and `optimizer.lr` overridden while `momentum` survives the merge:

```json
{"data_root":"/mnt/data","optimizer":{"lr":0.03,"momentum":0.9,"name":"sgd"},"seed":0}
```

Check the hash with anything you like — it is a plain sha256 of those bytes:

```bash
shasum -a 256 runs/demo/resolved.json
```

To get the hash without writing anything:

```bash
python3 bin/resolve-config.py --config configs/example.json --print-hash --set DATA_ROOT=/mnt/data
```

### What it refuses, and why

**The environment is never read.** A config that silently absorbs the machine
it was resolved on is not reproducible, so an unset variable stops the run and
nothing is written:

```bash
python3 bin/resolve-config.py --config configs/example.json --out /tmp/x.json; echo "EXIT=$?"
```

```
  *** data_root: DATA_ROOT is not set. The environment is never consulted -- pass it with --set DATA_ROOT=<value>
EXIT=1
```

It also refuses, by name: an `include` that is missing, cyclic, or reaches
outside the config root; a duplicate key (one value would be dropped without a
word); `NaN` and `Infinity`; and anything JSON cannot carry. Nothing is
skipped quietly.

**Why JSON and not YAML for the resolved file:** JSON has a canonical form
reachable from the standard library, and YAML has neither. `gpus: 8` and
`gpus: 8.0`, quoted and unquoted strings, differing key order — all change the
bytes without changing the settings, which would make the hash meaningless.

## Running the tests

Pull requests and pushes to `main` run checks on linux x86_64 —
[`.github/workflows/tests.yml`](.github/workflows/tests.yml). The pre-commit
hook is machinery, but machinery on one machine: it needs configuring per
clone, `--no-verify` skips it, and it only ever exercises the platform the
committer happens to have. CI closes all three, and it is where the linux
x86_64 claim is actually checked — everything end-to-end before it ran on
macOS arm64.

| Job | When | What |
|---|---|---|
| `core` | every PR and main push | the suite with **nothing installed** |
| `discover` | every PR and main push | discover locks and select affected environments |
| `locked` | PRs with selected methods | install hash-checked and run the whole suite once per distinct supported lock input |
| `container` | PRs with selected methods | build each selected image and run its contract chain |
| `downstream` | non-prose PRs and every main push | run the downstream task checks |

See [CI selection](docs/CI_SELECTION.md) for the exact prose allowlist,
method selection, full-matrix fallback and local execution limits. Shared code,
tests, CI settings and unknown paths select every method. Identical supported
lock inputs share a whole-suite job; every selected container remains separate.

The workflow is itself under test (`tests/test_ci.py`): **a CI job that cannot
fail is worse than no CI**, so `continue-on-error`, `|| true`, `set +e` and
`if: always()` are refused, and the actions are pinned to commit SHAs rather
than tags.

**Decide by exit status.** Grepping the output for a success string misses
failures.

```bash
./tests/run-tests.sh; echo "EXIT=$?"
```

Once per clone, so that the pre-commit hook is active:

```bash
git config core.hooksPath .githooks
```

After committing, validate and push with `python3 bin/validate-push.py push --remote origin`.
The long Torch suite runs locally before SSH starts; the pre-push hook checks
its saved evidence without rerunning tests. Unchanged network retries reuse
that evidence. See [offline push validation](docs/PUSH_VALIDATION.md) for
interpreter selection, logs, expiration and validation limits.

## Training a method

The following example trains Context Prediction and checks its outputs. Each step is checked by
`tests/test_method_context_prediction.py`, which runs exactly this sequence
on synthetic images.

**1. Build the environment.** Only the training needs packages; the tools
below need nothing installed.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --require-hashes --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r methods/context_prediction/requirements.lock.txt -r requirements-tools.lock.txt
```

Swap the index for a CUDA one (`.../whl/cu121`) to get a GPU build at the same
versions. See the method's README for what that does and does not guarantee.

`requirements-tools.lock.txt` is PyYAML and nothing else, used by the configuration resolver to read a YAML authoring config. Omit it if your config is JSON. It is not a
method dependency, and no method declares it.

**2. Resolve the config**, so the run is identified by a hash rather than by a
file somebody may edit afterwards:

```bash
python3 bin/resolve-config.py --config methods/context_prediction/configs/pretrain.yaml --out runs/ctxpred/resolved.json --set DATA_ROOT=/path/to/ILSVRC2012
```

**3. Run the adapter.** It runs from the method's directory, and reaches this
repository through `PYTHONPATH`:

```bash
cd methods/context_prediction && PYTHONPATH=../.. python3 -m adapter --config ../../runs/ctxpred/resolved.json --out ../../runs/ctxpred/out; echo "EXIT=$?"
```

**4. Check the result against the contract**, passing the adapter's exit
status so that both signals have to agree:

```bash
python3 bin/contract-test.py --out runs/ctxpred/out --config runs/ctxpred/resolved.json --exit-status <adapter-exit-status>
```

`runs/ctxpred/out` then holds `encoder.pt`, `metrics.json`,
`run_manifest.json`, and a `work/` directory with the training run's own
checkpoints and logs. Nothing is written outside it.

The method's own README covers what it implements, which settings it uses and
where each came from, and what changed during the port:
[methods/context_prediction/README.md](methods/context_prediction/README.md).

## One command for a whole run

The steps above are what the launcher does, in order, so that nobody has to
remember the order:

```bash
python3 bin/launch.py --config methods/context_prediction/configs/pretrain.yaml --method context_prediction --set DATA_ROOT=/path/to/ILSVRC2012
```

It resolves the config, submits the job through the chosen platform, verifies
the outputs against the contract, and records the invocation. Three decisions
worth knowing:

- **The run directory is named after the configuration**, not the clock:
  `runs/<method>-<config_sha256[:12]>/`. Two runs of one experiment then
  collide, which is information rather than a nuisance — you meant to change
  something and did not. `--again` repeats one deliberately
- **`--gpus` and `--hours` are launcher arguments, not config keys.** How long
  a scheduler is asked to allow does not change the result, and folding it
  into the config would make two identical experiments hash differently. What
  *does* affect the result, `WORLD_SIZE`, is recorded by the run itself
- **A submitted job is not a finished one.** Where the platform can report how
  it went, the launcher verifies immediately; where it can only queue the work
  it says `submitted` and stops, rather than checking an output directory
  nothing has written yet. `--verify-only <run-dir>` finishes the job later
- **The distribution variables are set, not inherited.** `WORLD_SIZE`, `RANK`
  and `LOCAL_RANK` are stated explicitly, because `adapterlib` records
  `WORLD_SIZE` in the manifest and a value left over in your shell would
  otherwise be written into the results. Multi-process fan-out is **not
  implemented**: `--processes` above 1 is refused rather than approximated,
  and `--gpus` is a resource request that does not imply a process count

Alongside the outputs it writes `launch.json`: the authoring config, the
substitutions, the platform, the resources and the verdict. The manifest says
what the run *did*; this says what was *asked of it*.

```
runs/_reference-dd6145f9edb6/
├── launch.json          what was asked, and how it turned out
├── resolved.json        the exact config that ran
└── out/                 encoder.pt, metrics.json, run_manifest.json
```

## Running CI here

CI runs on GitHub. When it cannot — the account hit a billing wall on
2026-07-31 — the same jobs can be run on this machine, provided Docker is
available:

```bash
python3 bin/run-ci-locally.py --event pull_request
```

**It reads `.github/workflows/tests.yml` and executes what it finds.** It
contains none of the workflow's commands, and `tests/test_run_ci_locally.py`
asserts that: a script that restated them would be a second implementation of
the workflow, and the two would agree until the day one was edited — at which
point "CI passed locally" would be false with nothing able to notice.

Each matrix job gets its own export of `HEAD`, so a run says something about
the committed tree rather than about whatever is open in an editor. The default
workflow also comes from `HEAD`; `--workflow` explicitly overrides it with a file.

What it cannot reproduce, it prints:

- the image is not GitHub's runner image
- `linux/amd64` is emulated here. Results have matched, but it is not the same
  silicon, and this repository already says agreement across different
  hardware is not guaranteed
- `uses:` steps are actions, not shell; their effect is provided differently
  and noted rather than executed

`--dry-run` exports the committed sources once, executes the workflow's discovery
steps and prints every resolved matrix row without running row commands. It does
not create unused per-row exports. Real execution still uses an independent
source tree for each row.

## Checking an adapter's output against the contract

**`contract-test` is how "the port is finished" is decided by a machine**
rather than by opinion.

```bash
python3 bin/contract-test.py --out <dir> --config runs/demo/resolved.json --exit-status <n>
```

**Success requires two signals to agree:** exit status 0 *and* `status: "ok"`
in `run_manifest.json`. Neither is trusted alone — on the Capture side a gate
once returned exit 0 while reporting detected secrets.

The tool also refuses any file in `--out` that the manifest does not list. An
output nobody knows about is a hole in reproducibility.

## What may be compared with what

`metrics.json` carries two blocks:

```json
{"schema_version": 2,
 "metrics": {"final_linear_probe_top1_accuracy": 42.3},
 "metrics_raw": {"final_top1_acc": 42.3}}
```

`metrics` uses a **fixed vocabulary**; `metrics_raw` keeps the names the
original gave its own numbers, so nothing is lost in translation. Both are
required, and `contract-test` reads the vocabulary from the same place the
adapters write it, so the two cannot drift apart.

The distinction the vocabulary exists for is `pretext` against
`linear_probe`:

| Prefix | What it measures | Comparable across methods |
|---|---|---|
| `pretext` | the method's **own** training objective or task | **No** |
| `linear_probe` | downstream classification from a linear probe, against real labels | Interpret within the specified dataset, metric and readout protocol |
| `epochs_completed`, `steps_completed` | counters | in kind only |

The first port's `val_acc1` is eight-way patch-position accuracy — its own
pretext task. A linear probe's top-1 is real classification accuracy. Putting
those in one column produces a comparison table that is wrong and looks right,
so **the stage decides which family a port may use**, and reaching across is
refused rather than discouraged. Accuracies are percentages, 0 to 100, which
was measured from both sources rather than assumed.

## What reproducibility means here

The operating checks compare artifacts produced with a fixed configuration and
software environment. For example, a Context Prediction test runs the same small
training configuration twice and checks the resulting artifact hashes. This
tests repeatability of that path; the [figure guide](docs/PAPER_REPRODUCTION_GUIDE.md)
records which full-data outcomes have been verified.

Hardware, kernels and floating-point arithmetic can affect results. Compare
the environment and package versions recorded in the manifests when runs differ.

And the record is checked, not merely kept. `bin/verify-environment.py`
answers both halves with one comparison — *is this environment the locked
one*, and *did that run use it*:

```bash
python3 bin/verify-environment.py --lock methods/context_prediction/requirements.lock.txt --lock requirements-tools.lock.txt --manifest runs/ctxpred/out/run_manifest.json
```

Any difference fails, including a package no lock mentions: something
installed that the lock does not describe means the environment cannot be
rebuilt from it. The one exemption is `pip`, which `python -m venv` seeds
before any lock is read — and it is *reported* as ignored, with the reason,
rather than passed over.

**Measured, on 2026-07-30.** The container was built and run on both
architectures Docker offers here, and the chain completed on each:

```
linux/arm64   same config twice -> encoder.pt bd90b98ef87dea0265bb…  (identical)
linux/amd64   same config twice -> encoder.pt cbf177068c3dcd6583f5…  (identical)
```

Each architecture produced identical hashes across its two runs, while the
hashes differed between architectures. The manifests identify the machines as
`aarch64` and `x86_64`. This July 2026 CPU check did not exercise Apptainer or
CUDA determinism.

## Execution platforms

**Running on any particular compute facility is optional.** The default is
your own machine (`platforms/local`), and that path is complete on its own.
Platform support lives in loosely coupled modules, and the core assumes none
of them.

The separation is **held by machinery** —
`tests/test_platform_isolation.py`. See [docs/PLATFORMS.md](docs/PLATFORMS.md).

## Where the design lives

For the mapping from a paper experiment to its implementation and available
evidence, use the [reproduction guide](docs/PAPER_REPRODUCTION_GUIDE.md).
The [adapter output checker](bin/contract-test.py) validates an individual
run. The [third-party inventory](docs/THIRD_PARTY_LICENSES.md) records the
upstream source and license information.

## Guards against repeated mistakes

Seven kinds of mistake recurred often enough here to be worth mechanising
rather than remembering. They are listed with their counts in
[CLAUDE.md](CLAUDE.md); the mechanisms are:

| Mechanism | What it prevents |
|---|---|
| `bin/mutate.py` | assertions that cannot fail, and mutation reports that lie. An absent or ambiguous anchor is an error, bytecode is never reused, and it refuses to mutate anything outside its own copy |
| `tests/test_no_hard_coded_methods.py` | shared machinery naming one method. A list looks right until the second method arrives |
| `tests/test_repo_hygiene.py` | the README layout drifting from the tree — in **both** directions |
| `tests/test_ci.py` | a CI job that cannot fail |
| `tests/test_language.py`, `tests/test_platform_isolation.py` | prose and platform vocabulary leaking where they do not belong |
| `tests/test_repository_scan.py` | the same rule implemented twice. Two copies of "which files belong to this repository" agreed everywhere git existed and diverged inside the container image. It pins one implementation, and proves the scan works with git actually removed from `PATH` |

Each was written after the same mistake had been made two or three times. The
counts are in the commit history and are not flattering; they are recorded
because a mistake nobody counted is a mistake that recurs.

The last row was added after the fourth occurrence, in the commit that
introduced the row above it -- which is the honest measure of how little a
written rule achieves on its own. What the guards do not cover, the container
job does: it runs the suite in an image with no `.git`, no `.github` and no
git binary, which is the closest thing here to what a reader downloads.

## Development

See the [contributor instructions](AGENTS.md) and [development handoff](docs/HANDOFF.md).
The [archive guide](docs/SUBMISSION_ARCHIVE.md) explains how to build a reviewer ZIP with its own README.

Strict TDD. The rules are in [CLAUDE.md](CLAUDE.md). Everything in this
repository is written in English, enforced by `tests/test_language.py`.

## Extended task implementations

For the 40 settings plotted in the paper, start with the [Figure 5 dataset and
metric ledger](docs/CVPR_EXTENDED_COVERAGE.md). The following guides also cover
tasks from earlier experiment catalogs.

Extended detection LP/AP training, continuation and official evaluation are described in
[the detection guide](docs/EXTENDED_DETECTION.md); released-weight/full-data validation and paper-score reproduction remain pending.

[Extended optical flow](docs/EXTENDED_FLOW.md) adds five-provider LP/AP execution,
Sintel/Middlebury split conversion, explicit Spring inputs, pixel-coordinate
metrics and distributed continuation. Source/companion head and loss conflicts
remain explicit; this component port does not certify paper scores.

[Extended NYUv2 LP/AP](docs/EXTENDED_DEPTH.md) connects official split-file input,
five spatial providers, depth training, distributed continuation and unaligned
pixel-pooled evaluation. Its head/loss differ from the Basic Five component path;
released-weight and paper-score verification remain pending.

[Extended structured tasks](docs/EXTENDED_STRUCTURED.md) add MPII/CrowdPose heatmaps,
7-Scenes/Cambridge absolute localization and 3DSRBench question-conditioned LP/AP,
with native annotation conversion and portable continuation. Localization recipe
conflicts and local reasoning holdout semantics remain explicit; these components
do not certify final-table scores.

The [DAVIS first-mask LP/AP path](docs/EXTENDED_VOS.md) now includes native
sequence membership, object-mask training/evaluation and portable continuation.
[Context and Mapillary converters](docs/EXTENDED_SEMANTIC_INPUTS.md) connect native
annotations to the existing semantic runner. Component support is distinct from
full-data paper-score reproduction.

[Native image staging](docs/EXTENDED_NATIVE_IMAGES.md) also supports prepared
Action40, CIFAR-10/100, KMNIST, ImageNet 1%/10% and Omniglot15 memberships;
upstream authentication and small-image recipe reconciliation remain pending.
Its CLUE and SUN397 profiles connect six further
native image inputs to Extended LP/AP. Source split, label order and overlap
limitations are recorded explicitly.
Flowers102 now stages the recorded `trnid`-only profile, separately from the
registry's train+validation declaration. Kuzushiji-Kanji stages the observed
1,966-class prepared benchmark without inventing a split; historical input-byte
identity and transform attribution remain unverified.

[Historical Extended accounting](docs/REFERENCE_ACCOUNTING.md) audits pinned
native LP/AP results across cells, retaining missing/invalid repeats and explicit
final-epoch evidence. Artifact consistency is separate from paper-score verification.

## Security maintenance

See [the 2026-10-24 dependency update](SECURITY_UPDATE_2026-10-24.md) before rebuilding environments.
