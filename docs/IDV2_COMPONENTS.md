# IDv2 Step-4 components

> **Legacy terminology:** numbered Step 1-4 labels in this file are historical
> code/experiment identifiers, not manuscript section numbers. See the
> [ASIS, CTRL and manuscript comparison map](PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Added 2026-09-25. Seven explicit profiles extend the existing Instance
Discrimination ViT adapter. The default single-view and ResNet paths remain
unchanged. These profiles implement FP32 components, with an explicit DDP entry
added on 2026-10-11. Neither entry claims canonical ImageNet training or
reproduction of any paper/workbook score.

| Profile suffix (between `idv2_` and `_components`) | Behavior |
| --- | --- |
| `twoview` | Two independent global augmentations; independently sampled negatives; average NCE; one update from their normalized mean |
| `false_negative` | Two-view baseline through zero-based epoch 19; from epoch 20 sample m+5 non-self candidates and exclude the five highest similarities, preserving retained order |
| `ema_bank` | Two-view objective, bank momentum 0.99; no teacher encoder |
| `koleo` | Two-view NCE plus 0.1 times mean per-view KoLeo; nearest neighbors never mix views |
| `multicrop` | Two globals and four locals; one shared bank lookup; six sequential partition updates and gradient contributions; bank uses only the globals |
| `multi_prototype` | Four slots per image; maximum initialized-slot score, random-slot fallback only when all empty; each view separately fills the next slot or updates the nearest filled slot |
| `dense_id` | Two globals plus four randomly selected distinct patch tokens per view through the existing projector; patch NCE weight 0.1; reuse each global lookup and the partition after both global calls |

All profiles make one AdamW update per source batch. Duplicate image IDs are
coalesced before bank updates (per view for multi-prototype). Patches and local
views never update the image bank. Global crops use scale [0.2, 1.0]; local crops
use [0.05, 0.32]. Photometric transforms retain the existing InstDisc ordering.
Multicrop uses timm dynamic position interpolation without adding parameters.
Dense ID also adds no model parameters. Encoder export retains only `encoder.*`.

## Resolved configuration examples

Save one object from the following array as `resolved.json`, replace data_root,
and use a method environment installed from the existing lock. These reference
sizes/batches require substantial memory, especially the four-slot bank. Tiny
CPU fixtures validate components; they do not validate these full-size runs.

```json
[
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_twoview_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_false_negative_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_ema_bank_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.99,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_koleo_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_multicrop_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ],
      "local_size": 96
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_multi_prototype_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  },
  {
    "stage": "pretrain",
    "seed": 42,
    "device": "cuda",
    "data_root": "/path/to/imagenet",
    "train": {
      "arch": "vit",
      "profile": "idv2_dense_id_components",
      "feature_dim": 128,
      "img_size": 224,
      "patch_size": 16,
      "embed_dim": 768,
      "depth": 12,
      "num_heads": 12,
      "mlp_ratio": 4.0,
      "drop_rate": 0.0,
      "attn_drop_rate": 0.0,
      "temperature": 0.07,
      "nce_momentum": 0.5,
      "num_negatives": 4096,
      "epochs": 300,
      "batch_size": 1024,
      "num_workers": 8,
      "lr": 0.0006,
      "weight_decay": 0.05,
      "warmup_epochs": 10,
      "min_lr": 0.0,
      "save_at_epochs": [
        100,
        200,
        300
      ]
    }
  }
]
```

From `methods/10_inst_disc`, run `PYTHONPATH=../.. python -m adapter --config /path/to/resolved.json --out /path/to/new-run`.
The input root contains `train/CLASS/IMAGE`; class labels are unused in pretraining.

`epochs` is the desired completed-epoch endpoint (1..300), while the LR schedule
always uses the 300-epoch reference horizon. Reducing it stops early without
compressing the schedule. The initial warmup LR has the captured 1e-6 floor.

## Resume and artifacts

Set `train.resume_checkpoint` to another run's full
`work/checkpoint_latest.pth`, choose a fresh output directory, and increase
`train.epochs`. Checkpoint `epoch` is zero-based. Same-profile, epoch-boundary
continuation restores model, AdamW, the complete NCE state (memory, Z, and all
prototype masks/counters), and Python/NumPy/Torch/loader-generator RNG state.
CUDA RNG state is saved/restored on CUDA, but GPU continuation has not been
empirically verified in this change. The ordinary single-process adapter accepts only version-1 component checkpoints:
legacy checkpoints, native distributed checkpoints and encoder-only files are
not supported inputs. A missing source, incompatible configuration/device/loader
length, incomplete optimizer, or source within the output directory is refused.
Checkpoint loading is for trusted local inputs only.

The dataset configuration and loader length must match. Data content hashes
are not authenticated; keep the input files and their ordering immutable.
The ordinary adapter still refuses distributed launches; use the explicit entry
below. Native historical-checkpoint import, BF16, full-data run aggregation and
paper-score validation remain outside this port's verified coverage. Checkpoints explicitly carry
`canonical_eligible: false`. Do not use an output-contract pass as evidence of
canonical experimental equivalence. The default legacy single-view checkpoint
format is unchanged and does not acquire this component resume guarantee.

## Explicit distributed FP32 execution

From `methods/10_inst_disc`, use the same resolved profile configuration, an
explicit `device` of `cpu` or `cuda`, and a new output directory:

`PYTHONPATH=../.. python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=2 --module idv2_distributed --config /path/to/resolved.json --out /path/to/new-run`

This example is for one host; choose a free local port. Multi-node rendezvous
and cluster allocation must be configured separately. No queue or allocation
is selected by this entry. CUDA uses NCCL and `LOCAL_RANK`; CPU uses Gloo.
`train.batch_size` is the **global source-image batch**, must divide evenly by
the world size, and is not multiplied by the number of views. The source
DistributedSampler shuffles with seed 0 and advances its epoch; padding and
per-rank `drop_last=True` retain its behavior. Each rank must have the same
immutable image membership and ordering. Dataset content is not hashed.

All seven profiles synchronize initial bank buffers, compute the partition
estimate from global sums/counts, and gather observations before coalescing
repeated image IDs. Multi-prototype updates retain separate view observations
and broadcast touched slots/masks and counters. KoLeo searches the global
per-view batch but retains only the local tensor's gradient, as in the inspected
implementation; no extra world-size gradient multiplier is inserted. Multicrop
uses `no_sync` for the first five views. Dense token forward calls pass through
DDP so their gradients participate in reduction.

The entry writes `work/checkpoint_latest.pth`, requested epoch checkpoints,
`encoder.pt`, requested epoch encoder exports, raw-only `metrics.json`, `results.json`, and a downstream
`run_manifest.json`. The latter describes component execution, not the ordinary
method adapter's `contract-test` artifact schema. Only rank zero writes. Existing
CLI output directories are refused. Results retain `canonical_eligible: false`.
Configuration/source disagreement, setup/read/save errors and nonfinite losses
are propagated across ranks; an arbitrary process/network failure still depends
on the launcher/backend's failure handling.

Distributed checkpoints have `component_version: 2`, world size and every
rank's Python/NumPy/Torch/CUDA/loader-generator RNG state. Resume requires the same
configuration, device type, world size, complete optimizer state and loader
length, with a larger epoch endpoint and a fresh output location. The ordinary
single-process version-1 checkpoints and native historical checkpoints are not
interchangeable with version 2. Both modes preserve the fixed 300-epoch horizon;
this does not port a distinct 90-epoch scaling recipe.

Portability differences remain explicit: this entry is deterministic FP32,
seeds Python/NumPy and a dedicated loader generator, uses the package's dataset
root layout and checkpoint schema, and omits native telemetry. Reduced CPU
comparisons do not establish original GPU trajectories, released-weight results,
CUDA resume, multi-node operation, or paper/workbook score reproduction.

## Verification boundary

Behavioral RED/GREEN tests cover the seven routes, sampling/update ordering,
prototype lifecycle, dense gradients, source protection and real interrupted
training. Reduced private-reference comparisons cover three optimizer updates
per profile, including gradients and NCE/AdamW state. Run
`OMP_NUM_THREADS=1 PYTHONPATH=tests:. .venvs/downstream/bin/python -m unittest test_idv2_components test_method_10_inst_disc`.
Mutation and full-suite results are recorded in the task PR. Private sources,
run paths and comparison artifacts remain outside Git. Matching this package's outputs to paper/workbook results remains unverified,
and source protocol contradictions remain unresolved. No reported score is
rewritten; these statements concern the port and its evidence review.

The distributed extension compares seven captured objectives with two CPU ranks
and three optimizer updates per profile (loss, gradients, AdamW and bank state).
Repository tests independently compare DDP with explicit mean-gradient reduction
and exercise actual tiny-ViT training, rank-local failure propagation, output
protection and exact interrupted continuation across all seven profiles. Run
`OMP_NUM_THREADS=1 PYTHONPATH=. python -m unittest tests.test_idv2_distributed`.
Capture/current source identities and numerical comparison evidence stay outside
Git. Record Linux, mutations, whole-suite and CI outcomes separately in the PR.
