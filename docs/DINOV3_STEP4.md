# DINOv3 Step-4 components

> **Legacy terminology:** numbered Step 1-4 labels in this file are historical
> code/experiment identifiers, not manuscript section numbers. See the
> [ASIS, CTRL and manuscript comparison map](PAPER_TERMINOLOGY.md) before matching a recipe to a paper result.

Status, support and validation statements describe this portable package at the
date recorded; see [scope and terminology](SUBMISSION_SCOPE.md)
for the distinction from original experimental implementations and results.

Status: 2026-10-14. This implements selectable projection layouts, Gram-stage
training and an explicit distributed component entry. It does **not** establish canonical
paper scores or complete distributed reproduction. The default remains the
previous `core` training profile and independent heads.

## Projection layouts

Set `train.head_layout` in the resolved pretraining JSON:

| Value | Projection ownership | Experiment mapping |
| --- | --- | --- |
| `separate` (default) | Independent DINO and iBOT heads | Original |
| `shared` | One MLP and prototype layer for both token types | H1 |
| `shared_prototypes` | Separate MLPs, one prototype layer | H1P |
| `shared_mlp` | One MLP, separate prototype layers | H1M |
| `token_affine` | Separate identity-initialized per-channel affine transforms before one head | H1TA |

Shared components are registered once, updated once by the optimizer and EMA,
and copied independently into the teacher. Token affine transforms affect only
head inputs; KoLeo and Gram receive the original backbone features. Shared
heads require matching hidden, bottleneck and output dimensions. H1P permits
different hidden dimensions; H1M permits different prototype counts. Invalid
layouts or incompatible dimensions fail. Compact H1P checkpoint keys deliberately
omit the reference implementation's duplicate aliases. The reference's repeated
prototype initialization is preserved explicitly, including its random-number
consumption; removing those apparently redundant draws changes seeded runs.
Native full training checkpoints are not directly interchangeable. Backbone exports remain compatible.

`train.dino_loss_weight` and `train.ibot_loss_weight` default to 1. They must be
finite and nonnegative. The existing head dimensions and mask bounds remain
configurable. The 2026-09-25 manuscript Tables 34–35 and captured active configs
support the following settings; this is a configuration mapping, not new scores:

| H1 trial | Change from fully shared baseline |
| --- | --- |
| H1-00 | DINO/iBOT weights 1/1, prototypes 65536, bottleneck 256, mask 0.10–0.50 |
| H1-01 / 02 | Weights 1.25/0.75 or 0.75/1.25 |
| H1-03 / 04 | Weights 1.5/0.5 or 0.5/1.5 |
| H1-05 / 06 | Both prototype counts 32768 or 131072 |
| H1-07 / 08 | Both bottleneck dimensions 128 or 512 |
| H1-09 | Mask upper bound 0.65 |

The later H1-09 result changes masking as well as sharing; it must not be
labelled a head-only ablation. Earlier workbook values are historical results,
not evidence of a newly reproduced run.

## Explicit Gram profile

Merge these overrides into `train` in an otherwise resolved pretraining config:

```json
{
  "head_layout": "shared",
  "training_profile": "step4_gram_components",
  "dino_loss_weight": 1.0,
  "ibot_loss_weight": 1.0,
  "lr": 0.0006,
  "min_lr": 0.000001,
  "warmup_epochs": 10,
  "teacher_momentum_start": 0.994,
  "teacher_momentum_end": 0.999,
  "teacher_temp_start": 0.04,
  "teacher_temp_end": 0.07,
  "teacher_temp_warmup_epochs": 25
}
```

For example, after resolving `methods/31_dinov3/configs/pretrain.yaml` with the
existing resolver, save that JSON as `resolved.json`, then run:

```bash
python -c 'import json; from pathlib import Path; p=Path("resolved.json"); c=json.loads(p.read_text()); c["train"].update(head_layout="shared", training_profile="step4_gram_components", dino_loss_weight=1.0, ibot_loss_weight=1.0, lr=0.0006, min_lr=0.000001, warmup_epochs=10, teacher_momentum_start=0.994, teacher_momentum_end=0.999, teacher_temp_start=0.04, teacher_temp_end=0.07, teacher_temp_warmup_epochs=25); Path("step4.json").write_text(json.dumps(c, indent=2)+"\n")'
```

Pass the absolute `step4.json` path to the existing `python -m adapter` invocation
in the [method guide](../methods/31_dinov3/README.md). Use a new output directory.
The executable documentation test merges the JSON block into a tiny training
fixture and verifies the saved profile, layout and noncanonical status.

This profile selects the fixed optimizer-step schedule in
`methods/31_dinov3/protocol.py`: 300-epoch clock, 10-epoch LR warmup from zero
to 0.0006 then decay to 0.000001, 25-epoch teacher-temperature warmup from 0.04
to 0.07, teacher EMA 0.994 before Gram and 0.999 during Gram. The profile rejects
LR, temperature and EMA settings that disagree with these values before
model construction. The adapter may create its diagnostic output directory
before reporting an invalid configuration. Shortening `epochs` truncates this clock, rather than compressing it.
Batch size, weight decay, masks, head dimensions and loss coefficients remain
configurable. This is a component runner, not a strict canonical-config validator.

After epoch 250 the EMA backbone is snapshotted into a separate frozen Gram
teacher. Epochs 251–300 match student masked-patch Gram matrices against that
teacher's undistorted companion crops with identical geometry. After epochs
253, 255 and 258 it refreshes from the EMA backbone. Gram weight rises from
0 to 2 and the DINO local-crop weight falls from 1 to 0.5 over the first quarter
of epoch 251. The implementation uses the existing schedule and loss components.
The checkpoint records the optimizer state, optimizer step, steps per epoch,
Gram teacher (once created), profile and `canonical_eligible: false`.

## Verification and remaining work

CPU tests cover both task-gradient routes, parameter ownership, EMA, strict
checkpoint round trips, unchanged default state initialization, weighted loss
and adapter/encoder contracts. Reduced comparisons against captured H1/H1P/H1M/
H1TA code matched seeded initialization, outputs, input/parameter gradients
and three SGD/EMA updates
exactly on identical inputs and weights. These comparisons concern heads,
not the entire distributed training pipeline.

An actual three-epoch tiny run compresses only the stage boundaries in the test
and verifies snapshot, refresh, frozen state, clean-crop targets and Gram's
contribution to the total loss. It does not execute 300 epochs or use ImageNet.
The legacy adapter uses float32. The explicit distributed entry below integrates
global Sinkhorn and CUDA BF16 execution; the 8-GPU physical batch and CUDA
numerics have not been measured in this change. No new GPU or paper
accuracy result is claimed.

## Continuation and joint assignment components

The following overrides select three additional profiles. Merge
one object into `train`, together with the fixed schedule settings above:

```json
[
  {"training_profile": "step4_core_components", "head_layout": "shared"},
  {"training_profile": "step4_split_components", "head_layout": "shared"},
  {"training_profile": "step4_joint_components", "head_layout": "shared"}
]
```

- H1CORE keeps EMA at 0.994, local DINO weight at 1 and Gram disabled for the
  whole 300-epoch clock. LR and teacher temperature keep the same fixed clock.
- H1S200 clones the student and teacher heads independently after epoch 200,
  duplicates AdamW step/first/second moments, and preserves the backbone and
  optimizer ordering. Subsequent iBOT updates use the independent head. Gram
  still starts after epoch 250.
- H1JA solves one joint weighted Sinkhorn problem with CLS/patch masses 0.5/0.5
  and uniform prototype mass. Per-token probabilities feed the existing losses
  without a second Sinkhorn. An empty group transfers its mass to the other.
  Unequal/empty ranks are covered by a real three-process CPU Gloo test, although
  integrated multi-rank training is described below.

All component profiles now save version-1 full checkpoints, including AdamW,
epoch/step, head layout, frozen Gram teacher when applicable, Python/NumPy/Torch
RNG state and the data-loader generator. `epoch` is zero-based in this format;
`epochs` is the desired total completed epoch count, not extra epochs to run.
For example, branch a shared Gram component checkpoint after epoch 200:

```json
{
  "training_profile": "step4_split_components",
  "head_layout": "shared",
  "resume_checkpoint": "/path/to/component_epoch200.pth"
}
```

Merge this into `train`, retain the original settings and increase `epochs` to
the target, then run the adapter with a **new** output directory. Same-profile
resumption is supported at completed epochs. Switching from the shared Gram
profile to H1CORE or H1S200 is allowed only after epoch 200. Source checkpoints
are read-only inputs. Data, model, loss, batch, seed and schedule configuration
must match; only output location, target epochs and checkpoint milestones may
change. Use the same immutable dataset: matching paths and loader length do not
verify dataset content hashes. A device-type change is refused.

Reduced CPU runs match uninterrupted versus resumed weights and final losses
exactly, including a Gram refresh boundary and the shared-to-separate split.
Reference comparisons match joint probabilities and three post-split AdamW/EMA
updates exactly. This is component parity, not paper-score reproduction.
The historical blanket resume refusal is superseded for version-1 component
checkpoints only: legacy core, earlier incomplete component checkpoints,
`encoder.pt` and native distributed checkpoints remain unsupported inputs to
this portable resume interface.

Not yet validated in this port: full-scale CUDA BF16 training and canonical
evaluation. Native full training-checkpoint continuation remains separate from
the portable continuation format. The inspected experimental implementations
are distinct from this portable component path.
See the [prioritized gap ledger](PAPER_REPRODUCTION_GAPS.md).


## Explicit distributed execution

The final submitted manuscript's Table 15 head-design experiments can use the
same five projection layouts and Gram, H1CORE, H1S200 and H1JA component profiles
through a dedicated `torchrun` entry. This closes the execution integration gap;
it does not establish the table's accuracy values or certify a canonical recipe.
The legacy adapter remains single-process FP32 and refuses multi-rank launches.

Add this top-level object to the resolved JSON (alongside `train`, not inside it):

```json
{"execution": {"precision": "float32"}}
```

For the captured GPU precision, explicitly change `precision` to `bfloat16` and
`device` to `cuda`. BF16 requires a capable CUDA device; CPU BF16 and FP16 are
refused. Model forwards, including the frozen Gram teacher, run under autocast;
objectives, stored parameters and AdamW state remain FP32. No gradient scaler
or accumulation is introduced. A single-node example from the repository root:

```bash
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 methods/31_dinov3/dinov3_distributed.py --config /path/to/resolved-distributed.json --out /path/to/new-run
```

Use a free rendezvous port and only allocated compute resources. Multi-node
launches must supply the usual matching node count, node rank and rendezvous
address. `train.batch_size` is **per rank**: the captured physical batch of 1024
uses 8 ranks of 128. Smaller explicit batches test components, not the original
experimental condition. The entry requires a Step-4 component profile; legacy
`core` is refused. All ranks must see the same immutable dataset and config.

Rank seeds are `seed + rank`. Student parameters and buffers synchronize before
the teacher is copied. DDP averages gradients; the distributed sampler uses its
reference default seed and advances each epoch. Class and patch Sinkhorn run
across ranks, while KoLeo remains local to each global crop on each rank.
An empty patch rank still participates in patch Sinkhorn: this deliberately
corrects an early-return deadlock risk in the inspected reference without
changing nonempty-rank loss semantics. Empty loaders and nonfinite rank losses
fail; no ordinary-queue fallback or remote submission is performed by this CLI.

Rank zero writes checkpoints, teacher encoder exports, metrics and a hashed
manifest. Existing output directories are refused. Checkpoints additionally
record world size, precision and each rank's RNG state. Resume requires this
same execution entry, world size, precision, data and configuration; it refuses
legacy single-process and native training checkpoints. Existing portable
single-process continuation remains unchanged. The H1S200 boundary reconstructs
DDP after adding the independent head and preserves cloned optimizer moments.
Matching dataset paths and batch counts do not authenticate dataset contents.

Validation: bounded two-rank CPU tests exercise all five layouts and four
profiles, live student/teacher synchronization, explicit mean-gradient parity,
rank-zero delivery, empty patch ranks, and exact uninterrupted/resumed weights
across compressed split and Gram boundaries. CLI artifacts pass the downstream
contract, and a second invocation leaves existing output bytes unchanged.
Two-rank comparisons against the refreshed original loss and initialization
functions match teacher state, losses, gradients and three AdamW updates exactly,
including unequal positive patch counts. The 20 inspected original source/config
files matched the Capture snapshot. Private evidence remains outside Git.
CUDA BF16, an 8-rank full-size run, multi-node transport and paper-score
reproduction remain unverified. CPU autocast instrumentation checks placement
only and is not GPU numerical validation.
