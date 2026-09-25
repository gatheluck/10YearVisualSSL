# DINOv3 Step-4 components

Status: 2026-09-25. This implements selectable projection layouts and a
single-process Gram-stage training path. It does **not** establish canonical
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
The runner uses float32; the captured CUDA BF16 path, 8-rank physical batch and
global Sinkhorn scope remain unverified/unintegrated here. No new GPU or paper
accuracy result is claimed.

## Continuation and joint assignment components

The following overrides select three additional single-process profiles. Merge
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
  the training entry point itself remains single-process.

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
`encoder.pt` and native distributed checkpoints remain unsupported.

Still pending: full distributed training, CUDA BF16 and canonical evaluation.
See the [prioritized gap ledger](PAPER_REPRODUCTION_GAPS.md).
