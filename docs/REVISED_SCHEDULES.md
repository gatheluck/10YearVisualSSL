# Explicit short LoRA schedules

See [package scope and replication terminology](SUBMISSION_SCOPE.md) for the
distinction between original experiments and portable component validation.

The default revised CLI schedules remain ImageNet 100 epochs and SSv2 50 epochs.
The author-supplied revised LoRA protocol additionally specifies two
compute-limited schedules for the released 7B image-encoder provider:

| Task | Explicit `schedule` | Epochs | Warm-up |
| --- | --- | --- | --- |
| `imagenet` | `B5X_LORA_LP_50EP_v2` | 50 | 5 epochs |
| `ssv2` | `B5X_LORA_LP_24EP_v2` | 24 | 5 epochs |

Add the applicable `schedule` string to a normal
[`downstream.revised_tasks` configuration](CVPR_ADAPTATION.md). Both require
`track: "lora"`, the matching provider, `arch: "released"`, and an explicit local
checkpoint. Provider loading still verifies the actual released architecture
and checkpoint. Other providers, tasks, tracks and reduced fixture profiles
cannot claim these named schedules. Omit `schedule` to retain the existing
reference horizon; an explicit null or empty name is invalid.

These are complete named schedules, not diagnostic early stopping. Warm-up and
cosine decay use the selected 50- or 24-epoch horizon. Other recipe parameters
remain unchanged, including effective-batch scaling, optimizer, head and adapter
rates, rank, clipping, weight decay and input processing. `diagnostic.epochs`
may truncate a diagnostic run but cannot exceed the selected horizon and does
not rescale the underlying cosine schedule.

Results record the selected horizon and protocol identity in `recipe`.
Portable completed-epoch continuation binds both configuration and recipe;
changing the schedule, horizon or protocol identity rejects resume before
restoring training state. This does not import native author checkpoints or
claim native resumed RNG equivalence.

The supplied protocol specifies these schedules, and inspected original run
metadata contains their matching IDs, horizons and five-epoch warm-up for three
seeds each. Original resumed logs contain only portions of those horizons; the
metadata alone is not independent evidence that all epochs or final scores were
reproduced here. Activation-checkpointing reports also differ between tasks
and must be considered separately from this schedule selection.

Validation checks source ID/horizon, scheduler endpoint, invalid provider/task
selection, diagnostic limits, result recording with a synthetic model and
continuation identity rejection. No released model training or paper score is
established by these CPU tests.
