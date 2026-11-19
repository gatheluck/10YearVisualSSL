"""Explicit revised image/dense adaptation recipes and update-based scheduling.

The reference horizon is retained even for a diagnostic run stopped early.
Recipe identity does not certify correspondence to a particular paper result.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RevisedRecipe:
    profile: str
    source_protocol_id: str
    task: str
    track: str
    epochs: int
    effective_batch: int
    reference_batch: int
    head_lr: float
    adapter_lr: float | None
    warmup_epochs: int
    weight_decay: float = .05
    betas: tuple[float, float] = (.9, .999)
    clip_norm: float = 1.
    rank: int | None = 16
    alpha: float | None = 16.
    min_lr: float = 1e-6
    start_lr: float = 1e-6
    dpt_hidden: int | None = None


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def resolve_recipe(task: str, track: str, *, effective_batch=None) -> RevisedRecipe:
    """Resolve only inspected routes; changed batch sizes remain explicit.

    All rates scale with effective batch, but the warmup start and cosine floor
    are absolute. Epoch overrides are intentionally not accepted here.
    """
    settings = {
        ('imagenet', 'lora'): (100, 256, .001, .000125, 5),
        ('ade20k', 'lora'): (20, 8, .001, .0001, 1),
        ('ade20k', 'dpt'): (20, 8, .0005, None, 1),
    }
    if (task, track) not in settings:
        raise ValueError(f'Unconfirmed revised task/track: {task}/{track}')
    epochs, reference_batch, head, adapter, warmup = settings[task, track]
    batch = reference_batch if effective_batch is None else effective_batch
    _integer(batch, 'effective_batch', 1)
    scale = batch / reference_batch
    dense_probe = track == 'dpt'
    return RevisedRecipe(
        profile='b5x_lora_dpt_v2',
        source_protocol_id='B5X_DPT_LP_v2' if dense_probe else 'B5X_LORA_LP_v2',
        task=task, track=track, epochs=epochs, effective_batch=batch,
        reference_batch=reference_batch, head_lr=head * scale,
        adapter_lr=None if adapter is None else adapter * scale,
        warmup_epochs=warmup, rank=None if dense_probe else 16,
        alpha=None if dense_probe else 16., dpt_hidden=512 if dense_probe else None,
    )


def cosine_lr(step: int, total: int, warmup: int, base_lr: float,
              min_lr: float = 1e-6, start_lr: float = 1e-6) -> float:
    """LR before a zero-based update; the floor is reached at ``step=total``.

    Invalid horizons fail rather than silently returning an unscheduled LR.
    Steps beyond a completed horizon stay at the floor, as in the reference.
    """
    _integer(step, 'step', 0)
    _integer(total, 'total', 1)
    _integer(warmup, 'warmup', 0)
    if warmup >= total:
        raise ValueError('warmup must be shorter than the reference horizon')
    if any(not math.isfinite(value) or value < 0
           for value in (base_lr, min_lr, start_lr)) or base_lr == 0:
        raise ValueError('Learning rates must be finite, with positive base_lr')
    if step < warmup:
        return start_lr + (base_lr - start_lr) * float(step) / warmup
    progress = min(1., (step - warmup) / (total - warmup))
    return min_lr + (base_lr - min_lr) * .5 * (1. + math.cos(math.pi * progress))


def set_learning_rates(optimizer, recipe: RevisedRecipe, step: int,
                       steps_per_epoch: int) -> None:
    """Set absolute per-group rates without changing their original base rates.

    Validate all groups before assigning any rate so malformed optimizer state
    cannot leave an otherwise valid group partially updated.
    """
    _integer(steps_per_epoch, 'steps_per_epoch', 1)
    total = recipe.epochs * steps_per_epoch
    warmup = recipe.warmup_epochs * steps_per_epoch
    rates = [cosine_lr(step, total, warmup, group['base_lr'],
                       recipe.min_lr, recipe.start_lr)
             for group in optimizer.param_groups]
    for group, rate in zip(optimizer.param_groups, rates):
        group['lr'] = rate
