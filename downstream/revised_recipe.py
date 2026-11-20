"""Explicit revised image/dense adaptation recipes and update-based scheduling.

The reference horizon is retained even for a diagnostic run stopped early.
Recipe identity does not certify correspondence to a particular paper result.
"""

import math
from dataclasses import dataclass, replace


POPULATIONS = {'imagenet': (1281167, 50000), 'ade20k': (20210, 2000),
               'ssv2': (168913, 24777), 'coco': (118287, 5000), 'nyuv2': (795, 654)}


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
    optimizer: str = 'AdamW'
    momentum: float = .9


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def resolve_recipe(task: str, track: str, *, effective_batch=None, schedule=None, backbone=None) -> RevisedRecipe:
    """Resolve only inspected routes; changed batch sizes remain explicit.

    All rates scale with effective batch, but the warmup start and cosine floor
    are absolute. Epoch overrides are intentionally not accepted here.
    """
    settings = {
        ('imagenet', 'lora'): (100, 256, .001, .000125, 5),
        ('ade20k', 'lora'): (20, 8, .001, .0001, 1),
        ('ade20k', 'dpt'): (20, 8, .0005, None, 1),
        ('nyuv2', 'lora'): (30, 8, .001, .0001, 1),
        ('nyuv2', 'dpt'): (30, 8, .0005, None, 1),
        ('ssv2', 'lora'): (50, 256, .001, .0005, 5),
        ('coco', 'lora'): (12, 16, .02, .02, 0),
    }
    if (task, track) not in settings:
        raise ValueError(f'Unconfirmed revised task/track: {task}/{track}')
    epochs, reference_batch, head, adapter, warmup = settings[task, track]
    batch = reference_batch if effective_batch is None else effective_batch
    _integer(batch, 'effective_batch', 1)
    scale = batch / reference_batch
    dense_probe = track == 'dpt'
    recipe = RevisedRecipe(
        profile='b5x_lora_dpt_v2',
        source_protocol_id=('B5X_LORA_LP_v1' if task == 'coco' else
                            'B5X_DPT_LP_v2' if dense_probe else 'B5X_LORA_LP_v2'),
        task=task, track=track, epochs=epochs, effective_batch=batch,
        reference_batch=reference_batch, head_lr=head * scale,
        adapter_lr=None if adapter is None else adapter * scale,
        warmup_epochs=warmup, rank=None if dense_probe else 16,
        alpha=None if dense_probe else 16., dpt_hidden=512 if dense_probe else None,
        optimizer='SGD' if task == 'coco' else 'AdamW',
        weight_decay=.0001 if task == 'coco' else .05,
    )

    if schedule is not None:
        from downstream.spatial_backbones import revised_schedule
        selected = revised_schedule(backbone, task, track, schedule)
        recipe = replace(recipe, epochs=selected['epochs'], source_protocol_id=selected['protocol_id'])
    return recipe


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


def recipe_learning_rate(recipe, step, steps_per_epoch, base_lr):
    """Pure pre-update LR, shared with continuation validation at U-1."""
    _integer(step, 'step', 0)
    _integer(steps_per_epoch, 'steps_per_epoch', 1)
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('base_lr must be finite and positive')
    if recipe.task == 'coco':
        epoch = step // steps_per_epoch
        factor = (.001 + .999 * (step / 500) if step < 500 else
                  .01 if epoch >= 11 else .1 if epoch >= 8 else 1.)
        return base_lr * factor
    return cosine_lr(step, recipe.epochs * steps_per_epoch,
                     recipe.warmup_epochs * steps_per_epoch, base_lr,
                     recipe.min_lr, recipe.start_lr)


def set_learning_rates(optimizer, recipe: RevisedRecipe, step: int,
                       steps_per_epoch: int) -> None:
    """Set absolute per-group rates without changing their original base rates.

    Validate all groups before assigning any rate so malformed optimizer state
    cannot leave an otherwise valid group partially updated.
    """
    _integer(steps_per_epoch, 'steps_per_epoch', 1)
    rates = [recipe_learning_rate(recipe, step, steps_per_epoch, group['base_lr'])
             for group in optimizer.param_groups]
    for group, rate in zip(optimizer.param_groups, rates):
        group['lr'] = rate


def validate_completed_epochs(losses, epoch, steps_per_epoch):
    """Check each reported epoch against its scheduled optimizer update count."""
    if not isinstance(losses, list) or len(losses) != epoch:
        raise ValueError('checkpoint loss prefix must match completed epochs')
    for number, row in enumerate(losses, 1):
        if (not isinstance(row, dict) or set(row) != {'epoch', 'loss', 'updates'}
                or type(row['epoch']) is not int or row['epoch'] != number
                or type(row['updates']) is not int or row['updates'] != steps_per_epoch
                or type(row['loss']) not in (float, int) or not math.isfinite(row['loss'])):
            raise ValueError('invalid checkpoint completed-epoch loss log')
