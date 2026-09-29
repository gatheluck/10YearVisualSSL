"""Explicit captured accumulation semantics shared by Extended image/dense probes."""
from contextlib import nullcontext

import torch

from downstream.attention import clip_attentive_gradients
from downstream.extended_distributed import launch, unwrap


def _require_all(condition,device,message):
    valid=torch.tensor(int(bool(condition)),device=device)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
        torch.distributed.all_reduce(valid,op=torch.distributed.ReduceOp.MIN)
    if not valid.item(): raise ValueError(message)


def resolve_execution(cfg, provider):
    """Keep reference-specific tail behavior, without silently correcting it."""
    tails = getattr(provider, 'EXTENDED_ACCUMULATION_TAILS', {})
    expected = tails.get(cfg['adaptation'])
    if expected not in ('discard', 'flush_scaled'):
        raise ValueError('provider lacks a verified Extended accumulation policy')
    settings = cfg.get('execution', dict(accumulation_steps=1, precision='fp32', tail_policy=expected))
    if not isinstance(settings, dict) or set(settings) != {'accumulation_steps','precision','tail_policy'}:
        raise ValueError('execution requires accumulation_steps, precision and tail_policy')
    count = settings['accumulation_steps']
    if type(count) is not int or count < 1:
        raise ValueError('accumulation_steps must be a positive integer')
    if settings['precision'] not in ('fp32','bf16') or settings['tail_policy'] != expected:
        raise ValueError('precision or captured provider tail policy does not agree')
    world=launch()[2]
    return dict(settings, effective_batch=cfg['probe']['batch_size']*count*world, world_size=world,
                schedule_clock='optimizer_updates_with_microbatch_horizon')


def autocast_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16':
        raise ValueError('precision must be fp32 or bf16')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('BF16 execution requires a CUDA device; no precision fallback')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('CUDA device lacks native BF16 support')
    return torch.autocast('cuda', dtype=torch.bfloat16)


def train_epoch(model, loader, loss_for_batch, optimizer, scheduler, *,
                accumulation_steps, tail_policy, adaptation):
    """Divide every loss by the full group size, including a flushed short tail.

    The captured schedule uses optimizer-update indices against a horizon
    measured in loader microbatches. Do not replace that clock with ceil(N/A).
    A discarded tail is cleared here, equivalent to the reference's next-epoch
    zero_grad, so it cannot leak to another caller or an exported checkpoint.
    """
    if type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError('invalid accumulation_steps')
    if tail_policy not in ('discard','flush_scaled') or adaptation not in ('frozen','attentive'):
        raise ValueError('invalid accumulation policy or adaptation')
    if not len(loader) or (tail_policy == 'discard' and len(loader) < accumulation_steps):
        raise ValueError('epoch cannot produce an optimizer update')
    model.train(); optimizer.zero_grad(set_to_none=True)
    stats = dict(microbatches=0, updates=0, tail_updates=0, discarded_microbatches=0)

    def update():
        device=next(model.parameters()).device
        _require_all(not any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()),
                     device,'nonfinite accumulated gradients')
        backbone = getattr(unwrap(model), 'backbone', None)
        _require_all(backbone is None or not any(p.requires_grad or p.grad is not None for p in backbone.parameters()),
                     device,'Extended backbone must remain frozen with no gradients')
        clip_attentive_gradients(model, adaptation)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        stats['updates'] += 1

    for batch in loader:
        loss = loss_for_batch(batch)
        _require_all(loss.ndim == 0 and torch.isfinite(loss),loss.device,'nonfinite or nonscalar training loss')
        (loss / accumulation_steps).backward()
        stats['microbatches'] += 1
        if stats['microbatches'] % accumulation_steps == 0:
            update()
    remaining = stats['microbatches'] % accumulation_steps
    if remaining and tail_policy == 'flush_scaled':
        update(); stats['tail_updates'] = 1
    elif remaining:
        stats['discarded_microbatches'] = remaining
    optimizer.zero_grad(set_to_none=True)
    return stats


def execution_report(plan, recipe, steps_per_epoch, scaled_lr):
    return dict(plan, scaled_lr=scaled_lr,
                schedule_steps_per_epoch=steps_per_epoch, schedule_horizon=recipe['epochs']*steps_per_epoch,
                microbatches=0, updates=0, tail_updates=0, discarded_microbatches=0)


def record_epoch(report, statistics):
    for key, value in statistics.items():
        report[key] += value
