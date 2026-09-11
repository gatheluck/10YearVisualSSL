"""The one place BASIC5_FAIR_v1 rule `opt` is implemented.

**rule opt** (docs/BASIC5_PROTOCOL.md): the linear probe optimiser is **SGD**
with **momentum 0.9** and **weight decay 0**, a **base LR of 0.1 defined at an
effective batch of 256** and **scaled linearly with the actual batch**
(``lr = base_lr * batch / 256``), a **cosine-decay** schedule, and **100
epochs**.

The base LR is defined at a reference batch of 256; a probe that trains at a
different batch (e.g. 32 for a large backbone that cannot fit 256) must scale the
LR linearly, or its effective step size is wrong. Historically each evaluator
read the config LR directly -- so only a batch-256 probe was correct, and the
batch-32 / batch-1024 configs silently trained at the wrong effective LR. This
module owns the scaling and the optimiser/scheduler construction so the rule is
implemented once, not reimplemented ~50 times (scanners that each reimplemented a
rule are the common root of past defects here).

``basic5_scaled_lr`` is pure arithmetic and imports nothing, so the base
environment (standard library only) can use it without torch installed. torch is
imported lazily inside the optimiser/scheduler builders.
"""

from __future__ import annotations

# The effective batch the base LR is defined at (rule `opt`).
BASIC5_REFERENCE_BATCH = 256


def basic5_scaled_lr(base_lr, batch_size, reference_batch=BASIC5_REFERENCE_BATCH):
    """The BASIC5 rule-`opt` linearly-scaled learning rate.

    Returns ``base_lr * batch_size / reference_batch``. The base LR is defined at
    ``reference_batch`` (256) and scales linearly with the actual batch, so a
    batch-32 probe uses 1/8 the LR and a batch-256 probe is unchanged. Everything
    is cast to ``float`` so integer inputs never trigger floor division.

    Args:
        base_lr: the base learning rate, defined at ``reference_batch``.
        batch_size: the actual per-step batch the probe trains at.
        reference_batch: the batch the base LR is defined at (256 by rule).

    Returns:
        the scaled learning rate, a ``float``.
    """
    return float(base_lr) * float(batch_size) / float(reference_batch)


def basic5_probe_optimizer(params, base_lr, batch_size, momentum=0.9,
                           weight_decay=0.0,
                           reference_batch=BASIC5_REFERENCE_BATCH):
    """The BASIC5 rule-`opt` optimiser: SGD at the linearly-scaled LR.

    Builds ``torch.optim.SGD`` with the rule's ``momentum`` (0.9) and
    ``weight_decay`` (0) and the LR from :func:`basic5_scaled_lr`, so the LR is
    scaled for the actual batch in exactly one place.

    Args:
        params: the parameters to optimise (the linear head's).
        base_lr: the base learning rate, defined at ``reference_batch``.
        batch_size: the actual per-step batch the probe trains at.
        momentum: SGD momentum (0.9 by rule).
        weight_decay: SGD weight decay (0 by rule).
        reference_batch: the batch the base LR is defined at (256 by rule).

    Returns:
        a ``torch.optim.SGD``.
    """
    import torch

    lr = basic5_scaled_lr(base_lr, batch_size, reference_batch)
    return torch.optim.SGD(params, lr=lr, momentum=momentum,
                           weight_decay=weight_decay)


def basic5_cosine_schedule(optimizer, epochs):
    """The BASIC5 rule-`opt` schedule: cosine decay over ``epochs``.

    Args:
        optimizer: the optimiser to schedule.
        epochs: the number of epochs the probe trains for (``T_max``).

    Returns:
        a ``torch.optim.lr_scheduler.CosineAnnealingLR``.
    """
    import torch

    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
