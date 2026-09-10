#!/usr/bin/env python3
"""Tests for `probe_optim`, the one place BASIC5 rule `opt` lives.

BASIC5_FAIR_v1 rule `opt` (docs/BASIC5_PROTOCOL.md): the linear probe optimiser
is **SGD, momentum 0.9, weight decay 0**, a **base LR of 0.1 defined at an
effective batch of 256** and **scaled linearly with the actual batch**
(`lr = base_lr * batch / 256`), a **cosine-decay** schedule, and **100 epochs**.

The base LR is defined at a reference batch of 256; a probe that trains at a
different batch (e.g. 32 for a large backbone that cannot fit 256) must scale the
LR linearly or its effective step size is wrong. Historically each evaluator read
the config LR directly, so only a batch-256 probe was correct. `probe_optim` owns
the scaling and the optimiser/scheduler construction so the rule is implemented
in exactly one place, not reimplemented once per method (there are ~50).

These tests pin what the rule is -- and, as negative controls, that scaling
actually happens (at a batch other than the reference the LR is *not* the base
LR), so a helper that quietly dropped the scaling would be caught here.

`basic5_scaled_lr` is pure arithmetic and is tested in the base environment; the
optimiser/scheduler builders need torch and are gated behind it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_optim                                          # noqa: E402
import _probe_opt                                           # noqa: E402

try:
    import torch                                            # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = unittest.skipUnless(HAVE_TORCH, "probe_optim builders need torch")


class TestBasic5ScaledLr(unittest.TestCase):
    """The linear LR scaling clause: lr = base_lr * batch / 256."""

    def test_the_reference_batch_leaves_the_lr_unchanged(self):
        # 0.1 defined at batch 256 stays 0.1 at batch 256.
        self.assertEqual(probe_optim.basic5_scaled_lr(0.1, 256), 0.1)

    def test_a_smaller_batch_scales_the_lr_down_linearly(self):
        # batch 32 is 1/8 of 256, so the LR is 1/8 of 0.1.
        self.assertAlmostEqual(probe_optim.basic5_scaled_lr(0.1, 32), 0.0125)

    def test_a_larger_batch_scales_the_lr_up_linearly(self):
        # batch 1024 is 4x 256, so the LR is 4x 0.1.
        self.assertAlmostEqual(probe_optim.basic5_scaled_lr(0.1, 1024), 0.4)

    def test_scaling_actually_happens_away_from_the_reference(self):
        # Negative control: away from batch 256 the scaled LR must NOT equal the
        # base LR. This is the assertion a "return base_lr" mutant must break.
        self.assertNotEqual(probe_optim.basic5_scaled_lr(0.1, 32), 0.1)
        self.assertNotEqual(probe_optim.basic5_scaled_lr(0.1, 1024), 0.1)

    def test_the_reference_batch_is_overridable(self):
        # If a caller declares a different reference batch, the ratio follows it.
        self.assertAlmostEqual(
            probe_optim.basic5_scaled_lr(0.1, 64, reference_batch=128), 0.05)

    def test_it_returns_a_float(self):
        # Integer inputs must not produce integer (floor) division.
        self.assertIsInstance(probe_optim.basic5_scaled_lr(1, 32), float)


class TestBasic5ProbeOptimizer(unittest.TestCase):
    """The optimiser builder: SGD, momentum 0.9, wd 0, scaled LR."""

    @needs_torch
    def _one_param(self):
        return [torch.nn.Parameter(torch.zeros(1))]

    @needs_torch
    def test_it_is_sgd_with_momentum_and_zero_weight_decay(self):
        opt = probe_optim.basic5_probe_optimizer(self._one_param(), 0.1, 256)
        self.assertIsInstance(opt, torch.optim.SGD)
        group = opt.param_groups[0]
        self.assertEqual(group["momentum"], 0.9)
        self.assertEqual(group["weight_decay"], 0.0)

    @needs_torch
    def test_the_lr_is_the_scaled_lr_not_the_base_lr(self):
        opt = probe_optim.basic5_probe_optimizer(self._one_param(), 0.1, 32)
        # batch 32 -> 0.1 * 32/256 = 0.0125, not the base 0.1.
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0125)

    @needs_torch
    def test_the_reference_batch_lr_is_the_base_lr(self):
        opt = probe_optim.basic5_probe_optimizer(self._one_param(), 0.1, 256)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.1)


class TestBasic5CosineSchedule(unittest.TestCase):
    """The schedule builder: cosine decay over the configured epochs."""

    @needs_torch
    def test_it_is_cosine_annealing_over_the_epoch_count(self):
        opt = probe_optim.basic5_probe_optimizer(
            [torch.nn.Parameter(torch.zeros(1))], 0.1, 256)
        sched = probe_optim.basic5_cosine_schedule(opt, epochs=100)
        self.assertIsInstance(
            sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(sched.T_max, 100)


class TestRunOptimizerDetectorControls(unittest.TestCase):
    """The AST detector that decides "this run() wires the rule" needs both a
    positive and a negative control (CLAUDE.md)."""

    _WIRED = (
        "def run(args):\n"
        "    optimizer = probe_optim.basic5_probe_optimizer(\n"
        "        params, base_lr=float(train['lr']), batch_size=bs)\n"
        "    scheduler = probe_optim.basic5_cosine_schedule(optimizer, epochs)\n"
        "    return optimizer\n")

    _RAW = (
        "def run(args):\n"
        "    optimizer = torch.optim.SGD(\n"
        "        params, lr=float(train['lr']), momentum=0.9, weight_decay=0.0)\n"
        "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
        "        optimizer, T_max=epochs)\n"
        "    return optimizer\n")

    def test_positive_a_wired_run_is_detected(self):
        self.assertTrue(_probe_opt.run_builds_scaled_optimizer(self._WIRED))

    def test_negative_a_raw_sgd_run_is_not_detected(self):
        # The historical un-scaled form must NOT satisfy the detector, or a
        # reverted method would pass unnoticed.
        self.assertFalse(_probe_opt.run_builds_scaled_optimizer(self._RAW))

    def test_no_run_function_is_not_detected(self):
        self.assertFalse(
            _probe_opt.run_builds_scaled_optimizer("x = 1\n"))


if __name__ == "__main__":
    unittest.main()
