"""Shared check for BASIC5_FAIR_v1 rule `opt` wiring in a linear-probe method.

**rule opt** (docs/BASIC5_PROTOCOL.md): the probe optimiser is SGD (momentum
0.9, weight decay 0) at a base LR of 0.1 defined at batch 256 and **scaled
linearly with the actual batch** (`lr = base_lr * batch / 256`), with a cosine
schedule over the configured epochs. The rule itself is implemented once, in
`probe_optim`, and pinned once, in `tests/test_probe_optim.py`. What is left is
per-method **wiring**: each evaluator's `run()` must build its optimiser and
schedule through those shared builders, so the LR is scaled for a non-256 batch
(the batch-32 large-backbone probes trained at the wrong effective LR before).

That wiring check is the same for every method, so it lives here once and each
method's test imports it -- a rule reimplemented once per method is the common
root of past defects in this repository (see CLAUDE.md). It is pure AST, so it
runs in the base environment with no torch and no GPU.

The detector `run_builds_scaled_optimizer` carries a positive and a negative
control in `tests/test_probe_optim.py` (a `run()` that wires the shared builders
vs a decoy `run()` that calls `torch.optim.SGD` directly), because a detector
that decides what gets checked needs both.
"""

from __future__ import annotations

import ast
from pathlib import Path

# The shared rule-`opt` builders a wired evaluator must call in run().
OPTIMIZER_BUILDER = "basic5_probe_optimizer"
SCHEDULE_BUILDER = "basic5_cosine_schedule"


def _run_function(src_text):
    for node in ast.parse(src_text).body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    return None


def _called_names(fn):
    names = set()
    for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
        name = getattr(call.func, "attr", getattr(call.func, "id", None))
        if name:
            names.add(name)
    return names


def run_builds_scaled_optimizer(src_text):
    """True iff `run()` builds its optimiser and schedule via `probe_optim`.

    Whole call names are compared (never a substring over the file), so a comment
    or an unrelated identifier cannot satisfy it. A `run()` that constructs
    `torch.optim.SGD` directly -- the historical un-scaled form -- returns False.
    """
    fn = _run_function(src_text)
    if fn is None:
        return False
    names = _called_names(fn)
    return OPTIMIZER_BUILDER in names and SCHEDULE_BUILDER in names


def assert_run_scales_probe_lr(tc, source_path):
    """`run()` builds its optimiser/schedule via `probe_optim` (rule `opt`).

    Proven structurally so it needs no GPU. Non-vacuity is proven per method by
    `mutations/<method>-probe-opt.json` (revert the wired call to a raw
    `torch.optim.SGD` with the config LR -> this assertion must fail).
    """
    src = Path(source_path).read_text()
    tc.assertTrue(
        run_builds_scaled_optimizer(src),
        f"{source_path}: run() must build its optimiser via "
        f"probe_optim.{OPTIMIZER_BUILDER} and its schedule via "
        f"{SCHEDULE_BUILDER} (rule opt: LR scaled by batch/256), not a raw "
        "torch.optim.SGD with the config LR")
