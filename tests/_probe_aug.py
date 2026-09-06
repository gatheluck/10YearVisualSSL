"""Shared checks for BASIC5_FAIR_v1 rule `aug` wiring in a feature-cache method.

**rule aug** (docs/BASIC5_PROTOCOL.md): the linear probe's train-time
augmentation is *RandomResizedCrop + HorizontalFlip only*. The rule itself is
implemented once, in `probe_transforms`, and pinned once, in
`tests/test_probe_transforms.py`. What is left is per-method **wiring**: each
feature-cache evaluator must route its *train* split through that transform and
leave its *val* split deterministic (the provider extracts val, so val must not
move).

That wiring check is the same for every method, so it lives here once and each
method's test imports it -- a rule implemented once per method is the common
root of past defects in this repository (see CLAUDE.md). A method's test supplies
only what differs: its own `_build_loader`, its evaluator source file, and -- for
the val control -- the deterministic op its val transform is expected to keep
(`CenterCrop` for most; `None` for a method whose val is a plain square resize).

The behavioural helpers need torchvision (the caller gates them with
`needs_deps`); `assert_run_wires_train_split` is pure AST and runs in the base
environment.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _transform_names(loader):
    return [type(t).__name__ for t in loader.dataset.transform.transforms]


def assert_train_augments(tc, build_loader, data_dir):
    """The train loader applies rule aug: RandomResizedCrop + HorizontalFlip.

    Positional call -- every feature-cache `_build_loader` is
    ``(data_root, split, size, batch_size, num_workers, train=False)``; only the
    size parameter's *name* differs between methods, not its position.
    """
    _ds, loader = build_loader(str(data_dir), "train", 32, 2, 0, train=True)
    names = _transform_names(loader)
    tc.assertIn("RandomResizedCrop", names,
                "the probe's train split must be RandomResizedCrop'd (rule aug)")
    tc.assertIn("RandomHorizontalFlip", names,
                "the probe's train split must be horizontally flipped (rule aug)")


def assert_val_deterministic(tc, build_loader, data_dir, must_include=None):
    """The val loader stays deterministic: none of rule aug's random ops.

    ``must_include`` names the deterministic op the method's val transform keeps
    (``CenterCrop`` for most methods), asserted as an extra positive so the
    control is not merely "the random ops are absent". A method whose val is a
    plain square resize (no centre crop) passes ``None``.
    """
    _ds, loader = build_loader(str(data_dir), "val", 32, 2, 0)
    names = _transform_names(loader)
    tc.assertNotIn("RandomResizedCrop", names, "val must not be randomly cropped")
    tc.assertNotIn("RandomHorizontalFlip", names, "val must not be randomly flipped")
    if must_include is not None:
        tc.assertIn(must_include, names,
                    f"val stays deterministic (keeps {must_include})")


def assert_run_wires_train_split(tc, source_path):
    """run() builds the train split with train=True and the val split without it.

    The wiring, not just the helper -- proven structurally so it needs no GPU.
    The val leg is a negative control: it must be built *without* ``train=True``,
    so a mutation that augmented val (moving the cached provider features) is
    caught here too.
    """
    src = Path(source_path).read_text()
    run_fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "run")
    saw_train_true = saw_val_without_train = False
    for call in (n for n in ast.walk(run_fn) if isinstance(n, ast.Call)):
        name = getattr(call.func, "attr", getattr(call.func, "id", None))
        if name != "_build_loader":
            continue
        split = None
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
            split = call.args[1].value
        train_true = any(
            kw.arg == "train" and isinstance(kw.value, ast.Constant)
            and kw.value.value is True for kw in call.keywords)
        if split == "train" and train_true:
            saw_train_true = True
        if split == "val" and not train_true:
            saw_val_without_train = True
    tc.assertTrue(
        saw_train_true,
        "run() must build the train split with _build_loader(..., train=True)")
    tc.assertTrue(
        saw_val_without_train,
        "run() must build the val split deterministically (no train=True)")
