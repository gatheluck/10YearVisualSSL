"""Shared check for BASIC5_FAIR_v1 rule `b` wiring in a probe evaluator.

**rule b** (docs/BASIC5_PROTOCOL.md): the eval preprocessing is **Resize
(shorter side) 256 + CenterCrop 224** -- a deterministic 224 centre crop that
preserves aspect ratio. The rule itself is implemented once, in
`probe_transforms.basic5_eval_transform`, and pinned once, in
`tests/test_probe_transforms.py`. What is left is per-method **wiring**: each
evaluator's val split -- which is also what its `feature_provider` extracts --
must route through that transform (a single-int `Resize`, then `CenterCrop`),
not the `Resize((h, w))` square that distorts aspect ratio and skips the crop.

That wiring check is the same for every method, so it lives here once and each
method's test imports it -- a rule implemented once per method is the common
root of past defects in this repository (see CLAUDE.md). A method's test supplies
only what differs: its own `_build_loader`.

The helper needs torchvision (the caller gates it with `needs_deps`).
"""

from __future__ import annotations


def _transforms(loader):
    return list(loader.dataset.transform.transforms)


def _by_type(loader, name):
    for t in _transforms(loader):
        if type(t).__name__ == name:
            return t
    return None


def assert_val_is_resize_centercrop(tc, build_loader, data_dir):
    """The val loader is Resize (shorter side, aspect-preserving) + CenterCrop.

    Positional call -- every probe `_build_loader` takes
    ``(data_root, split, size, batch_size, num_workers[, train=False])``; the
    aug-family methods carry a trailing ``train`` that defaults to ``False``, so
    the same five positional arguments build the val split for all of them.
    """
    _ds, loader = build_loader(str(data_dir), "val", 32, 2, 0)
    names = [type(t).__name__ for t in _transforms(loader)]
    tc.assertIn("Resize", names, "rule b: eval preprocessing resizes first")
    tc.assertIn("CenterCrop", names, "rule b: ... then centre-crops to 224")
    resize = _by_type(loader, "Resize")
    tc.assertIsInstance(
        resize.size, int,
        "rule b: Resize must scale the shorter side (a single int) and preserve "
        "aspect ratio, not a (h, w) square that distorts it and skips the crop")
