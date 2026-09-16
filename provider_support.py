"""Collision-safe import of a method's own sibling module.

A `feature_provider` imports its method's siblings (`adapter`,
`evaluate_linear`, ...) by bare name, relying on the method directory being
first on `sys.path`. That is correct in the isolated worker subprocess
(`bin/extract-features.py --worker`), where only one method is ever loaded.
But the test suite -- and the driver's in-process debug path -- load many
methods in one interpreter, and `sys.modules` keeps the first module imported
under a given name. Single-file module names shared across methods
(`evaluate_linear` is defined by four methods, `evaluate_linear_official` by
two) then resolve to whichever method ran first, so a later method's provider
silently gets another method's `_IMAGENET_MEAN` / `extract_features`. The
`locked` CI matrix, which runs the whole suite in one process per method venv,
caught exactly this on the provider fan-out.

`tests/_method_import.load_from` already solves the same problem for the
*packages* a method's trainer imports (`data`, `models`, `nce`, `adapter`), by
purging shared package names. It does not cover single-file modules, and it is
a test helper -- the provider runs in production too. This module is the
runtime counterpart the provider itself calls: it puts the method directory
first on `sys.path` and drops every cached top-level module whose file lives in
a *different* method, then imports. Modules from the standard library or
site-packages are never touched -- their files are outside `methods/`. In the
one-method subprocess there is nothing foreign to drop, so it is a no-op there.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _purge_foreign(method_dir: Path) -> None:
    """Drop every cached top-level module whose file belongs to a *different*
    method under the same `methods/` root. Never touches modules outside
    `methods/` (stdlib, site-packages), and keeps this method's own modules."""
    method_dir = Path(method_dir).resolve()
    methods_root = method_dir.parent
    mine = str(method_dir) + os.sep       # trailing sep: 21 must not match 21x
    root = str(methods_root) + os.sep
    for name in list(sys.modules):
        if "." in name:                   # submodules go with their parent
            continue
        mod = sys.modules.get(name)
        origin = getattr(mod, "__file__", None) or ""
        if origin.startswith(root) and not origin.startswith(mine):
            del sys.modules[name]
            for sub in [k for k in sys.modules if k.startswith(f"{name}.")]:
                del sys.modules[sub]


def import_sibling(method_dir, name: str):
    """Import the module `name` as a sibling of `method_dir`, resolving against
    this method even when another method already imported a module of the same
    name in this interpreter."""
    method_dir = Path(method_dir).resolve()
    p = str(method_dir)
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    _purge_foreign(method_dir)
    return importlib.import_module(name)


def load_native_options(encoder_path, method: str, supported: set) -> dict:
    """Read explicit extraction options bound to this exported encoder's hash.

    Ordinary encoder files and older exports retain provider defaults. New
    options fail closed on mismatched identity or unsupported protocol fields.
    Keep encoder.pt and export.json together when transferring native exports.
    """
    import hashlib
    import json
    encoder = Path(encoder_path)
    sidecar = encoder.with_name('export.json')
    if not sidecar.exists():
        return {}
    record = json.loads(sidecar.read_text())
    if 'feature_options' not in record:
        return {}
    options = record['feature_options']
    if not isinstance(options, dict):
        raise ValueError('feature options must be a mapping')
    if set(options) - supported:
        raise ValueError('unsupported native feature options')
    if record.get('method') != method:
        raise ValueError('native export method mismatch')
    digest = hashlib.sha256()
    with encoder.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != record.get('encoder_sha256'):
        raise ValueError('native export encoder hash mismatch')
    return options


def configure_native_resize(transform, options: dict) -> None:
    """Override one Resize in an existing val pipeline, preserving its tail."""
    if not ({'resize_short_side', 'interpolation'} & options.keys()):
        return
    from torchvision import transforms as T
    modes = {'bilinear': T.InterpolationMode.BILINEAR,
             'bicubic': T.InterpolationMode.BICUBIC}
    mode = options.get('interpolation')
    if mode is not None and mode not in modes:
        raise ValueError('unsupported interpolation')
    size = options.get('resize_short_side')
    if size is not None and (type(size) is not int or size <= 0):
        raise ValueError('resize_short_side must be a positive integer')
    indices = [i for i, step in enumerate(transform.transforms) if isinstance(step, T.Resize)]
    if len(indices) != 1:
        raise ValueError('native resize requires exactly one Resize')
    i = indices[0]
    old = transform.transforms[i]
    transform.transforms[i] = T.Resize(size if size is not None else old.size,
                                      interpolation=modes[mode] if mode else old.interpolation,
                                      max_size=old.max_size, antialias=old.antialias)
