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
