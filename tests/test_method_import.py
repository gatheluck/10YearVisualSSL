#!/usr/bin/env python3
"""Regression tests for tests/_method_import.py.

The suite is the one place two methods' `models/` (and `data/`, `nce/`, ...)
packages are present at once. `_method_import.load_from` keeps them from
colliding in `sys.modules`. This has broken twice:

  1. a hand-kept shared-name list missed `nce` when a second method defined one;
     fixed by discovering shared names from `methods/` (`_shared_names`).
  2. two methods sharing an *alias* (`this_methods_models`, used by the
     round-trip tests) and a *submodule* name (`vision_transformer`, in both
     `23_dino` and `27_ibot`) cross-imported: loading the alias for the second
     method left the first method's `alias.vision_transformer` in `sys.modules`,
     so the second's `from .vision_transformer import vit_large` resolved against
     the first method's file and `vit_large` was not found.

These tests use fabricated fixture packages (no torch) so they run in the base
gate -- the torch-gated method smokes are exactly what hid bug 2 until CI.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import _method_import                                   # noqa: E402
from _method_import import load_from, _shared_names     # noqa: E402


class TestTheAliasSubmodulePurge(unittest.TestCase):
    """A shared alias must not leak one method's submodule into another's."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="methodimport-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._drop_alias()
        self.addCleanup(self._drop_alias)

    def _drop_alias(self) -> None:
        for key in [k for k in sys.modules
                    if k == "shared_alias" or k.startswith("shared_alias.")]:
            del sys.modules[key]

    def _fake_method(self, name: str, symbol: str) -> Path:
        """A method dir whose models/ imports a symbol from a submodule that is
        named the same across methods (models/vision_transformer.py)."""
        pkg = self.tmp / name / "models"
        pkg.mkdir(parents=True)
        (pkg / "vision_transformer.py").write_text(
            f"{symbol} = {symbol!r}\n", encoding="utf-8")
        (pkg / "__init__.py").write_text(
            f"from .vision_transformer import {symbol}\n", encoding="utf-8")
        return self.tmp / name

    def test_two_methods_sharing_an_alias_and_submodule_do_not_cross_import(self):
        a = self._fake_method("methodA", "alpha")
        b = self._fake_method("methodB", "beta")

        ma = load_from(a, "shared_alias", a / "models" / "__init__.py")
        self.assertTrue(hasattr(ma, "alpha"))

        # The bug: this raised ImportError, because shared_alias.vision_transformer
        # was still methodA's file and had no `beta`.
        mb = load_from(b, "shared_alias", b / "models" / "__init__.py")
        self.assertTrue(hasattr(mb, "beta"),
                        "the alias returned another method's submodule")
        self.assertFalse(hasattr(mb, "alpha"),
                         "methodA's symbol leaked into methodB")

        # Symmetric: going back the other way must not leak the other direction.
        ma2 = load_from(a, "shared_alias", a / "models" / "__init__.py")
        self.assertTrue(hasattr(ma2, "alpha"))
        self.assertFalse(hasattr(ma2, "beta"))

    def test_reloading_the_same_file_under_the_same_name_is_cached(self):
        a = self._fake_method("methodA", "alpha")
        first = load_from(a, "shared_alias", a / "models" / "__init__.py")
        second = load_from(a, "shared_alias", a / "models" / "__init__.py")
        self.assertIs(first, second, "same (name, file) should hit the cache")


class TestSharedNamesAreDiscoveredNotListed(unittest.TestCase):
    """The shared-name set is computed from methods/, so a newly shared package
    name cannot be silently missed."""

    def test_models_and_data_are_discovered_as_shared(self):
        shared = _shared_names()
        # More than one shipped method defines each of these packages.
        self.assertIn("models", shared)
        self.assertIn("data", shared)

    def test_it_is_computed_from_the_tree_not_a_hard_coded_list(self):
        import ast
        src = Path(_method_import.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_shared_names")
        # A hand-kept list would appear as string literals like "models"/"data"
        # inside the function body; the discovered version has none.
        literals = {n.value for n in ast.walk(fn)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertNotIn("models", literals,
                         "shared names look hard-coded, not discovered")
        self.assertNotIn("data", literals)


if __name__ == "__main__":
    unittest.main()
