#!/usr/bin/env python3
"""Regression tests for provider_support.import_sibling.

A `feature_provider` imports its method's siblings (`adapter`,
`evaluate_linear`, ...) by bare name. That is correct in the isolated worker
subprocess, where only one method is ever loaded, but the test suite -- and the
driver's in-process debug path -- load many methods in one interpreter, and
`sys.modules` keeps the first module imported under a given name. Single-file
module names shared across methods (`evaluate_linear` is defined by four
methods, `evaluate_linear_official` by two) then resolve to whichever method
ran first, so a later method's provider silently got another method's
`_IMAGENET_MEAN` / `extract_features` -- the failure the `locked` CI matrix
caught on the fan-out.

`import_sibling` makes the import resolve against the calling method regardless
of what a previous method left cached. These tests use fabricated method dirs
(no torch) so they run in the base gate -- the torch-gated method smokes are
exactly what hid the collision until CI.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import provider_support                                   # noqa: E402
from provider_support import import_sibling               # noqa: E402


class TestImportSibling(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="providersupport-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self._drop()
        self.addCleanup(self._drop)
        self._saved_path = list(sys.path)
        self.addCleanup(self._restore_path)

    def _restore_path(self) -> None:
        sys.path[:] = self._saved_path

    def _drop(self) -> None:
        for key in [k for k in sys.modules
                    if k in ("sib", "pkg") or k.startswith(("sib.", "pkg."))]:
            del sys.modules[key]

    def _single_file_method(self, name: str, value: str) -> Path:
        """A method dir with a single-file sibling module `sib.py`."""
        d = self.tmp / "methods" / name
        d.mkdir(parents=True)
        (d / "sib.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
        return d

    def test_a_shared_single_file_name_resolves_to_the_calling_method(self):
        a = self._single_file_method("methodA", "A")
        b = self._single_file_method("methodB", "B")

        # A is imported first, so sys.modules['sib'] holds A's module.
        self.assertEqual(import_sibling(a, "sib").VALUE, "A")

        # Without the foreign purge, this returns A's cached module -- the bug.
        self.assertEqual(import_sibling(b, "sib").VALUE, "B",
                         "a sibling leaked in from another method")

        # Symmetric: coming back to A must not now return B's.
        self.assertEqual(import_sibling(a, "sib").VALUE, "A")

    def test_the_method_directory_is_put_first_on_sys_path(self):
        a = self._single_file_method("methodA", "A")
        b = self._single_file_method("methodB", "B")
        import_sibling(a, "sib")
        # Even with A's dir already on sys.path, B's import must win. This fails
        # if the method dir is appended instead of inserted at the front.
        self.assertEqual(import_sibling(b, "sib").VALUE, "B")

    def test_stdlib_modules_are_never_purged(self):
        import json as _json
        a = self._single_file_method("methodA", "A")
        import_sibling(a, "sib")
        self.assertIs(sys.modules["json"], _json,
                      "a module outside methods/ was purged")

    def test_a_foreign_package_and_its_submodules_are_purged_together(self):
        """A shared *package* name from another method, and its submodules, are
        both dropped, so a relative import in the calling method re-resolves."""
        def make(name: str, symbol: str) -> Path:
            d = self.tmp / "methods" / name / "pkg"
            d.mkdir(parents=True)
            (d / "leaf.py").write_text(f"{symbol} = {symbol!r}\n",
                                       encoding="utf-8")
            (d / "__init__.py").write_text(
                f"from .leaf import {symbol}\n", encoding="utf-8")
            return self.tmp / "methods" / name

        a = make("methodA", "alpha")
        b = make("methodB", "beta")
        self.assertTrue(hasattr(import_sibling(a, "pkg"), "alpha"))
        mb = import_sibling(b, "pkg")
        self.assertTrue(hasattr(mb, "beta"),
                        "the package returned another method's copy")
        self.assertFalse(hasattr(mb, "alpha"),
                         "methodA's symbol leaked into methodB's package")


if __name__ == "__main__":
    unittest.main()
