#!/usr/bin/env python3
"""`encoder.pt` means the same thing in every method.

**Found by porting a second stage, which is the first thing that had to
*consume* an `encoder.pt` produced by a different method.** Two ports had
already written one, and they disagreed:

- the first strips the module prefix, so the keys load straight into the
  encoder module
- the third keeps `backbone.`, so they do not

The first attempt at this file demanded that every port strip its prefix, and
**that rule was wrong.** One port's encoder is not a single submodule -- it is
three of them together, with three different prefixes -- so there is nothing
to remove. A key-naming rule cannot hold across architectures that differ this
much, and writing one would have forced a port to lie about its own model.

What *can* be required is the property that matters to a consumer: **a port
must be able to load back the `encoder.pt` it wrote.** Without that, a
consumer gets `load_state_dict(..., strict=False)` quietly matching nothing,
and a linear evaluation on default initialisation reports a number that looks
like a result.

So every port declares `load_encoder`, and its own tests -- where its
dependencies are guaranteed to be installed -- prove the round trip. Methods
are discovered here, never listed.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ROOT / "methods"

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

# `_reference` writes a placeholder rather than a model: it exists to be a
# correct implementation of the contract without training anything, so it has
# no encoder module to compare against. One named entry, checked to exist.
EXEMPT = {"_reference"}


def adapters() -> list[Path]:
    return sorted(METHODS.glob("*/adapter/__init__.py"))


def ports() -> list[Path]:
    return [p for p in adapters() if p.parent.parent.name not in EXEMPT]


def prefix_constants(path: Path) -> dict:
    """The `*_PREFIX*` constants a port declares, read without importing."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and "PREFIX" in t.id:
                if isinstance(node.value, ast.Constant):
                    out[t.id] = node.value.value
                elif isinstance(node.value, ast.Tuple):
                    out[t.id] = tuple(
                        e.value for e in node.value.elts
                        if isinstance(e, ast.Constant))
    return out


def defines(path: Path, name: str) -> bool:
    """Whether the port defines a function, read without importing it.

    Importing every adapter would need every method's dependencies installed
    at once, which is exactly what the locked environments are arranged to
    avoid.
    """
    return any(isinstance(n, ast.FunctionDef) and n.name == name
               for n in ast.parse(path.read_text(encoding="utf-8")).body)


def round_trip_tested(method: str) -> bool:
    """Whether the method's own tests load an `encoder.pt` back."""
    f = ROOT / "tests" / f"test_method_{method}.py"
    if not f.is_file():
        return False
    src = f.read_text(encoding="utf-8")
    return "load_encoder" in src and "encoder.pt" in src


class TestEveryPortAgreesOnWhatEncoderPtHolds(unittest.TestCase):
    def test_the_ports_were_found(self):
        self.assertGreater(len(ports()), 1,
                           "with one port this file cannot fail")

    def test_every_exempt_name_exists(self):
        present = {p.parent.parent.name for p in adapters()}
        for name in EXEMPT:
            with self.subTest(exempt=name):
                self.assertIn(name, present)

    def test_every_port_declares_which_prefix_it_matches(self):
        for path in ports():
            with self.subTest(method=path.parent.parent.name):
                self.assertTrue(prefix_constants(path),
                                "no *_PREFIX* constant: what the encoder is "
                                "cannot be read from this port")

    def test_every_port_can_load_back_what_it_wrote(self):
        """Declared, so a consumer has something to call.

        Writing `encoder.pt` and never reading one is how a file that loads
        nothing goes unnoticed: `strict=False` matches no keys and says so to
        nobody.
        """
        missing = [p.parent.parent.name for p in ports()
                   if not defines(p, "load_encoder")]
        self.assertEqual(
            missing, [],
            "these write encoder.pt but declare no way to load one back:\n"
            + "\n".join(f"  - {x}" for x in missing))

    def test_every_port_proves_the_round_trip_in_its_own_tests(self):
        """Declaring it is not exercising it. The proof belongs in the
        method's own test file, which is the one place its dependencies are
        guaranteed to be installed."""
        missing = [p.parent.parent.name for p in ports()
                   if not round_trip_tested(p.parent.parent.name)]
        self.assertEqual(
            missing, [],
            "these never load an encoder.pt back in their own tests:\n"
            + "\n".join(f"  - {x}" for x in missing))

    def test_the_reader_can_tell_the_difference(self):
        """Against a reader that answers True to everything, the checks above
        are vacuous."""
        import shutil
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="encconv-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        has = d / "a.py"
        has.write_text("def load_encoder(x):\n    return x\n", encoding="utf-8")
        lacks = d / "b.py"
        lacks.write_text("def something_else(x):\n    return x\n",
                         encoding="utf-8")
        self.assertTrue(defines(has, "load_encoder"))
        self.assertFalse(defines(lacks, "load_encoder"))


if __name__ == "__main__":
    unittest.main()
