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

**Not every port produces an `encoder.pt`.** An eval-only port -- one whose
adapter has no `step1` stage -- trains nothing and probes a frozen, downloaded
backbone (36_franca is the first). It writes no encoder, so the round-trip
requirement does not apply to it; instead it must *declare* that it produces
none (`_absent_reason`, which adapterlib enforces at run time). Which ports
produce an encoder is discovered from each adapter's own `STAGES` -- `step1` is
the stage that trains and writes one -- never from a list of names.
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


def stage_names(path: Path) -> "set | None":
    """The stage names a port declares in `STAGES`, read without importing.

    `STAGES` is written as a tuple in most ports and as a dict (stage -> keys)
    in the first one, so both shapes are read. Returns None when no `STAGES` can
    be read, which the caller treats conservatively rather than as 'no stages'.
    """
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if not (isinstance(t, ast.Name) and t.id == "STAGES"):
                continue
            v = node.value
            if isinstance(v, (ast.Tuple, ast.List)):
                return {e.value for e in v.elts if isinstance(e, ast.Constant)}
            if isinstance(v, ast.Dict):
                return {k.value for k in v.keys if isinstance(k, ast.Constant)}
    return None


def produces_encoder(path: Path) -> bool:
    """Whether the port writes an `encoder.pt`. In this contract `step1` is the
    stage that trains and writes one; an eval-only port (no `step1`, only
    `linear_eval`) probes a frozen/downloaded backbone and writes none. When
    `STAGES` cannot be read, assume it produces an encoder -- the stricter path,
    so a parse failure never quietly exempts a real port."""
    names = stage_names(path)
    if names is None:
        return True
    return "step1" in names


def encoder_ports() -> list[Path]:
    return [p for p in ports() if produces_encoder(p)]


def eval_only_ports() -> list[Path]:
    return [p for p in ports() if not produces_encoder(p)]


def defines(path: Path, name: str) -> bool:
    """Whether the port defines a function, read without importing it.

    Importing every adapter would need every method's dependencies installed
    at once, which is exactly what the locked environments are arranged to
    avoid.
    """
    return any(isinstance(n, ast.FunctionDef) and n.name == name
               for n in ast.parse(path.read_text(encoding="utf-8")).body)


def calls(path: Path, name: str) -> bool:
    """Whether the file **invokes** `name` -- a call, read from the AST, not a
    substring over the text.

    A substring is how `round_trip_tested` first read this, and it let a comment
    or a docstring mentioning `load_encoder` stand in for an actual round trip
    (the recorded 'substring match over too wide a scope'). A call node cannot
    be faked by prose. Matches both `name(...)` and `obj.name(...)`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == name:
            return True
        if isinstance(f, ast.Attribute) and f.attr == name:
            return True
    return False


def round_trip_tested(method: str) -> bool:
    """Whether the method's own tests load an `encoder.pt` back.

    Read as a **call** to `load_encoder`, not as the substrings `load_encoder`
    and `encoder.pt` appearing anywhere in the file -- the latter passed a test
    that only mentioned them. Every encoder-port's test file was checked to make
    a real call, so this is stricter, not merely different.
    """
    f = ROOT / "tests" / f"test_method_{method}.py"
    return f.is_file() and calls(f, "load_encoder")


class TestEveryPortAgreesOnWhatEncoderPtHolds(unittest.TestCase):
    def test_the_ports_were_found(self):
        self.assertGreater(len(ports()), 1,
                           "with one port this file cannot fail")

    def test_every_exempt_name_exists(self):
        present = {p.parent.parent.name for p in adapters()}
        for name in EXEMPT:
            with self.subTest(exempt=name):
                self.assertIn(name, present)

    def test_every_encoder_port_declares_which_prefix_it_matches(self):
        for path in encoder_ports():
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
        missing = [p.parent.parent.name for p in encoder_ports()
                   if not defines(p, "load_encoder")]
        self.assertEqual(
            missing, [],
            "these write encoder.pt but declare no way to load one back:\n"
            + "\n".join(f"  - {x}" for x in missing))

    def test_every_port_proves_the_round_trip_in_its_own_tests(self):
        """Declaring it is not exercising it. The proof belongs in the
        method's own test file, which is the one place its dependencies are
        guaranteed to be installed."""
        missing = [p.parent.parent.name for p in encoder_ports()
                   if not round_trip_tested(p.parent.parent.name)]
        self.assertEqual(
            missing, [],
            "these never load an encoder.pt back in their own tests:\n"
            + "\n".join(f"  - {x}" for x in missing))

    def test_every_eval_only_port_declares_it_produces_no_encoder(self):
        """The exemption is not a silent hole: a port with no `step1` writes no
        encoder, and must say so via `_absent_reason` (which adapterlib enforces
        at run time), rather than quietly skipping the convention."""
        missing = [p.parent.parent.name for p in eval_only_ports()
                   if not defines(p, "_absent_reason")]
        self.assertEqual(
            missing, [],
            "these produce no encoder but never declare it (_absent_reason):\n"
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

    def test_the_round_trip_reader_needs_an_actual_call(self):
        """`round_trip_tested` must read a *call*, not a substring.

        The earlier version tested ``"load_encoder" in src and "encoder.pt" in
        src`` -- a substring over the whole file -- so a test that merely
        *mentions* those strings in a comment, with no round trip, passed
        silently. This is the recorded mistake (a substring match over too wide
        a scope), the same class as the git/logits CI timeout. The negative
        control carries both strings but never calls anything; the positive
        control makes the call.
        """
        import shutil
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="rt-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        decoy = d / "test_method_decoy.py"
        decoy.write_text(
            "# this test loads adapter.load_encoder on encoder.pt somewhere\n"
            'DOC = "load_encoder ... encoder.pt"\n'
            "x = 1\n", encoding="utf-8")
        real = d / "test_method_real.py"
        real.write_text(
            "import adapter\n"
            "def test_it():\n"
            "    m = adapter.load_encoder(saved, cfg)  # loads encoder.pt back\n",
            encoding="utf-8")
        self.assertFalse(calls(decoy, "load_encoder"),
                         "a substring match over too wide a scope is back")
        self.assertTrue(calls(real, "load_encoder"))

    def test_the_stage_reader_can_tell_encoder_ports_apart(self):
        """Against a reader that calls every port an encoder-producer (or none),
        the split above is vacuous. A step1 port produces one; a linear_eval-only
        port does not."""
        import shutil
        import tempfile
        d = Path(tempfile.mkdtemp(prefix="encconv-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        trains = d / "trains.py"
        trains.write_text('STAGES = ("step1", "linear_eval")\n', encoding="utf-8")
        evals = d / "evals.py"
        evals.write_text('STAGES = ("linear_eval",)\n', encoding="utf-8")
        as_dict = d / "as_dict.py"
        as_dict.write_text('STAGES = {"step1": {}, "linear_eval": {}}\n',
                           encoding="utf-8")
        unreadable = d / "unreadable.py"
        unreadable.write_text("x = 1\n", encoding="utf-8")
        self.assertTrue(produces_encoder(trains))
        self.assertFalse(produces_encoder(evals))
        self.assertTrue(produces_encoder(as_dict))   # dict STAGES read too
        self.assertTrue(produces_encoder(unreadable))  # unknown -> stricter path

    def test_both_shapes_are_present_so_the_split_is_exercised(self):
        """Guard against the split silently covering nothing: the repository has
        at least one encoder-producing port and at least one eval-only port."""
        self.assertTrue(encoder_ports(), "no encoder-producing port found")
        self.assertTrue(eval_only_ports(), "no eval-only port found")


if __name__ == "__main__":
    unittest.main()
