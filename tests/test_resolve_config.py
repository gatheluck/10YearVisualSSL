#!/usr/bin/env python3
"""Specification for resolve-config.py.

**`config_sha256` is the hinge of reproducibility.** The manifest claims that
a particular configuration produced a particular result, and that claim is
only worth something if the same configuration always hashes the same way.

So the resolved config is a *canonical* artifact, not a file somebody typed:

- **One spelling per value.** Sorted keys, fixed separators, no trailing
  whitespace. Two authors expressing the same settings get the same bytes,
  therefore the same hash
- **Nothing unresolved may survive.** An `include` left unexpanded, or a
  `${VAR}` left standing, means the file does not say what actually ran
- **The ambient environment is not an input.** `resolve-config` never reads
  `os.environ`. A config that silently absorbs the machine it was resolved on
  is not reproducible; values must be passed explicitly with `--set`
- **Nothing is dropped quietly.** Duplicate keys, values JSON cannot carry,
  and unset variables are all refused by name (DESIGN 2.4)

YAML is for authoring only, and is optional. The resolved artifact is JSON
because JSON has a canonical form reachable from the standard library and
YAML has neither.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"

try:
    import yaml                                       # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rc_mod = load("resolve_config", "resolve-config.py")


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rctest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, rel: str, obj_or_text) -> Path:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj_or_text, str):
            p.write_text(obj_or_text, encoding="utf-8")
        else:
            p.write_text(json.dumps(obj_or_text), encoding="utf-8")
        return p

    def resolve(self, path: Path, **kw):
        return rc_mod.resolve(path, **kw)


class TestCanonicalForm(Base):
    """Same settings, same bytes, same hash. Nothing else gives that."""

    def test_keys_are_sorted(self):
        p = self.write("c.json", {"z": 1, "a": 2, "m": {"y": 1, "b": 2}})
        self.assertEqual(rc_mod.canonical(self.resolve(p)),
                         b'{"a":2,"m":{"b":2,"y":1},"z":1}\n')

    def test_key_order_in_the_source_does_not_matter(self):
        a = self.write("a.json", '{"x": 1, "y": 2}')
        b = self.write("b.json", '{"y": 2, "x": 1}')
        self.assertEqual(rc_mod.canonical(self.resolve(a)),
                         rc_mod.canonical(self.resolve(b)))

    def test_whitespace_in_the_source_does_not_matter(self):
        a = self.write("a.json", '{"x":1}')
        b = self.write("b.json", '{\n  "x" :  1\n}\n')
        self.assertEqual(rc_mod.canonical(self.resolve(a)),
                         rc_mod.canonical(self.resolve(b)))

    def test_output_ends_with_exactly_one_newline(self):
        """A missing or doubled newline changes the hash for no reason."""
        out = rc_mod.canonical(self.resolve(self.write("c.json", {"x": 1})))
        self.assertTrue(out.endswith(b"}\n"))
        self.assertFalse(out.endswith(b"\n\n"))

    def test_non_ascii_is_stored_as_utf8_not_escaped(self):
        """One representation, chosen deliberately, so the bytes are fixed."""
        p = self.write("c.json", {"note": "café"})
        self.assertIn("café".encode(), rc_mod.canonical(self.resolve(p)))

    def test_the_hash_is_of_the_canonical_bytes(self):
        p = self.write("c.json", {"b": 1, "a": 2})
        data = rc_mod.canonical(self.resolve(p))
        self.assertEqual(rc_mod.sha256_of_config(self.resolve(p)),
                         hashlib.sha256(data).hexdigest())

    def test_two_spellings_reach_the_same_hash(self):
        a = self.write("a.json", '{"seed":0,"lr":0.1}')
        b = self.write("b.json", '{ "lr" : 0.1 ,\n"seed" : 0 }')
        self.assertEqual(rc_mod.sha256_of_config(self.resolve(a)),
                         rc_mod.sha256_of_config(self.resolve(b)))

    def test_different_settings_reach_different_hashes(self):
        """A hash that never changes would pass every test above."""
        a = self.write("a.json", {"seed": 0})
        b = self.write("b.json", {"seed": 1})
        self.assertNotEqual(rc_mod.sha256_of_config(self.resolve(a)),
                            rc_mod.sha256_of_config(self.resolve(b)))


class TestIncludes(Base):
    def test_include_is_expanded(self):
        self.write("base.json", {"seed": 0, "lr": 0.1})
        p = self.write("c.json", {"include": ["base.json"], "lr": 0.5})
        self.assertEqual(self.resolve(p), {"seed": 0, "lr": 0.5})

    def test_no_include_key_survives_resolution(self):
        """A resolved file still carrying `include` does not say what ran."""
        self.write("base.json", {"seed": 0})
        p = self.write("c.json", {"include": ["base.json"]})
        self.assertNotIn("include", self.resolve(p))

    def test_includes_apply_in_order_with_later_winning(self):
        self.write("one.json", {"x": 1})
        self.write("two.json", {"x": 2})
        p = self.write("c.json", {"include": ["one.json", "two.json"]})
        self.assertEqual(self.resolve(p)["x"], 2)

    def test_the_including_file_wins_over_what_it_includes(self):
        self.write("base.json", {"x": 1})
        p = self.write("c.json", {"include": ["base.json"], "x": 9})
        self.assertEqual(self.resolve(p)["x"], 9)

    def test_paths_are_relative_to_the_including_file(self):
        self.write("sub/base.json", {"x": 1})
        p = self.write("sub/c.json", {"include": ["base.json"]})
        self.assertEqual(self.resolve(p)["x"], 1)

    def test_includes_nest(self):
        self.write("a.json", {"x": 1})
        self.write("b.json", {"include": ["a.json"], "y": 2})
        p = self.write("c.json", {"include": ["b.json"], "z": 3})
        self.assertEqual(self.resolve(p), {"x": 1, "y": 2, "z": 3})

    def test_dictionaries_merge_deeply(self):
        self.write("base.json", {"opt": {"lr": 0.1, "momentum": 0.9}})
        p = self.write("c.json", {"include": ["base.json"],
                                  "opt": {"lr": 0.5}})
        self.assertEqual(self.resolve(p)["opt"],
                         {"lr": 0.5, "momentum": 0.9})

    def test_lists_replace_rather_than_append(self):
        """Appending would make the result depend on how many times a file is
        included, which is not something an author can reason about."""
        self.write("base.json", {"layers": [1, 2, 3]})
        p = self.write("c.json", {"include": ["base.json"], "layers": [9]})
        self.assertEqual(self.resolve(p)["layers"], [9])

    def test_a_cycle_is_refused_and_names_the_files(self):
        self.write("a.json", {"include": ["b.json"]})
        self.write("b.json", {"include": ["a.json"]})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(self.tmp / "a.json")
        self.assertIn("a.json", str(e.exception))
        self.assertIn("b.json", str(e.exception))

    def test_a_file_including_itself_is_refused(self):
        p = self.write("s.json", {"include": ["s.json"]})
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_a_missing_include_is_refused_by_name(self):
        p = self.write("c.json", {"include": ["nope.json"]})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("nope.json", str(e.exception))

    def test_include_must_be_a_list(self):
        p = self.write("c.json", {"include": "base.json"})
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_an_include_escaping_the_root_is_refused(self):
        """Reaching outside the config tree makes the result depend on the
        machine's directory layout."""
        self.write("outside.json", {"x": 1})
        p = self.write("sub/c.json", {"include": ["../outside.json"]})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p, root=self.tmp / "sub")
        self.assertIn("outside", str(e.exception))


class TestSubstitution(Base):
    def test_a_variable_is_substituted(self):
        p = self.write("c.json", {"data": "${DATA}/train"})
        self.assertEqual(self.resolve(p, values={"DATA": "/mnt/d"})["data"],
                         "/mnt/d/train")

    def test_substitution_reaches_nested_values(self):
        p = self.write("c.json", {"a": {"b": ["${X}"]}})
        self.assertEqual(self.resolve(p, values={"X": "v"})["a"]["b"], ["v"])

    def test_several_variables_in_one_string(self):
        p = self.write("c.json", {"s": "${A}-${B}"})
        self.assertEqual(self.resolve(p, values={"A": "1", "B": "2"})["s"],
                         "1-2")

    def test_an_unset_variable_is_refused_by_name(self):
        p = self.write("c.json", {"data": "${DATA}/train"})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("DATA", str(e.exception))

    def test_the_error_says_where_the_variable_was(self):
        p = self.write("c.json", {"outer": {"inner": "${GONE}"}})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("outer", str(e.exception))
        self.assertIn("inner", str(e.exception))

    def test_the_environment_is_never_consulted(self):
        """**A config that absorbs the machine it ran on is not reproducible.**

        If `os.environ` were a fallback, the same config would resolve
        differently on two machines and the hash would stop meaning anything.
        Values must be passed explicitly.
        """
        p = self.write("c.json", {"data": "${RESOLVE_CONFIG_PROBE}"})
        os.environ["RESOLVE_CONFIG_PROBE"] = "leaked"
        self.addCleanup(os.environ.pop, "RESOLVE_CONFIG_PROBE", None)
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_a_marker_that_is_not_a_usable_reference_is_refused(self):
        """`${1bad}` and `${unclosed` look like references and are not.

        Left alone they would travel into the resolved config untouched, and
        the file would claim to be fully resolved while carrying something
        that was meant to be substituted.
        """
        for bad in ("${1bad}", "${unclosed", "${has-dash}", "${}"):
            with self.subTest(text=bad):
                p = self.write("c.json", {"s": bad})
                with self.assertRaises(rc_mod.ConfigError) as e:
                    self.resolve(p)
                self.assertIn("s", str(e.exception))

    def test_no_marker_survives_resolution(self):
        """Belt and braces: whatever the syntax, nothing may look unresolved."""
        p = self.write("c.json", {"s": "${A}"})
        out = json.dumps(self.resolve(p, values={"A": "ok"}))
        self.assertNotIn("${", out)

    def test_a_substituted_value_is_not_itself_expanded(self):
        """Otherwise the result depends on substitution order."""
        p = self.write("c.json", {"s": "${A}"})
        self.assertEqual(self.resolve(p, values={"A": "${B}", "B": "x"})["s"],
                         "${B}")

    def test_substitution_happens_after_includes(self):
        self.write("base.json", {"p": "${DATA}/x"})
        p = self.write("c.json", {"include": ["base.json"]})
        self.assertEqual(self.resolve(p, values={"DATA": "/d"})["p"], "/d/x")

    def test_keys_are_not_substituted(self):
        """Substituting keys would make the merge order unknowable."""
        p = self.write("c.json", {"${K}": 1})
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p, values={"K": "x"})
        self.assertIn("${K}", str(e.exception))


class TestRefusals(Base):
    """Every one of these is a value that would be lost or changed silently."""

    def test_duplicate_keys_are_refused(self):
        """json.loads keeps the last one without a word."""
        p = self.write("c.json", '{"x": 1, "x": 2}')
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("x", str(e.exception))

    def test_duplicate_keys_deeper_in_the_tree_are_refused(self):
        p = self.write("c.json", '{"a": {"b": 1, "b": 2}}')
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_nan_is_refused(self):
        p = self.write("c.json", '{"x": NaN}')
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("x", str(e.exception))

    def test_infinity_is_refused(self):
        p = self.write("c.json", '{"x": Infinity}')
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_a_non_object_at_the_top_is_refused(self):
        p = self.write("c.json", "[1, 2]")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_unparsable_input_is_refused(self):
        p = self.write("c.json", "{ not json")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    def test_a_missing_file_is_refused_by_name(self):
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(self.tmp / "absent.json")
        self.assertIn("absent.json", str(e.exception))

    def test_an_unknown_extension_is_refused(self):
        p = self.write("c.conf", "{}")
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn(".conf", str(e.exception))


class TestYamlAuthoring(Base):
    """YAML is for authoring and is optional. It must never fail quietly."""

    def test_the_absence_of_pyyaml_is_reported_not_ignored(self):
        """**Refusing is fine. Pretending the file was empty is not.**"""
        p = self.write("c.yaml", "seed: 0\n")
        real = rc_mod.yaml
        rc_mod.yaml = None
        self.addCleanup(setattr, rc_mod, "yaml", real)
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        msg = str(e.exception)
        self.assertIn("PyYAML", msg)
        self.assertIn("pip install", msg, "the message does not say how to fix it")

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_yaml_resolves_to_the_same_hash_as_the_json_spelling(self):
        y = self.write("c.yaml", "seed: 0\nlr: 0.1\n")
        j = self.write("c.json", '{"lr": 0.1, "seed": 0}')
        self.assertEqual(rc_mod.sha256_of_config(self.resolve(y)),
                         rc_mod.sha256_of_config(self.resolve(j)))

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_quoting_style_does_not_change_the_hash(self):
        a = self.write("a.yaml", 'name: x\n')
        b = self.write("b.yaml", 'name: "x"\n')
        self.assertEqual(rc_mod.sha256_of_config(self.resolve(a)),
                         rc_mod.sha256_of_config(self.resolve(b)))

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_comments_do_not_change_the_hash(self):
        a = self.write("a.yaml", "seed: 0\n")
        b = self.write("b.yaml", "# why this seed\nseed: 0\n")
        self.assertEqual(rc_mod.sha256_of_config(self.resolve(a)),
                         rc_mod.sha256_of_config(self.resolve(b)))

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_duplicate_yaml_keys_are_refused(self):
        """PyYAML keeps the last one silently."""
        p = self.write("c.yaml", "x: 1\nx: 2\n")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_a_yaml_date_is_refused_rather_than_stringified(self):
        """JSON cannot carry it, and guessing a format would be a decision
        made silently on the author's behalf."""
        p = self.write("c.yaml", "when: 2026-07-29\n")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_a_non_string_yaml_key_is_refused(self):
        """JSON would turn 1 into "1" and could collide with a real key."""
        p = self.write("c.yaml", "1: a\n")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_unparsable_yaml_is_refused_not_read_as_empty(self):
        """Mutation testing found this path untested.

        Swallowing the parse error and returning an empty mapping produces a
        config that resolves, hashes, and says nothing about what ran.
        """
        p = self.write("c.yaml", "a: [1, 2\nb: {\n")
        with self.assertRaises(rc_mod.ConfigError) as e:
            self.resolve(p)
        self.assertIn("c.yaml", str(e.exception))

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_an_empty_yaml_file_is_refused(self):
        """`None` is not a mapping, and must not become one quietly."""
        p = self.write("c.yaml", "")
        with self.assertRaises(rc_mod.ConfigError):
            self.resolve(p)

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_yaml_can_include_json_and_the_other_way_round(self):
        self.write("base.json", {"x": 1})
        p = self.write("c.yaml", 'include: ["base.json"]\ny: 2\n')
        self.assertEqual(self.resolve(p), {"x": 1, "y": 2})


class TestCommandLine(Base):
    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, str(BIN / "resolve-config.py"), *map(str, args)],
            capture_output=True, text=True)

    def test_it_writes_the_resolved_file_and_prints_the_hash(self):
        self.write("base.json", {"seed": 0})
        c = self.write("c.json", {"include": ["base.json"], "lr": 0.1})
        out = self.tmp / "resolved.json"
        r = self.run_tool("--config", c, "--out", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out.read_bytes(), b'{"lr":0.1,"seed":0}\n')
        self.assertIn(hashlib.sha256(out.read_bytes()).hexdigest(), r.stdout)

    def test_set_supplies_a_value(self):
        c = self.write("c.json", {"d": "${D}"})
        out = self.tmp / "r.json"
        r = self.run_tool("--config", c, "--out", out, "--set", "D=/x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(out.read_text())["d"], "/x")

    def test_a_malformed_set_is_refused(self):
        c = self.write("c.json", {"x": 1})
        r = self.run_tool("--config", c, "--out", self.tmp / "r.json",
                          "--set", "NOEQUALS")
        self.assertNotEqual(r.returncode, 0)

    def test_a_value_containing_an_equals_sign_survives(self):
        c = self.write("c.json", {"d": "${D}"})
        out = self.tmp / "r.json"
        self.run_tool("--config", c, "--out", out, "--set", "D=a=b")
        self.assertEqual(json.loads(out.read_text())["d"], "a=b")

    def test_failure_is_reported_and_nothing_is_written(self):
        """A half-written resolved config would be worse than none."""
        c = self.write("c.json", {"d": "${MISSING}"})
        out = self.tmp / "r.json"
        r = self.run_tool("--config", c, "--out", out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MISSING", r.stdout + r.stderr)
        self.assertFalse(out.exists(), "a failed run left a file behind")

    def test_it_is_deterministic_across_processes(self):
        c = self.write("c.json", {"b": 1, "a": [2, {"d": 4, "c": 3}]})
        a, b = self.tmp / "a.json", self.tmp / "b.json"
        self.run_tool("--config", c, "--out", a)
        self.run_tool("--config", c, "--out", b)
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_hash_only_mode_writes_nothing(self):
        c = self.write("c.json", {"x": 1})
        r = self.run_tool("--config", c, "--print-hash")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(),
                         hashlib.sha256(b'{"x":1}\n').hexdigest())


if __name__ == "__main__":
    unittest.main()
