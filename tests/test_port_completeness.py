"""Every port carries the machinery that proves it, and every stage it declares
ships a config to run it.

Two structural gaps went unnoticed until an audit found them by hand, which is
the failure this file turns into machinery (CLAUDE.md, "make it machinery"):

- `01_context_prediction` was the only method of forty with **no mutation
  spec** -- its guards were never proved non-vacuous. Nothing required one.
- `20_simsiam` declared a `linear_eval` stage but shipped **no
  `configs/linear_eval.yaml`** -- the only two-stage method that could not be
  run from a shipped config. Its test built the config inline, so nothing on
  disk revealed the gap.

The rules here **discover, never list**: they read the method directories and
each adapter's declared `STAGES`, so a newly added port is held to the same
bar without editing this file. Each detector carries a positive control (it
fires on a planted gap) and a negative control (it does not fire on a
well-formed method), because a guard that cannot fail is not a guard.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"
MUTATIONS_DIR = ROOT / "mutations"

# Stages that are runnable training/eval work and therefore need a shipped
# config. `knowledge_transfer` is a real stage (09_jigsaw_puzzle_pp) but is not
# required to ship its own top-level config, so it is not in this set.
CONFIG_STAGES = ("step1", "linear_eval")


def discover_methods() -> list[str]:
    """The method directories, discovered. `_reference` is scaffolding."""
    return sorted(
        d.name for d in METHODS_DIR.iterdir()
        if d.is_dir() and d.name != "_reference"
        and (d / "adapter" / "__init__.py").is_file())


def stages_of(method: str, methods_dir: Path = METHODS_DIR) -> list[str]:
    """The stage names a method's adapter declares, read from the `STAGES`
    assignment via AST.

    Read structurally, not by import: importing forty adapters in one process
    collides on the shared `models`/`data`/`adapter` package names, and the
    first port's `STAGES` values are `frozenset(...)` calls that `literal_eval`
    cannot take. Only the stage *names* are needed, and for a dict those are the
    keys and for a tuple/list the elements -- both readable without evaluating
    values.
    """
    src = (methods_dir / method / "adapter" / "__init__.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "STAGES"
                   for t in node.targets):
            continue
        val = node.value
        if isinstance(val, ast.Dict):
            return [k.value for k in val.keys
                    if isinstance(k, ast.Constant)]
        if isinstance(val, (ast.Tuple, ast.List)):
            return [e.value for e in val.elts if isinstance(e, ast.Constant)]
    raise AssertionError(f"{method}: no module-level STAGES assignment found")


# --- detectors (pure, so they can be controlled) --------------------------

def specs_for(method: str, mutations_dir: Path) -> list[Path]:
    """The mutation specs belonging to a method: `mutations/<method>-*.json`.

    Matched on the `<method>-` prefix (whole method name plus a separator), not
    a substring, so `mar` does not swallow `mar`-prefixed siblings only, and
    `01_context_prediction` is never matched by a bare `01`.
    """
    return sorted(Path(p) for p in glob.glob(
        str(mutations_dir / f"{method}-*.json")))


def methods_missing_specs(methods: list[str], mutations_dir: Path) -> list[str]:
    return [m for m in methods if not specs_for(m, mutations_dir)]


def spec_problems(spec_path: Path) -> list[str]:
    """What is wrong with one mutation spec, or an empty list.

    A spec that exists but was never measured, or whose `_result` does not
    account for every target, is a spec that proves nothing.
    """
    d = json.loads(spec_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    targets = d.get("targets")
    if not isinstance(targets, list) or not targets:
        problems.append("no targets")
        targets = []
    for i, t in enumerate(targets):
        missing = [k for k in ("label", "file", "old", "new", "tests")
                   if k not in t]
        if missing:
            problems.append(f"target {i} missing {', '.join(missing)}")
    r = d.get("_result") or {}
    killed, total = r.get("killed"), r.get("total")
    equivalent = r.get("equivalent", [])
    if not r.get("measured_on"):
        problems.append("no measured_on: a result no one measured")
    if killed is None or total is None:
        problems.append("no killed/total in _result")
    else:
        if total != len(targets):
            problems.append(f"_result total {total} != {len(targets)} targets")
        if killed + len(equivalent) != total:
            problems.append(
                f"killed {killed} + equivalent {len(equivalent)} != total {total}")
        if killed == 0 and not equivalent:
            problems.append("nothing was killed")
    return problems


def methods_missing_stage_config(
        methods: list[str], methods_dir: Path = METHODS_DIR
) -> list[tuple[str, str]]:
    """(method, stage) pairs where a declared config-stage ships no config.

    `step1` accepts any `configs/step1*.yaml` (the VAE ships a MNIST-named one);
    `linear_eval` requires `configs/linear_eval.yaml` exactly.
    """
    gaps: list[tuple[str, str]] = []
    for m in methods:
        cfg_dir = methods_dir / m / "configs"
        stages = stages_of(m, methods_dir)
        if "step1" in stages and not list(cfg_dir.glob("step1*.yaml")):
            gaps.append((m, "step1"))
        if "linear_eval" in stages and not (cfg_dir / "linear_eval.yaml").is_file():
            gaps.append((m, "linear_eval"))
    return gaps


# --- the guards, applied to the real repository ---------------------------

class TestEveryMethodHasAMeasuredMutationSpec(unittest.TestCase):
    def test_every_method_has_at_least_one_spec(self):
        missing = methods_missing_specs(discover_methods(), MUTATIONS_DIR)
        self.assertEqual(
            missing, [],
            f"methods with no mutation spec (their guards are unproven): "
            f"{missing}")

    def test_every_spec_is_measured_and_accounts_for_its_targets(self):
        bad = {}
        for m in discover_methods():
            for spec in specs_for(m, MUTATIONS_DIR):
                probs = spec_problems(spec)
                if probs:
                    bad[spec.name] = probs
        self.assertEqual(bad, {}, f"mutation specs proving nothing: {bad}")


class TestEveryConfigStageIsShipped(unittest.TestCase):
    def test_every_declared_config_stage_ships_a_config(self):
        gaps = methods_missing_stage_config(discover_methods())
        self.assertEqual(
            gaps, [],
            f"declared stages with no shipped config (cannot be run from "
            f"disk): {gaps}")


# --- controls: a detector that cannot fail is not a detector --------------

class TestTheDetectorsActuallyFire(unittest.TestCase):
    """Controls discover names; they never spell a method out. A literal method
    name in a shared file is exactly what tests/test_no_hard_coded_methods.py
    refuses, and a list looks right until the next method arrives."""

    def test_missing_spec_detector_fires_and_stays_quiet(self):
        # positive: a name absent from mutations/ is reported
        self.assertEqual(
            methods_missing_specs(["ghost_method"], MUTATIONS_DIR),
            ["ghost_method"])
        # negative: a real, discovered method is not reported (it has a spec)
        a_real_method = discover_methods()[0]
        self.assertEqual(
            methods_missing_specs([a_real_method], MUTATIONS_DIR), [])

    def test_spec_problem_detector_fires_and_stays_quiet(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        # positive: an unmeasured, target-less spec is flagged
        planted = tmp / "x-planted.json"
        planted.write_text(json.dumps({"targets": []}), encoding="utf-8")
        self.assertTrue(spec_problems(planted))
        # positive: a spec whose _result under-counts its targets is flagged
        undercount = tmp / "x-undercount.json"
        undercount.write_text(json.dumps({
            "targets": [{"label": "a", "file": "f", "old": "o", "new": "n",
                         "tests": ["t"]}],
            "_result": {"measured_on": "2026-01-01", "killed": 0, "total": 1,
                        "equivalent": []}}), encoding="utf-8")
        self.assertIn("nothing was killed",
                      " ".join(spec_problems(undercount)))
        # negative: a well-formed, measured spec is silent
        good = tmp / "x-good.json"
        good.write_text(json.dumps({
            "targets": [{"label": "a", "file": "f", "old": "o", "new": "n",
                         "tests": ["t"]}],
            "_result": {"measured_on": "2026-01-01", "killed": 1, "total": 1,
                        "equivalent": []}}), encoding="utf-8")
        self.assertEqual(spec_problems(good), [])

    def test_config_stage_detector_fires_and_stays_quiet(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        # positive: a planted method declaring both config-stages but shipping
        # no config is reported for each stage. Built on disk so the detector
        # exercises its real AST + glob path, with no method named here.
        planted = tmp / "planted"
        (planted / "adapter").mkdir(parents=True)
        (planted / "configs").mkdir()
        (planted / "adapter" / "__init__.py").write_text(
            'STAGES = ("step1", "linear_eval")\n', encoding="utf-8")
        gaps = methods_missing_stage_config(["planted"], methods_dir=tmp)
        self.assertEqual(sorted(gaps),
                         [("planted", "linear_eval"), ("planted", "step1")])
        # negative: shipping the two configs silences it
        (planted / "configs" / "step1.yaml").write_text("stage: step1\n")
        (planted / "configs" / "linear_eval.yaml").write_text(
            "stage: linear_eval\n")
        self.assertEqual(
            methods_missing_stage_config(["planted"], methods_dir=tmp), [])


if __name__ == "__main__":
    unittest.main()
