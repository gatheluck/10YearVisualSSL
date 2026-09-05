#!/usr/bin/env python3
"""Specification for bin/aggregate-seeds.py.

BASIC5_FAIR_v1 rule `seed` requires the linear probe to be run under
**seeds 0, 1, 2** and reported as **mean ± std**. A single run reports one
number; the protocol compares the mean over three seeds and its spread.

The port already runs one seed per launch (each config carries `seed`, and
`bin/resolve-config.py --override seed=N` sets it per run). So the rule is
implemented in exactly one shared place -- this aggregator -- rather than by
editing every method's `evaluate_linear_*.py` (there are ~50 of them). The
aggregator reads the per-seed run outputs each launch already writes
(`run_manifest.json` for the seed/status/method/stage, `metrics.json` for the
contract metrics) and produces one mean/std per comparable metric.

The guards below are the point: a mean over the wrong seed set, over a failed
run, over mixed methods/stages, or over a metric that is missing from one seed
is not a BASIC5 result. Each guard carries a positive and a negative control,
and `mutations/aggregate-seeds.json` proves none of them is vacuous.

Standard-library-only, so it runs in the base environment.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
TOOL = BIN / "aggregate-seeds.py"
_MOD_NAME = "aggregate_seeds_tool"


def tool():
    """Load bin/aggregate-seeds.py by path (its name is not importable).

    Fails loudly if the tool is absent -- a missing driver is a red test,
    never a silent skip.
    """
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    spec = importlib.util.spec_from_file_location(_MOD_NAME, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[_MOD_NAME]
        raise
    return mod


def _run_dir(root: Path, seed: int, *, top1: float, top5: float,
             status: str = "ok", method: str = "example_method",
             stage: str = "linear_eval",
             metrics: "dict | None" = None) -> Path:
    """A synthetic per-seed run output directory shaped exactly as a launch
    leaves it: a `run_manifest.json` and a `metrics.json` side by side."""
    d = root / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(json.dumps({
        "schema_version": 2, "method": method, "stage": stage,
        "status": status, "seed": seed,
        "config_sha256": f"sha-{seed}",
    }), encoding="utf-8")
    if metrics is None:
        metrics = {
            "final_linear_probe_top1_accuracy": top1,
            "final_linear_probe_top5_accuracy": top5,
        }
    (d / "metrics.json").write_text(json.dumps({
        "schema_version": 2, "metrics": metrics, "metrics_raw": {},
    }), encoding="utf-8")
    return d


class TestSampleStddev(unittest.TestCase):
    """Mean ± std uses the **sample** standard deviation (ddof=1): the seeds
    are three draws and the spread estimates the population, the convention
    papers report over independent seeds. A population std (ddof=0) would be
    smaller and would misstate the reported error bar."""

    def test_mean_of_three(self):
        self.assertAlmostEqual(tool().mean([70.0, 72.0, 74.0]), 72.0)

    def test_sample_stddev_of_three(self):
        # variance = ((-2)^2 + 0 + 2^2) / (3 - 1) = 8/2 = 4 -> std = 2.0
        self.assertAlmostEqual(tool().sample_stddev([70.0, 72.0, 74.0]), 2.0)

    def test_sample_stddev_needs_at_least_two(self):
        with self.assertRaises(ValueError):
            tool().sample_stddev([1.0])


class TestAggregateHappyPath(unittest.TestCase):
    def _runs(self):
        return [
            {"seed": 0, "status": "ok", "method": "example_method",
             "stage": "linear_eval", "config_sha256": "a",
             "metrics": {"final_linear_probe_top1_accuracy": 70.0,
                         "final_linear_probe_top5_accuracy": 90.0}},
            {"seed": 1, "status": "ok", "method": "example_method",
             "stage": "linear_eval", "config_sha256": "b",
             "metrics": {"final_linear_probe_top1_accuracy": 72.0,
                         "final_linear_probe_top5_accuracy": 91.0}},
            {"seed": 2, "status": "ok", "method": "example_method",
             "stage": "linear_eval", "config_sha256": "c",
             "metrics": {"final_linear_probe_top1_accuracy": 74.0,
                         "final_linear_probe_top5_accuracy": 92.0}},
        ]

    def test_it_reports_mean_and_std_per_metric(self):
        agg = tool().aggregate(self._runs())
        top1 = agg["aggregate"]["final_linear_probe_top1_accuracy"]
        self.assertAlmostEqual(top1["mean"], 72.0)
        self.assertAlmostEqual(top1["std"], 2.0)
        self.assertEqual(top1["n"], 3)
        top5 = agg["aggregate"]["final_linear_probe_top5_accuracy"]
        self.assertAlmostEqual(top5["mean"], 91.0)
        self.assertAlmostEqual(top5["std"], 1.0)

    def test_it_records_the_seeds_and_method_and_stage(self):
        agg = tool().aggregate(self._runs())
        self.assertEqual(sorted(agg["seeds"]), [0, 1, 2])
        self.assertEqual(agg["n"], 3)
        self.assertEqual(agg["method"], "example_method")
        self.assertEqual(agg["stage"], "linear_eval")

    def test_it_keeps_the_per_seed_numbers(self):
        agg = tool().aggregate(self._runs())
        self.assertAlmostEqual(
            agg["per_seed"]["1"]["final_linear_probe_top1_accuracy"], 72.0)


class TestSeedSetGuard(unittest.TestCase):
    """The mean must be over exactly seeds 0, 1, 2 -- not a subset, not a
    superset, not a repeat. Averaging the wrong set is not a BASIC5 number."""

    def _base(self):
        return TestAggregateHappyPath()._runs()

    def test_negative_the_canonical_set_is_accepted(self):
        tool().aggregate(self._base())  # 0,1,2 -- must not raise

    def test_positive_a_missing_seed_is_refused(self):
        runs = self._base()[:2]  # only 0,1
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_positive_a_duplicate_seed_is_refused(self):
        runs = self._base()
        runs[2]["seed"] = 1  # 0,1,1
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_positive_a_repeated_seed_within_the_set_is_refused(self):
        # 0,1,2,2 -- the *set* is exactly {0,1,2}, so only the repeat guard
        # (not the seed-set guard) can catch this. Isolates that guard: a
        # run output passed twice must not be averaged as two seeds.
        runs = self._base()
        extra = dict(runs[2])
        extra["config_sha256"] = "c2"
        runs.append(extra)  # seeds 0,1,2,2
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_positive_an_out_of_set_seed_is_refused(self):
        runs = self._base()
        runs[2]["seed"] = 42  # 0,1,42
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_a_declared_seed_set_can_differ_but_must_match(self):
        runs = self._base()
        # declaring {0,1,2} matches; declaring {5,6,7} does not
        tool().aggregate(runs, expected_seeds={0, 1, 2})
        with self.assertRaises(ValueError):
            tool().aggregate(runs, expected_seeds={5, 6, 7})


class TestFailedRunGuard(unittest.TestCase):
    def test_a_failed_run_is_refused(self):
        runs = TestAggregateHappyPath()._runs()
        runs[1]["status"] = "failed"
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_all_ok_is_accepted(self):
        tool().aggregate(TestAggregateHappyPath()._runs())  # must not raise


class TestMixedRunGuard(unittest.TestCase):
    def test_mixed_methods_are_refused(self):
        runs = TestAggregateHappyPath()._runs()
        runs[2]["method"] = "other_method"
        with self.assertRaises(ValueError):
            tool().aggregate(runs)

    def test_a_non_linear_eval_stage_is_refused(self):
        runs = TestAggregateHappyPath()._runs()
        for r in runs:
            r["stage"] = "pretrain"
        with self.assertRaises(ValueError):
            tool().aggregate(runs)


class TestMissingMetricGuard(unittest.TestCase):
    """A metric present in some seeds but not all cannot be averaged -- doing
    so would divide a smaller sum by three, understating it in silence."""

    def test_a_metric_missing_from_one_seed_is_refused(self):
        runs = TestAggregateHappyPath()._runs()
        del runs[1]["metrics"]["final_linear_probe_top5_accuracy"]
        with self.assertRaises(ValueError):
            tool().aggregate(runs)


class TestReadRun(unittest.TestCase):
    def test_it_reads_seed_status_method_stage_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _run_dir(Path(tmp), 0, top1=70.0, top5=90.0)
            r = tool().read_run(d)
            self.assertEqual(r["seed"], 0)
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["method"], "example_method")
            self.assertEqual(r["stage"], "linear_eval")
            self.assertAlmostEqual(
                r["metrics"]["final_linear_probe_top1_accuracy"], 70.0)

    def test_a_run_dir_without_a_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "nope"
            empty.mkdir()
            with self.assertRaises(Exception):
                tool().read_run(empty)


class TestEndToEnd(unittest.TestCase):
    def test_it_reads_three_run_dirs_and_writes_an_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [
                _run_dir(root, 0, top1=70.0, top5=90.0),
                _run_dir(root, 1, top1=72.0, top5=91.0),
                _run_dir(root, 2, top1=74.0, top5=92.0),
            ]
            out = root / "agg"
            rc = tool().main(["--run", str(dirs[0]), "--run", str(dirs[1]),
                              "--run", str(dirs[2]), "--out", str(out)])
            self.assertEqual(rc, 0)
            written = json.loads((out / "aggregate.json").read_text())
            top1 = written["aggregate"]["final_linear_probe_top1_accuracy"]
            self.assertAlmostEqual(top1["mean"], 72.0)
            self.assertAlmostEqual(top1["std"], 2.0)
            self.assertEqual(sorted(written["seeds"]), [0, 1, 2])

    def test_the_cli_refuses_the_wrong_seed_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [
                _run_dir(root, 0, top1=70.0, top5=90.0),
                _run_dir(root, 1, top1=72.0, top5=91.0),
            ]
            out = root / "agg"
            rc = tool().main(["--run", str(dirs[0]), "--run", str(dirs[1]),
                              "--out", str(out)])
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
