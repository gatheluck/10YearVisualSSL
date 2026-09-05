#!/usr/bin/env python3
"""Aggregate a BASIC5 linear probe over seeds 0, 1, 2 into mean ± std.

BASIC5_FAIR_v1 rule `seed`: the linear probe is run under **seeds 0, 1, 2**
and reported as **mean ± std** (see docs/BASIC5_PROTOCOL.md). The port already
runs one seed per launch -- every `configs/linear_eval*.yaml` carries a `seed`,
and `bin/resolve-config.py --override seed=N` sets it per run. So the seed rule
is implemented in exactly **one** shared place -- here -- rather than by editing
each method's `evaluate_linear_*.py` (there are ~50, and one rule must not be
implemented fifty times).

This reads the per-seed run outputs a launch already writes:

- `run_manifest.json` -- the seed, the run status, the method, and the stage
  (`adapterlib` writes it; the seed and status are recorded there, not guessed);
- `metrics.json` -- the contract metrics (`final_linear_probe_top1_accuracy`
  and friends).

and produces one `aggregate.json` with, per comparable metric, its mean and its
**sample** standard deviation (ddof=1) across the three seeds, plus the per-seed
numbers kept verbatim so nothing is hidden behind the average.

The guards refuse a mean that would not be a BASIC5 number:

- the seed set must be **exactly** the declared set (default {0, 1, 2}) -- not a
  subset, superset, or repeat;
- every run must have `status: ok` -- a failed run is not averaged;
- all runs must be the same method and the `linear_eval` stage -- the seed rule
  is about the probe, and averaging across methods or stages mixes tasks;
- a metric must be present in **every** seed -- averaging a metric missing from
  one seed would divide a smaller sum by three and understate it in silence.

Standard-library-only, so it runs in the base environment and under CI without a
method venv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1
MANIFEST = "run_manifest.json"
METRICS = "metrics.json"
CANONICAL_SEEDS = frozenset({0, 1, 2})
REQUIRED_STAGE = "linear_eval"


def mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean of no values")
    return sum(values) / len(values)


def sample_stddev(values) -> float:
    """Sample standard deviation (ddof=1): the seeds are independent draws and
    the spread estimates the population, which is the error bar papers report
    over seeds. Needs at least two values -- a single run has no spread."""
    values = list(values)
    n = len(values)
    if n < 2:
        raise ValueError(
            f"sample standard deviation needs at least two values, got {n}")
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return variance ** 0.5


def read_run(run_dir) -> dict:
    """Read one per-seed run output directory into a normalised record.

    Raises if the manifest or metrics file is missing -- an unreadable run is
    reported, never skipped (a skipped seed would silently shrink the set)."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST
    metrics_path = run_dir / METRICS
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{run_dir}: no {MANIFEST} (not a run output)")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{run_dir}: no {METRICS} (run produced no "
                                "metrics to aggregate)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "seed": manifest["seed"],
        "status": manifest["status"],
        "method": manifest["method"],
        "stage": manifest["stage"],
        "config_sha256": manifest.get("config_sha256"),
        "metrics": dict(metrics_doc.get("metrics", {})),
    }


def aggregate(runs, expected_seeds=CANONICAL_SEEDS) -> dict:
    """Mean ± std over the per-seed run records, or raise on a guard.

    `runs` is a list of records shaped like `read_run` returns."""
    expected_seeds = set(expected_seeds)
    if not runs:
        raise ValueError("no runs to aggregate")

    # Guard: every run succeeded.
    failed = [r for r in runs if r["status"] != "ok"]
    if failed:
        seeds = ", ".join(str(r["seed"]) for r in failed)
        raise ValueError(
            f"cannot aggregate: seed(s) {seeds} did not succeed "
            f"(status != ok); a failed run is not a BASIC5 number")

    # Guard: one method, the linear_eval stage.
    methods = {r["method"] for r in runs}
    if len(methods) != 1:
        raise ValueError(
            f"cannot aggregate across methods {sorted(methods)}: the seed rule "
            "reports one method's probe, not a mix")
    stages = {r["stage"] for r in runs}
    if stages != {REQUIRED_STAGE}:
        raise ValueError(
            f"cannot aggregate stage(s) {sorted(stages)}: BASIC5's seed rule is "
            f"about the {REQUIRED_STAGE!r} probe")

    # Guard: exactly the declared seed set, with no repeats.
    seed_list = [r["seed"] for r in runs]
    if len(seed_list) != len(set(seed_list)):
        raise ValueError(
            f"cannot aggregate: seeds {sorted(seed_list)} contain a repeat; "
            "each of the declared seeds must appear exactly once")
    if set(seed_list) != expected_seeds:
        raise ValueError(
            f"cannot aggregate: seeds {sorted(seed_list)} are not the declared "
            f"set {sorted(expected_seeds)} (BASIC5 default is seeds 0, 1, 2)")

    # Guard: every metric present in every seed (no silent partial average).
    metric_sets = [set(r["metrics"]) for r in runs]
    common = set.intersection(*metric_sets)
    union = set.union(*metric_sets)
    if common != union:
        missing = union - common
        raise ValueError(
            f"cannot aggregate: metric(s) {sorted(missing)} are absent from "
            "some seeds; a metric must be present in every seed to be averaged")

    ordered = sorted(runs, key=lambda r: r["seed"])
    aggregate_block = {}
    for key in sorted(common):
        values = [r["metrics"][key] for r in ordered]
        aggregate_block[key] = {
            "mean": mean(values),
            "std": sample_stddev(values),
            "n": len(values),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "method": next(iter(methods)),
        "stage": REQUIRED_STAGE,
        "seeds": sorted(seed_list),
        "n": len(seed_list),
        "aggregate": aggregate_block,
        "per_seed": {str(r["seed"]): r["metrics"] for r in ordered},
        "source_config_sha256": sorted(
            r["config_sha256"] for r in ordered if r["config_sha256"]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", dest="runs", action="append", default=[],
                   metavar="DIR", required=True,
                   help="a per-seed run output directory (repeat once per "
                        "seed); each holds run_manifest.json and metrics.json")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="directory to write aggregate.json into")
    p.add_argument("--seeds", type=int, nargs="+", default=sorted(CANONICAL_SEEDS),
                   help="the declared seed set (default: 0 1 2)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = [read_run(d) for d in args.runs]
        result = aggregate(records, expected_seeds=set(args.seeds))
    except (ValueError, OSError) as exc:
        print(f"aggregate-seeds: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    top = result["aggregate"]
    for key in sorted(top):
        print(f"{key}: {top[key]['mean']:.2f} +/- {top[key]['std']:.2f} "
              f"(n={top[key]['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
