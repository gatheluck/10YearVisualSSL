"""Compare full public LoRA and DPT runs at the precision printed in Figure 3."""

import json

from downstream.figure3_lora_configs import MODELS_FILE
from downstream.revised_reporting import aggregate


def compare_panel(cells):
    targets = json.loads(
        MODELS_FILE.with_name("figure3-adaptation-targets.json").read_bytes()
    )
    models = json.loads(MODELS_FILE.read_bytes())
    results = {}
    for model, tasks in targets.items():
        for task, tracks in tasks.items():
            for track, expected in tracks.items():
                results[model, task, track] = {
                    "model": model,
                    "task": task,
                    "track": track,
                    "expected": expected,
                    "status": "missing",
                    "completed_seeds": [],
                    "unit": "meters" if task == "nyuv2" else "percentage_points",
                }

    seen = set()
    for cell in cells:
        key = model, task, track = cell["model"], cell["task"], cell["track"]
        if key not in results or key in seen:
            raise ValueError("supply distinct cells from the Figure 3 adaptation panel")
        seen.add(key)
        row = results[key]
        runs = cell["runs"]
        if not isinstance(runs, list):
            raise TypeError("cell runs must be a list")
        if not runs:
            continue
        try:
            report = aggregate(
                runs,
                expected_seeds=[0, 1, 2],
                portable=True,
                panel_cell=(model, task, track, models[model]),
            )
        except ValueError:
            row.update(
                status="invalid", reason="inconsistent run list or seed configurations"
            )
            continue
        row.update(
            completed_seeds=report["completed_seeds"], per_seed=report["per_seed"]
        )
        if report["invalid_runs"]:
            row.update(status="invalid", invalid_runs=report["invalid_runs"])
        elif not report["complete"]:
            row["status"] = "incomplete"
        else:
            stats = report["paper_metric"]
            digits = 3 if task == "nyuv2" else 2
            observed = {
                "mean": format(stats["mean"], f".{digits}f"),
                "sd": format(stats["std"], f".{digits}f"),
            }
            row.update(
                observed=observed,
                status="match" if observed == row["expected"] else "mismatch",
            )
    return {"schema": "portable_figure3_comparison_v1", "cells": list(results.values())}
