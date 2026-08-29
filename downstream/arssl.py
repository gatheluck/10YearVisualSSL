"""The ARSSL downstream harness: one frozen backbone, the task battery, aggregated.

    python -m downstream.arssl --config <resolved.json> --out <dir>

A1 of the Step-3 plan (docs/STEP3_PORTING_PLAN.md). It ports the *shape* of the
capture's ARSSL evaluation driver
(`origin/snapshots:methods_step3/ARSSL/src/run_eval.py`): drive one **frozen
backbone** through a battery of downstream task probes and aggregate the per-task
numbers into a single result.

This is a **thin driver only** (the plan's chosen scope). It re-uses the task
runners already in this repo (`downstream/{ade20k,coco,nyuv2,ssv2}.py`) instead
of re-implementing any task head -- one implementation, invoked, never copied
(CLAUDE.md; the same stance `bin/matrix-run.py` takes toward `launch.py`). Each
task is driven as a subprocess and checked with the downstream contract
(`downstream.contract`), and the ARSSL result is `ok` only when every selected
task is `ok`. Because it drives subprocesses, the driver itself needs no torch:
it is pure standard library, so its discovery, config composition and
aggregation run in the base environment; only the real task-runner subprocess
needs torch + timm.

The ImageNet columns the capture also evaluates are deliberately **not** re-homed
here: ImageNet-1k stays the per-method `linear_eval` (docs/DOWNSTREAM.md), and
ImageNet-100 is not yet ported. Those are later plan items, not A1.

Task runners are **discovered, not listed**: a `downstream/*.py` module is a task
runner iff it declares, at module level, a string `TASK` constant and both a
`run` and a `main` function. The match is structural (the module's AST is
parsed), never a substring over the file text.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DOWNSTREAM_DIR = Path(__file__).resolve().parent
ROOT = DOWNSTREAM_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from downstream import contract                                     # noqa: E402

TASK = "arssl"

TOP_KEYS = frozenset({"task", "seed", "device", "backbone", "tasks"})
# Set once, for the whole battery, at the top level -- a task may not carry its
# own, so every task probes the identical frozen backbone at the same seed.
SHARED_KEYS = ("seed", "device", "backbone")
DEVICES = ("auto", "cuda", "cpu")


class ConfigError(Exception):
    """A refusal, always naming what was refused."""


def _named(missing, unknown, where: str) -> None:
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(sorted(missing))}")
    if unknown:
        raise ConfigError(f"{where}: unknown {', '.join(sorted(unknown))}")


def _module_task_key(path: Path) -> "str | None":
    """The module-level `TASK` string of a task-runner module, else None.

    A module qualifies only when it has, at module level, a `TASK = "..."`
    assignment (a string constant) *and* both a `run` and a `main` function.
    Nested definitions and non-string / non-module-level `TASK` names do not
    count -- the AST is read structurally, so a substring like `TASK` inside
    another name or a string never matches."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    task_value = None
    funcs = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "TASK"
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    task_value = node.value.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)
    if task_value is not None and {"run", "main"} <= funcs:
        return task_value
    return None


def discover_tasks(package_dir: Path = DOWNSTREAM_DIR) -> "dict[str, str]":
    """Map each discovered task key to its module stem (e.g. ``ade20k``).

    Discovers by structure (`_module_task_key`), so a new task runner joins the
    ARSSL battery with no edit here. The driver module itself and dunder /
    private modules are skipped."""
    package_dir = Path(package_dir)
    self_stem = Path(__file__).stem
    found: dict[str, str] = {}
    for path in sorted(package_dir.glob("*.py")):
        if path.stem == self_stem or path.stem.startswith("_"):
            continue
        key = _module_task_key(path)
        if key is not None:
            found[key] = path.stem
    return found


def validate_config(cfg: dict, known_tasks: "dict[str, str] | None" = None) -> None:
    """Refuse a malformed ARSSL config, naming what was refused.

    The battery is one frozen backbone at one seed/device (the top level); each
    entry under `tasks` names a *discovered* downstream task and carries only
    that task's own keys (`data_root` plus `probe`/`detector`/...). An unknown
    task is a hard refusal, never a silent skip. The per-task probe/detector
    schema is checked by the task runner itself, not duplicated here."""
    known = discover_tasks() if known_tasks is None else known_tasks
    for key in ("output", "out", "result_dir"):
        if key in cfg:
            raise ConfigError(
                f"config: {key} is set; the output location is fixed at --out")
    _named(TOP_KEYS - set(cfg), set(cfg) - TOP_KEYS, "config")
    if cfg["device"] not in DEVICES:
        raise ConfigError(f"config: device is {cfg['device']!r}; expected "
                          f"{', '.join(DEVICES)}")
    tasks = cfg["tasks"]
    if not isinstance(tasks, dict) or not tasks:
        raise ConfigError(
            "config: tasks is empty; the ARSSL battery would have nothing to run")
    shared = set(SHARED_KEYS) | {"task"}
    for key, sub in tasks.items():
        if key not in known:
            raise ConfigError(
                f"config.tasks: {key!r} is not a discovered downstream task; "
                f"known: {', '.join(sorted(known)) or '(none)'}")
        if not isinstance(sub, dict):
            raise ConfigError(f"config.tasks.{key}: not a mapping")
        clash = set(sub) & shared
        if clash:
            raise ConfigError(
                f"config.tasks.{key}: {', '.join(sorted(clash))} is shared "
                "across the battery; set it at the top level, not per task")


def compose_task_config(cfg: dict, task_key: str) -> dict:
    """The resolved config the task runner expects: the task's own keys, with the
    shared backbone/seed/device overlaid and `task` set to the task key."""
    composed = dict(cfg["tasks"][task_key])
    composed["task"] = task_key
    for key in SHARED_KEYS:
        composed[key] = cfg[key]
    return composed


def aggregate(cells: "list[dict]") -> dict:
    """Combine per-task cells into the ARSSL verdict + comparison table.

    The verdict is `ok` only when there is at least one task and every one is
    `ok` -- "nothing ran" never reads as success (as in `bin/matrix-run.py`).
    The comparable metrics of every task are unioned into one table; two tasks
    claiming the same metric name is a hard error, since the union would
    silently drop one."""
    status = "ok" if cells and all(c["status"] == "ok" for c in cells) \
        else "failed"
    metrics: dict = {}
    for cell in cells:
        for name, value in (cell.get("metrics") or {}).items():
            if name in metrics:
                raise ValueError(
                    f"two tasks report metric {name!r}; one would be lost")
            metrics[name] = value
    tasks = {cell["task"]: {k: v for k, v in cell.items() if k != "task"}
             for cell in cells}
    return {"status": status, "metrics": metrics, "tasks": tasks}


def run_task(task_key: str, module: str, cfg_path: Path, sub_out: Path,
             python: "str | None" = None,
             device_override: "str | None" = None) -> dict:
    """Drive one task runner as a subprocess and read its verdict via the
    downstream contract. Never raises for a failed run: a failure is a cell with
    status != "ok" carrying its diagnostics, so the battery reports it rather
    than aborting (as the capture harness's per-task try/except does)."""
    cmd = [python or sys.executable, "-m", f"downstream.{module}",
           "--config", str(cfg_path), "--out", str(sub_out)]
    if device_override:
        cmd += ["--device", device_override]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                          text=True)
    ok, violations = contract.verify(sub_out, cfg_path, proc.returncode)
    cell = {
        "task": task_key,
        "module": module,
        "out": sub_out.name,
        "returncode": proc.returncode,
        "status": "ok" if ok else "failed",
        "metrics": None,
    }
    if ok:
        doc = json.loads((sub_out / contract.METRICS).read_text(encoding="utf-8"))
        cell["metrics"] = doc.get("metrics", {})
    else:
        # A failed task must carry its reason. The runner records the structured
        # error in its manifest (not on stderr), so read that first, then fall
        # back to the subprocess output and finally the contract violations, so
        # the cell is never a silent empty failure.
        reason = ""
        man = sub_out / contract.MANIFEST
        if man.is_file():
            try:
                reason = json.loads(man.read_text(encoding="utf-8")).get("error") \
                    or ""
            except ValueError:
                reason = ""
        tail = (proc.stdout[-1500:] + proc.stderr[-1500:]).strip()
        cell["error"] = reason or tail or (
            f"exit {proc.returncode}; " + "; ".join(violations))
        cell["violations"] = violations
    return cell


def run(cfg: dict, out: Path, python: "str | None" = None,
        device_override: "str | None" = None) -> dict:
    """Drive the whole ARSSL battery over one frozen backbone and aggregate.

    Each task runs in its own subdirectory of `out`, so its contract artifacts
    are self-contained; the aggregate `arssl_results.json` and the top-level
    contract metrics are written after every task has run."""
    out = Path(out)
    validate_config(cfg)
    known = discover_tasks()
    cells = []
    for task_key in cfg["tasks"]:                      # insertion order preserved
        module = known[task_key]
        sub_out = out / module
        sub_out.mkdir(parents=True, exist_ok=True)
        resolved = compose_task_config(cfg, task_key)
        cfg_path = sub_out / "config.json"
        cfg_path.write_text(json.dumps(resolved, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"ARSSL: {task_key} -> downstream.{module}")
        cells.append(run_task(task_key, module, cfg_path, sub_out, python,
                              device_override))
    combined = aggregate(cells)
    combined["seed"] = int(cfg["seed"])
    combined["backbone"] = cfg["backbone"]
    (out / "arssl_results.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # The comparison-table metrics are already downstream-vocabulary names.
    contract.write_metrics(out, combined["metrics"],
                           {name: name for name in combined["metrics"]})
    return combined


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None, choices=[None, *DEVICES])
    parser.add_argument("--python", default=None,
                        help="interpreter for each task subprocess (default: "
                             "this one); point it at the downstream venv")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(args.config).read_bytes()
    cfg = json.loads(config_bytes)
    method_ref = str(cfg.get("backbone", {}).get("encoder") or "random-smoke")
    started = _now()
    error = None
    try:
        combined = run(cfg, out, python=args.python, device_override=args.device)
        status = combined["status"]
    except ConfigError as exc:
        print(f"  *** {exc}", file=sys.stderr)
        # A refused config is misuse, not a run result: no manifest, exit 2.
        return 2
    except Exception:                       # a run failure is a result
        import traceback
        error = traceback.format_exc(limit=8).strip()
        status = "failed"
    contract.write_manifest(
        out, task=TASK, method_ref=method_ref, status=status,
        config_sha256=contract.sha256_bytes(config_bytes),
        started_at=started, finished_at=_now(), seed=int(cfg.get("seed", 0)),
        backbone=cfg.get("backbone", {}), error=error)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
