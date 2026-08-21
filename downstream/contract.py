"""The downstream contract: write a machine-checkable result, and check it.

The method contract (`adapterlib` + `bin/contract-test.py`) decides "the port is
finished" for a per-method run. Downstream tasks are cross-method evaluations, so
they get a sibling contract of the **same shape** — a run writes `run_manifest.json`
(status, config hash, every output listed and hashed) and `metrics.json` (numeric
metrics from a downstream vocabulary) into `--out`, and `verify()` decides the task
ran by machine: exit status 0 **and** `status: ok`, neither trusted alone.

It is a separate vocabulary and checker, not a reuse of the method one, because
`adapterlib.METRIC_VOCABULARY` is scoped to `methods/*/adapter` (every name there
must be produced by a method — `tests/test_metric_vocabulary.py`), and a
cross-method task metric like `ade20k_miou` is produced by no method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = "run_manifest.json"
METRICS = "metrics.json"
SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1

COMPARABLE = "comparable"
PER_TASK = "per-task"

# The downstream metric vocabulary. A name means the same thing across every
# method's backbone (the task is fixed), so the task metrics are comparable;
# counters are per-task bookkeeping. A task runner may write only these names.
# Add a name together with the runner that writes it, never ahead of one.
DOWNSTREAM_METRICS = {
    "ade20k_miou": COMPARABLE,
    "ade20k_pixel_accuracy": COMPARABLE,
    "epochs_completed": PER_TASK,
    "metrics_unavailable": PER_TASK,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def collect_artifacts(out: Path) -> list[dict]:
    """Every file under `out` except the manifest itself, hashed.

    An unlisted output is a hole in reproducibility, so the manifest names them
    all and `verify()` refuses any file it did not list."""
    out = Path(out)
    artifacts = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == MANIFEST:
            continue
        artifacts.append({
            "path": str(path.relative_to(out)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return artifacts


def write_metrics(out: Path, raw: dict, names: dict) -> None:
    """Write `metrics.json`: contract names (from the downstream vocabulary) and
    the runner's own raw names, checked here so a bad number fails before a
    checker has to spend anything."""
    contract: dict = {}
    for key, value in raw.items():
        if not _is_number(value):
            raise ValueError(f"metric {key!r} is {value!r}, not a number")
        if key not in names:
            raise ValueError(
                f"metric {key!r} has no entry in the runner's translation table; "
                "give it a downstream name or None to keep it raw-only")
        target = names[key]
        if target is None:
            continue
        if target not in DOWNSTREAM_METRICS:
            raise ValueError(
                f"{target!r} is not in the downstream vocabulary; known: "
                + ", ".join(sorted(DOWNSTREAM_METRICS)))
        if target in contract:
            raise ValueError(f"two metrics map to {target!r}; one would be lost")
        contract[target] = value
    (Path(out) / METRICS).write_text(
        json.dumps({"schema_version": METRICS_SCHEMA_VERSION,
                    "metrics": contract, "metrics_raw": dict(raw)},
                   sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")


def write_manifest(out: Path, *, task: str, method_ref: str, status: str,
                   config_sha256: str, started_at: str, finished_at: str,
                   seed: int, backbone: dict, error: str | None = None) -> None:
    """Write `run_manifest.json` last, after every other output exists, so its
    artifact list is complete."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "method_ref": method_ref,
        "status": status,
        "config_sha256": config_sha256,
        "started_at": started_at,
        "finished_at": finished_at,
        "seed": seed,
        "backbone": backbone,
        "artifacts": collect_artifacts(Path(out)),
    }
    if error is not None:
        manifest["error"] = error
    (Path(out) / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")


def verify(out: Path, config: Path, exit_status: int) -> tuple[bool, list[str]]:
    """Decide the task ran, by machine. Returns (ok, violations).

    ok is exit_status == 0 AND manifest status == "ok" AND every check passes;
    neither signal is trusted alone (a task that crashes after writing an ok
    manifest, or one that writes nothing but exits 0, is caught)."""
    out = Path(out)
    v: list[str] = []
    manifest_path = out / MANIFEST
    if not manifest_path.is_file():
        return False, [f"no {MANIFEST} in {out}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return False, [f"{MANIFEST} is not JSON: {exc}"]

    for field in ("schema_version", "task", "method_ref", "status",
                  "config_sha256", "seed", "artifacts"):
        if field not in manifest:
            v.append(f"manifest is missing {field!r}")
    status = manifest.get("status")
    if status not in ("ok", "failed"):
        v.append(f"status is {status!r}, not ok/failed")

    config_bytes = Path(config).read_bytes()
    if manifest.get("config_sha256") != sha256_bytes(config_bytes):
        v.append("config_sha256 does not match the supplied config")

    listed = {a["path"] for a in manifest.get("artifacts", [])}
    for a in manifest.get("artifacts", []):
        p = out / a["path"]
        if not p.is_file():
            v.append(f"listed artifact missing: {a['path']}")
            continue
        if sha256_file(p) != a.get("sha256") or p.stat().st_size != a.get("bytes"):
            v.append(f"artifact changed since it was listed: {a['path']}")
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != MANIFEST:
            rel = str(path.relative_to(out))
            if rel not in listed:
                v.append(f"unlisted file under --out: {rel}")

    if status == "ok":
        metrics_path = out / METRICS
        if not metrics_path.is_file():
            v.append(f"status ok but no {METRICS}")
        else:
            doc = json.loads(metrics_path.read_text(encoding="utf-8"))
            if doc.get("schema_version") != METRICS_SCHEMA_VERSION:
                v.append("metrics schema_version mismatch")
            for name, value in doc.get("metrics", {}).items():
                if name not in DOWNSTREAM_METRICS:
                    v.append(f"metric {name!r} is not in the downstream vocabulary")
                if not _is_number(value):
                    v.append(f"metric {name!r} is not a number: {value!r}")
            if "metrics_raw" not in doc:
                v.append("metrics.json has no metrics_raw")

    if exit_status != 0:
        v.append(f"exit status was {exit_status}, not 0")
    ok = not v and status == "ok" and exit_status == 0
    return ok, v
