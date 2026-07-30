"""The one place a `run_manifest.json` is written.

Ten years of methods means ten years of incompatible environments, so every
adapter is its own process (CONTRACT section 1). What must **not** be repeated
per method is the manifest logic. Scanners that each reimplemented the same
classification disagreed with each other and produced false reports; that is
the common root of past defects here, and the rule against implementing one
rule twice comes from it.

So an adapter supplies only the part that is actually method-specific:

    import adapterlib

    def body(ctx):
        ...                                   # train, evaluate
        (ctx.out / "encoder.pt").write_bytes(weights)
        ctx.write_metrics({"top1": 42.5})

    raise SystemExit(adapterlib.run(
        config=args.config, out=args.out,
        method="<this method>", stage="step1", body=body))

Everything the contract requires -- times, hashes, the artifact listing, both
success signals -- is produced here, once.

**Standard library only.** This has to import inside every method
environment, and measurement of 64 captured environment definitions found
Python 3.10 in 62 of them and 3.12 in 2. Nothing installed may be assumed, and
no syntax newer than 3.10 may be used.

Two kinds of failure, kept distinct:

- **The run failed.** Recorded as `status: "failed"` with the reason, and a
  non-zero return. Both signals are written together here, so they cannot
  disagree
- **The adapter misused this module** -- no seed in the config, an encoder
  that vanished with no explanation. That is a mistake in the port, not a
  result, so it raises `AdapterError` and no manifest is written. The contract
  already says an absent manifest means failure
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA_VERSION = 1
MANIFEST = "run_manifest.json"
ENCODER = "encoder.pt"
METRICS = "metrics.json"

# The contract fixes these two names and their roles. Anything else an adapter
# writes is still listed -- an unlisted output is a hole in reproducibility --
# under a role that says only that we did not classify it.
FIXED_ROLES = {ENCODER: "encoder", METRICS: "metrics"}
DEFAULT_ROLE = "extra"

__all__ = ["AdapterError", "Context", "fingerprint", "run"]


class AdapterError(Exception):
    """The port is wrong, as opposed to the run having failed."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def installed_packages() -> dict:
    """Every distribution visible to this interpreter, and its version.

    **This is what makes a difference between two runs explainable.** Bitwise
    agreement across different hardware is not achievable for floating-point
    work, so the guarantee is narrower and more useful: the same environment
    reproduces, and when two runs disagree the manifests show why. Before
    this, `env` held python and hostname alone -- nothing said which torch had
    produced a result.
    """
    out = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            out[name] = dist.version or ""
    return dict(sorted(out.items()))


def fingerprint(packages: dict) -> str:
    """One value for a whole environment, so two runs compare at a glance.

    Sorted before hashing: two identical environments must not look different
    because a directory happened to be walked in another order.
    """
    body = "\n".join(f"{k}=={v}" for k, v in sorted(packages.items()))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _is_number(v: Any) -> bool:
    """Reject bool: in Python it subclasses int and would slip through."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class Context:
    """What the body is given. Deliberately small."""

    def __init__(self, out: Path, config: dict) -> None:
        self.out = out
        self.config = config

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Write `metrics.json` in the shape the contract fixes.

        Values are checked here rather than left to `contract-test`, which can
        only object once the run has already spent its GPU hours.
        """
        for k, v in metrics.items():
            if not _is_number(v):
                raise AdapterError(
                    f"metric {k!r} is {v!r}, which is not a number; "
                    "nothing can compare it")
        (self.out / METRICS).write_text(
            json.dumps({"schema_version": SCHEMA_VERSION,
                        "metrics": dict(metrics)},
                       sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")


def _read_config(config: Path) -> tuple[dict, str]:
    try:
        raw = Path(config).read_bytes()
    except OSError as exc:
        raise AdapterError(f"cannot read the config {config}: {exc}") from None
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise AdapterError(f"{config} cannot be parsed: {exc}") from None
    if not isinstance(cfg, dict):
        raise AdapterError(f"{config}: the top level is not a mapping")
    if "seed" not in cfg:
        raise AdapterError(
            f"{config} has no seed. A run whose seed is not recorded cannot "
            "be reproduced, so it is refused rather than run")
    return cfg, hashlib.sha256(raw).hexdigest()


def _world_size(env: Mapping[str, str]) -> int:
    raw = env.get("WORLD_SIZE", "1")
    try:
        return int(raw)
    except ValueError:
        raise AdapterError(
            f"WORLD_SIZE is {raw!r}, which is not a number. The launcher sets "
            "it; guessing a value would change how the run is recorded"
        ) from None


def _collect(out: Path) -> list[dict]:
    """Every file under `--out`, except the manifest, in a stable order.

    The manifest cannot contain its own hash, which is the one exception.
    Everything else is listed: this is the moment, and the only moment, at
    which all of the run's outputs are visible at once.
    """
    found = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = p.relative_to(out).as_posix()     # the same on every platform
        if rel == MANIFEST:
            continue
        found.append({"path": rel,
                      "role": FIXED_ROLES.get(rel, DEFAULT_ROLE),
                      "sha256": _sha256(p),
                      "bytes": p.stat().st_size})
    return sorted(found, key=lambda a: a["path"])


def _check_encoder(out: Path, reason: str | None) -> None:
    present = (out / ENCODER).is_file()
    if present and reason:
        raise AdapterError(
            f"{ENCODER} was produced, but encoder_absent_reason says "
            f"{reason!r}. The two statements contradict each other")
    if not present and not reason:
        raise AdapterError(
            f"{ENCODER} was not produced and no reason was given. A method "
            "that cannot produce one is allowed to say so (CONTRACT section "
            "3); not producing one quietly is not")


def _environment() -> dict:
    packages = installed_packages()
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        # Different instruction sets reorder floating-point work, so the
        # machine is part of what identifies a result.
        "system": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "packages_sha256": fingerprint(packages),
    }


def run(*, config: Path, out: Path, method: str, stage: str,
        body: Callable[[Context], None],
        upstream: dict | None = None,
        encoder_absent_reason: str | None = None,
        env: Mapping[str, str] | None = None) -> int:
    """Run `body` and record what happened. Returns the exit status to use."""
    env = os.environ if env is None else env
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg, config_sha256 = _read_config(config)
    world_size = _world_size(env)

    started_at = _now()
    error: str | None = None
    try:
        body(Context(out, cfg))
    except AdapterError:
        # Misuse of this module is not a run result, so it is not recorded as
        # one. A metric that is not a number means the port is wrong, and
        # burying it in a `failed` manifest would hide which one it was.
        raise
    except Exception:                       # the run failed; that is a result
        error = traceback.format_exc(limit=8).strip()
    finished_at = _now()

    # Only demanded of a run that claims to have succeeded: a run that died
    # has already explained itself.
    if error is None:
        _check_encoder(out, encoder_absent_reason)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "stage": stage,
        "status": "ok" if error is None else "failed",
        "config_sha256": config_sha256,
        "started_at": started_at,
        "finished_at": finished_at,
        "seed": cfg["seed"],
        "world_size": world_size,
        "env": _environment(),
        "upstream": upstream,
        "artifacts": _collect(out),
    }
    if encoder_absent_reason:
        manifest["encoder_absent_reason"] = encoder_absent_reason
    if error is not None:
        manifest["error"] = error

    (out / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n", encoding="utf-8")
    return 0 if error is None else 1
