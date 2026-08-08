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
        ctx.write_metrics({"acc": 42.5},
                          names={"acc": "final_pretext_top1_accuracy"})

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

# `metrics.json` changed shape when the vocabulary below arrived, so it
# carries its own version. The manifest did not change and keeps 1.
METRICS_SCHEMA_VERSION = 2

# Whether a number means the same thing in every method.
COMPARABLE = "comparable"
PER_METHOD = "per-method"

# **The vocabulary `metrics.json` may use, and what each name may be compared
# with.** Three ported stages had produced three spellings of the same kinds
# of number; CONTRACT section 7 deferred the choice until two pilots were
# through, and they now are.
#
# The split that matters is `pretext` against `linear_probe`. A method's
# pretext accuracy is the accuracy of *its own* task -- eight-way patch
# position in the first port, a reconstruction objective in the second -- and
# those share no scale. The linear-probe numbers are downstream classification
# against real labels, and they are what this project exists to compare.
# Folding the two into one name would let a machine build a comparison out of
# numbers that cannot be compared, and the result would look right.
#
# Comparability is recorded here rather than in a sentence in the contract
# because a sentence in a document does not hold; this project has the counts
# to prove it. Accuracies are percentages, 0 to 100, which was measured from
# both sources rather than assumed. Losses are per-sample means.
METRIC_VOCABULARY = {
    "final_pretext_loss": PER_METHOD,
    "final_pretext_top1_accuracy": PER_METHOD,
    "best_linear_probe_top1_accuracy": COMPARABLE,
    "final_linear_probe_top1_accuracy": COMPARABLE,
    "best_linear_probe_top5_accuracy": COMPARABLE,
    "final_linear_probe_top5_accuracy": COMPARABLE,
    "epochs_completed": PER_METHOD,
    "steps_completed": PER_METHOD,
    "metrics_unavailable": PER_METHOD,
}

# Which family of names each stage may use.
#
# **The vocabulary alone does not stop the mapping that matters.** A port that
# sends its pretext accuracy to `final_linear_probe_top1_accuracy` passes every
# other check: the name is defined and the value is a number. That one line
# puts a method's own eight-way task in the same column as real classification
# accuracy, and the column looks right.
#
# No machine can read a number and tell which task produced it. It can read
# the stage, and the contract already separates them. So the stage decides
# which family is reachable, and crossing over takes more than a word.
#
# Every name here is written by some port; a name nothing writes has never
# been checked against a real method and reads as settled while nothing has
# produced it. `best_pretext_top1_accuracy` was in this table for exactly one
# commit before it was noticed and removed. Add a name with the mapping that
# uses it.
#
# Names belonging to neither family -- counters, and the unavailable count --
# are available everywhere. An unrecognised stage is refused rather than
# defaulted: defaulting would settle the question by accident.
PRETEXT = "pretext"
LINEAR_PROBE = "linear_probe"
# knowledge_transfer: cluster a frozen encoder's features into pseudo-labels and
# train a fresh network to predict them (Noroozi et al. 2018, "Boosting SSL via
# Knowledge Transfer"). It trains a method's own objective, so its loss is a
# pretext number, never a comparable probe -- the stage sits in the pretext
# family. It is a general SSL stage, named for what it does, not for any method.
CONTRACT_STAGES = ("step1", "knowledge_transfer", "linear_eval")
STAGE_FAMILIES = {
    "step1": PRETEXT,
    "knowledge_transfer": PRETEXT,
    "linear_eval": LINEAR_PROBE,
}


def _family(name: str) -> "str | None":
    if PRETEXT in name:
        return PRETEXT
    if LINEAR_PROBE in name:
        return LINEAR_PROBE
    return None
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

    def __init__(self, out: Path, config: dict, stage: str = "") -> None:
        self.out = out
        self.config = config
        self.stage = stage

    def write_metrics(self, raw: Mapping[str, Any],
                      names: Mapping[str, "str | None"]) -> None:
        """Write `metrics.json`: the contract's names, and the original's.

        `raw` is what the original produced, under the original's own names.
        `names` is the port's translation table, and it is required -- without
        one a port drifts back to inventing names, which is how three stages
        came to spell top-1 accuracy three ways.

        Every key in `raw` must appear in `names`. A key with no contract slot
        is written `None` there, which keeps it in `metrics_raw` and out of
        `metrics`: nothing is lost and nothing is invented. Dropping it
        silently would lose a number the original produced with nothing to
        say so.

        Values are checked here rather than left to `contract-test`, which can
        only object once the run has already spent its GPU hours.
        """
        contract: dict = {}
        for key, value in raw.items():
            if not _is_number(value):
                raise AdapterError(
                    f"metric {key!r} is {value!r}, which is not a number; "
                    "nothing can compare it")
            if key not in names:
                raise AdapterError(
                    f"metric {key!r} has no entry in this port's translation "
                    "table. Give it a contract name, or None to keep it in "
                    "metrics_raw only -- it will not be dropped in silence")
            target = names[key]
            if target is None:
                continue
            if target not in METRIC_VOCABULARY:
                raise AdapterError(
                    f"{target!r} is not in the contract vocabulary, so "
                    "nothing downstream knows what it means. Known names: "
                    + ", ".join(sorted(METRIC_VOCABULARY)))
            if target in contract:
                raise AdapterError(
                    f"two of the original's metrics both map to {target!r}, "
                    "so one would overwrite the other")
            family = _family(target)
            if family is not None:
                if self.stage not in STAGE_FAMILIES:
                    raise AdapterError(
                        f"stage {self.stage!r} is not one the contract "
                        f"defines ({', '.join(CONTRACT_STAGES)}), so which "
                        f"family of metric names it may use is undecided. "
                        "Add it to the contract rather than guessing here")
                allowed = STAGE_FAMILIES[self.stage]
                if family != allowed:
                    raise AdapterError(
                        f"stage {self.stage!r} produces {allowed} numbers, "
                        f"but {target!r} is a {family} name. These measure "
                        "different tasks and cannot share a column; mapping "
                        "one to the other is the mistake the vocabulary "
                        "exists to prevent")
            contract[target] = value
        (self.out / METRICS).write_text(
            json.dumps({"schema_version": METRICS_SCHEMA_VERSION,
                        "metrics": contract,
                        "metrics_raw": dict(raw)},
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
        body(Context(out, cfg, stage))
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
