#!/usr/bin/env python3
"""Decide whether an adapter's output satisfies the contract.

**This is how "the port is finished" gets decided by a machine rather than by
opinion.** The contract itself is defined in the Capture repository, in
`docs/CONTRACT.md`, which is the single source of truth.

    contract-test.py --out <dir> --config <resolved.json> [--exit-status N]

What is checked (CONTRACT.md section 5):

  1. the required files are present
  2. `run_manifest.json` parses and carries every required field
  3. `config_sha256` matches the config that was actually handed in.
     The config is the canonical JSON produced by `resolve-config.py`;
     the hash is taken over its bytes, so this tool needs no parser
  4. every listed artifact exists, with matching `sha256` and `bytes`
  5. `encoder.pt` is registered under the role `encoder`
  6. `metrics.json` parses, every value in both of its blocks is a number,
     and every name in `metrics` is one the contract's vocabulary defines
  7. **no file in `--out` is missing from the manifest**
  8. `finished_at >= started_at`

Item 7 mirrors the capture index used on the Capture side: what was written
and what remains are reconciled against each other. An unlisted output is an
artifact nobody knows about, which is a hole in reproducibility.

**Success requires two signals to agree:** exit status 0 *and*
`status: "ok"`. Neither is trusted alone. On the Capture side a gate once
returned exit 0 while reporting detected secrets, and they went straight
through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# The vocabulary is read from the one place that defines it, never copied.
# Two copies of a rule agree everywhere except the case that matters; that is
# how the container jobs broke (DESIGN 5.38). The writing side enforcing it
# alone would make it a convention: a run arriving from an older adapter or
# written by hand has to be caught here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapterlib import METRIC_VOCABULARY, METRICS_SCHEMA_VERSION  # noqa: E402

MANIFEST = "run_manifest.json"
ENCODER = "encoder.pt"
METRICS = "metrics.json"

REQUIRED_FIELDS = ("schema_version", "method", "stage", "status",
                   "config_sha256", "started_at", "finished_at",
                   "seed", "env", "artifacts")
STATUSES = ("ok", "failed")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_number(v) -> bool:
    """Reject bool. In Python bool subclasses int, so it would slip through."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(out: Path, config: Path, exit_status: int | None = None
          ) -> tuple[int, dict]:
    out = Path(out)
    v: list[dict] = []

    def bad(kind: str, detail: str) -> None:
        v.append({"kind": kind, "detail": detail})

    man_path = out / MANIFEST
    if not man_path.is_file():
        bad("manifest-missing",
            f"{MANIFEST} is absent; the run may have died part-way")
        return 1, {"schema_version": 1, "status": None, "violations": v}
    try:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if not isinstance(man, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as exc:
        bad("manifest-unparsable", f"{MANIFEST} cannot be parsed: {exc}")
        return 1, {"schema_version": 1, "status": None, "violations": v}

    for f in REQUIRED_FIELDS:
        if f not in man:
            bad("manifest-field", f"required field is missing: {f}")

    status = man.get("status")
    if status is not None and status not in STATUSES:
        bad("status-unknown",
            f"status is {status!r}; allowed values are {', '.join(STATUSES)}")

    # Success requires both signals to agree.
    if exit_status is not None and status in STATUSES:
        agree = (exit_status == 0) == (status == "ok")
        if not agree:
            bad("status-disagreement",
                f"exit status {exit_status} disagrees with status {status!r}; "
                "one of the two is lying")

    s, f_ = man.get("started_at"), man.get("finished_at")
    if isinstance(s, str) and isinstance(f_, str) and f_ < s:
        bad("time-order", f"finished_at ({f_}) precedes started_at ({s})")

    try:
        want = sha256_of(Path(config))
    except OSError as exc:
        want = None
        bad("config-unreadable", f"cannot read the config: {exc}")
    if want and man.get("config_sha256") != want:
        bad("config-mismatch",
            "config_sha256 does not match the config supplied; "
            "this is not the configuration that ran")

    arts = man.get("artifacts")
    listed: set[str] = set()
    roles: dict[str, str] = {}
    if not isinstance(arts, list):
        if "artifacts" in man:
            bad("manifest-field", "artifacts is not an array")
        arts = []
    for a in arts:
        if not isinstance(a, dict) or "path" not in a:
            bad("artifact-shape", f"malformed entry in artifacts: {a!r}")
            continue
        rel = a["path"]
        listed.add(rel)
        roles[rel] = a.get("role", "")
        p = out / rel
        if not p.is_file():
            bad("artifact-missing", f"listed but absent: {rel}")
            continue
        if a.get("sha256") != sha256_of(p):
            bad("artifact-sha256",
                f"content differs from what was recorded: {rel}")
        if a.get("bytes") != p.stat().st_size:
            bad("artifact-bytes",
                f"size differs from what was recorded: {rel}")

    # A run that reported failure has no outputs to show, because that is
    # what failing means (CONTRACT section 4). Its integrity is still checked
    # below -- whatever it did write must be described correctly -- but the
    # required-file checks belong to a run that claims to have succeeded.
    failed = status == "failed"
    reason = man.get("encoder_absent_reason")
    has_reason = isinstance(reason, str) and reason.strip() != ""

    if not (out / ENCODER).is_file():
        # CONTRACT section 3: a method that cannot produce one may say so.
        # Not producing one quietly is what is forbidden.
        if not failed and not has_reason:
            bad("encoder-missing",
                f"{ENCODER} is absent and no encoder_absent_reason is "
                "recorded; a method that produces none may say so, but not "
                "silently")
    elif has_reason:
        bad("encoder-contradiction",
            f"{ENCODER} exists, yet encoder_absent_reason says {reason!r}; "
            "the two statements contradict each other")
    elif roles.get(ENCODER) != "encoder":
        bad("encoder-role",
            f"{ENCODER} is not registered under the role 'encoder'")

    mp = out / METRICS
    if not mp.is_file():
        if not failed:
            bad("metrics-missing", f"{METRICS} is absent")
    else:
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
            metrics = m["metrics"]
            if not isinstance(metrics, dict):
                raise ValueError("metrics is not an object")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bad("metrics-unparsable", f"{METRICS} cannot be parsed: {exc}")
        else:
            version = m.get("schema_version")
            if version != METRICS_SCHEMA_VERSION:
                bad("metrics-schema",
                    f"schema_version is {version!r}; the contract fixes "
                    f"{METRICS} at {METRICS_SCHEMA_VERSION}")

            # The original's own names for its own numbers. Required: losing
            # them loses what the method called its results, with nothing in
            # the output to say so.
            raw = m.get("metrics_raw")
            if not isinstance(raw, dict):
                bad("metrics-raw-missing",
                    "metrics_raw is absent or is not an object; the names "
                    "the original gave its numbers have to survive")
            else:
                for k, val in raw.items():
                    if not _is_number(val):
                        bad("metrics-not-numeric",
                            f"metrics_raw.{k} is not a number: {val!r}")

            for k, val in metrics.items():
                if not _is_number(val):
                    bad("metrics-not-numeric",
                        f"{k} is not a number: {val!r}; nothing can compare it")
                if k not in METRIC_VOCABULARY:
                    bad("metrics-unknown-name",
                        f"{k} is not a name the contract defines, so nothing "
                        "downstream knows what it may be compared with. "
                        "Known: " + ", ".join(sorted(METRIC_VOCABULARY)))

    # Unlisted files are not allowed. The manifest cannot contain its own
    # hash, so it is the single exception.
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out))
        if rel == MANIFEST or rel in listed:
            continue
        bad("unlisted-file",
            f"output not listed in the manifest: {rel}; "
            "an artifact nobody knows about is a hole in reproducibility")

    ok = not v and status == "ok"
    return (0 if ok else 1), {
        "schema_version": 1,
        "status": status,
        "counts": {"violations": len(v), "artifacts": len(listed)},
        "violations": v,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--exit-status", type=int, default=None,
                    help="the adapter's exit status; checked for agreement "
                         "with the manifest")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rc, rep = check(a.out, a.config, a.exit_status)
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    for x in rep["violations"]:
        print(f"  VIOLATION [{x['kind']}] {x['detail']}")
    if rc == 0:
        print("  ok: the contract is satisfied")
    elif not rep["violations"] and rep.get("status") == "failed":
        print("  reported as a failure, correctly (no contract violations)")
    else:
        print(f"  *** {len(rep['violations'])} contract violation(s) ***")
    return rc


if __name__ == "__main__":
    sys.exit(main())
