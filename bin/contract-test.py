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
  6. `metrics.json` parses and every value is a number
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

    if not (out / ENCODER).is_file():
        bad("encoder-missing", f"{ENCODER} is absent")
    elif roles.get(ENCODER) != "encoder":
        bad("encoder-role",
            f"{ENCODER} is not registered under the role 'encoder'")

    mp = out / METRICS
    if not mp.is_file():
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
            for k, val in metrics.items():
                if not _is_number(val):
                    bad("metrics-not-numeric",
                        f"{k} is not a number: {val!r}; nothing can compare it")

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
