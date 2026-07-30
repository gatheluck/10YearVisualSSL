#!/usr/bin/env python3
"""Decide whether an environment is the locked one.

    verify-environment.py --lock <file> [--lock <file> ...] [--manifest <run_manifest.json>]

**Installing from a lock and having installed the lock are different claims.**
Only the second is worth anything, and nothing checked it. The check used to
be a shell one-liner comparing `pip freeze` against a single lock file while
the install used two -- so it reported a difference that was not one, and a
correct environment came back looking wrong. A one-liner people have to
assemble correctly is not a mechanism.

Two questions, one comparison:

  --manifest omitted   is *this* environment the locked one?
  --manifest given     did *that run* use the locked environment?

The second is the one that matters after the fact. `run_manifest.json` records
every installed distribution and version, so a finished result can be checked
against the lock long after the machine is gone.

**Any difference is a failure**, including a package the lock does not
mention. Something installed that the lock does not describe means the
environment cannot be rebuilt from the lock, which is the whole point of
having one. Nothing is reported as a warning and passed over.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path


# `python -m venv` puts this there before any lock is read. Measured: a bare
# venv on CPython 3.12 contains exactly `pip`. `setuptools` is *not* seeded --
# it is a real torch requirement and is compared like anything else.
#
# **Ignoring is not hiding.** Every exemption is reported, with its reason.
SEEDED_BY_VENV = frozenset({"pip"})
SEEDED_REASON = ("seeded by `python -m venv` before any lock is read, and its "
                 "version comes from the interpreter build")


class EnvironmentMismatch(Exception):
    """A refusal, always naming what was refused."""


def canon(name: str) -> str:
    """Distribution names compare case- and separator-insensitively."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_lines(text: str) -> list[str]:
    """Requirement lines, with hash continuations folded back in.

    A hashed lock spreads one requirement over several lines with trailing
    backslashes; read line by line, a hash looks like a requirement.
    """
    out = []
    for line in text.replace("\\\n", " ").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(line)
    return out


def read_locks(paths) -> dict:
    """Merge lock files into one mapping of canonical name to version."""
    merged: dict = {}
    for path in paths:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EnvironmentMismatch(f"cannot read {path}: {exc}") from None
        for line in _requirement_lines(text):
            # Everything before the first option is the requirement. Taking
            # line.split()[0] instead assumed no space before `==`, and a lock
            # that wrapped before its specifier was reported as unpinned.
            head = re.split(r"\s--", line, 1)[0]
            spec = re.sub(r"\s+", "", head)
            if "==" not in spec:
                raise EnvironmentMismatch(
                    f"{path}: {spec} is not pinned; a lock states exactly one "
                    "version")
            name, _, version = spec.partition("==")
            key = canon(name)
            if key in merged and merged[key] != version:
                raise EnvironmentMismatch(
                    f"{key} is {merged[key]} in one lock and {version} in "
                    f"{path}; whichever won would be silent")
            merged[key] = version
    return merged


def read_manifest(path) -> dict:
    """The environment a run recorded for itself."""
    path = Path(path)
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentMismatch(f"cannot read {path}: {exc}") from None
    packages = (man.get("env") or {}).get("packages")
    if not isinstance(packages, dict):
        raise EnvironmentMismatch(
            f"{path} records no env.packages, so the run cannot be checked "
            "against a lock at all. Manifests written before this was added "
            "carry only python and hostname")
    return packages


def installed() -> dict:
    out = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            out[name] = dist.version or ""
    return out


def compare(locked: dict, present: dict) -> tuple[int, dict]:
    want = {canon(k): v for k, v in locked.items()}
    got = {canon(k): v for k, v in present.items()}
    original = {canon(k): k for k, v in present.items()}
    original.update({canon(k): k for k, v in locked.items()})

    # Only exempt when no lock takes responsibility for it. Once one names it,
    # it is checked like anything else.
    exempt = sorted((SEEDED_BY_VENV & set(got)) - set(want))
    ignored = [{"package": original.get(k, k), "version": got[k],
                "reason": SEEDED_REASON} for k in exempt]

    diffs = []
    for key in sorted((set(want) | set(got)) - set(exempt)):
        name = original.get(key, key)
        if key not in got:
            diffs.append({"kind": "missing", "package": name,
                          "locked": want[key], "present": None,
                          "detail": f"{name} {want[key]} is locked but not "
                                    "installed"})
        elif key not in want:
            diffs.append({"kind": "unexpected", "package": name,
                          "locked": None, "present": got[key],
                          "detail": f"{name} {got[key]} is installed but no "
                                    "lock describes it, so this environment "
                                    "cannot be rebuilt from the locks given"})
        elif want[key] != got[key]:
            diffs.append({"kind": "version", "package": name,
                          "locked": want[key], "present": got[key],
                          "detail": f"{name} is {got[key]}, the lock says "
                                    f"{want[key]}"})
    return (0 if not diffs else 1), {
        "schema_version": 1,
        "counts": {"locked": len(want), "present": len(got),
                   "differences": len(diffs), "ignored": len(ignored)},
        "ignored": ignored,
        "differences": diffs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lock", action="append", required=True, type=Path,
                    metavar="FILE",
                    help="a lock file; repeat for every file the install used")
    ap.add_argument("--manifest", type=Path,
                    help="check a finished run instead of this interpreter")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    try:
        locked = read_locks(a.lock)
        present = read_manifest(a.manifest) if a.manifest else installed()
    except EnvironmentMismatch as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
    rc, rep = compare(locked, present)
    rep["source"] = str(a.manifest) if a.manifest else "this interpreter"
    if a.json:
        a.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    for i in rep["ignored"]:
        print(f"  ignored: {i['package']} {i['version']} -- {i['reason']}")
    for d in rep["differences"]:
        print(f"  DIFFERENCE [{d['kind']}] {d['detail']}")
    if rc == 0:
        print(f"  ok: {rep['counts']['locked']} packages, "
              f"{rep['source']} matches the locks exactly")
    else:
        print(f"  *** {len(rep['differences'])} difference(s); "
              f"{rep['source']} is not the locked environment ***")
    return rc


if __name__ == "__main__":
    sys.exit(main())
