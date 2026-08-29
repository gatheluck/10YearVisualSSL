#!/usr/bin/env python3
"""Break the code on purpose and check the tests notice.

    mutate.py --spec mutations.json [--python PATH] [--json report.json]

A test that passes proves nothing on its own; a test that fails when the code
is broken proves something. This runs that check, and it exists as a tool
rather than as a script written fresh each time **because the scripts written
fresh each time lied twice**:

- an anchor that did not match meant the mutation was never applied, and the
  run was reported as `SURVIVED` -- reading as "the tests missed this" when
  nothing had been changed. Here, an absent anchor is a hard error
- a stale `__pycache__` meant one run executed the *previous* mutation's
  bytecode. The report looked plausible and was wrong. Here, bytecode is
  never written and never read

Both of those turn a mutation run into a source of false confidence, which is
worse than not running one.

Each mutation names the tests that should catch it. Running the whole suite
for every mutation is too slow to do honestly, and a mutation checked against
tests that cannot see it teaches nothing.

The spec is JSON:

    {"targets": [
       {"label": "seeding removed",
        "file": "methods/x/train.py",
        "old": "random.seed(args.seed)",
        "new": "pass",
        "tests": ["tests.test_method_x.TestReproducibility"]}
     ]}

Exit status is 0 only when every mutation was killed. A surviving mutant is
either a missing test or an equivalent mutant, and telling those apart is the
reader's job -- this refuses to guess.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PATTERNS = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv",
                                   "runs", ".pytest_cache", ".mypy_cache")
# A Python environment is not our source, and copying one -- a per-method GPU
# venv is measured at 4.6 GB -- made every mutation copy it. It is excluded by
# its PEP 405 marker rather than its name: a name list is the listing mistake
# this repository keeps making, and it would miss a venv wherever someone put
# one under a different name. tests/_repo_files.py excludes environments the
# same way, by pyvenv.cfg.
VENV_MARKER = "pyvenv.cfg"


def _ignore(directory, names):
    """copytree's `ignore`: the old name patterns, plus any directory that
    declares itself a Python environment."""
    skip = set(_PATTERNS(directory, names))
    for name in names:
        if (Path(directory) / name / VENV_MARKER).is_file():
            skip.add(name)
    return skip


def _copy_tree(src, dst) -> None:
    """Copy the tree for a mutation run, carrying symlinks across **as
    symlinks**.

    A vendored submodule can hold a dangling symlink -- fairseq's kaldi example
    under `third_party/unilm` points `st/utils` and `st/steps` at kaldi targets
    that are not checked out. copytree follows symlinks by default, so it tries
    to read the missing target, and the whole copy then dies with an OSError
    whose text is printed as if it were a mutation result. Copying symlinks as
    symlinks (and never following a dangling one) keeps the copy faithful and
    silent."""
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore,
                    symlinks=True, ignore_dangling_symlinks=True)


class MutationError(Exception):
    """A refusal. Never reported as a test result."""


def _run_tests(work: Path, tests: list[str], python: str) -> tuple[int, list]:
    """Run the named tests in `work`, with bytecode reuse made impossible."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([python, "-B", "-m", "unittest", *tests, "-q"],
                       cwd=work, capture_output=True, text=True, env=env)
    failed = sorted(set(re.findall(r"(?:FAIL|ERROR): (\S+)", r.stderr)))
    return r.returncode, failed


def target_in(work: Path, spec: dict) -> Path:
    """The file to mutate, refusing anything outside the work tree.

    Mutating the real tree and restoring it afterwards leaves no trace once
    the run finishes, so a check of the final state cannot see it -- but the
    repository is broken while the run is in progress, and a crash or a
    concurrent read leaves it broken for good. The guard is here rather than
    in a test because a test could only observe the aftermath.
    """
    path = (work / spec["file"]).resolve()
    if not path.is_relative_to(Path(work).resolve()):
        raise MutationError(
            f"{spec['label']}: {spec['file']} resolves outside the work tree. "
            "Mutations are applied to a copy, never to the repository")
    return path


def apply_one(work: Path, spec: dict) -> None:
    """Apply a mutation, or refuse. **Never silently no-op.**"""
    path = target_in(work, spec)
    if not path.is_file():
        raise MutationError(f"{spec['label']}: no such file {spec['file']}")
    text = path.read_text(encoding="utf-8")
    count = text.count(spec["old"])
    if count == 0:
        raise MutationError(
            f"{spec['label']}: the anchor is not in {spec['file']}. "
            "Reporting this as a surviving mutant would be a lie: nothing "
            "was changed")
    if count > 1 and not spec.get("all"):
        raise MutationError(
            f"{spec['label']}: the anchor appears {count} times in "
            f"{spec['file']}, so which one is mutated is a coin toss. Make it "
            'unique, or set "all": true deliberately')
    path.write_text(text.replace(spec["old"], spec["new"],
                                 -1 if spec.get("all") else 1),
                    encoding="utf-8")


def check_baseline(work: Path, tests: list[str], python: str) -> None:
    rc, failed = _run_tests(work, tests, python)
    if rc != 0:
        raise MutationError(
            "the tests do not pass before anything is mutated, so nothing "
            f"below would mean anything. Failing: {', '.join(failed) or '?'}")


def run(spec: dict, python: str = sys.executable) -> tuple[int, dict]:
    targets = spec.get("targets") or []
    if not targets:
        raise MutationError("the spec names no mutations")

    results = []
    work = Path(tempfile.mkdtemp(prefix="mutate-"))
    try:
        _copy_tree(ROOT, work)
        every = sorted({t for m in targets for t in m["tests"]})
        check_baseline(work, every, python)

        for m in targets:
            path = target_in(work, m)
            original = path.read_text(encoding="utf-8")
            apply_one(work, m)
            rc, failed = _run_tests(work, m["tests"], python)
            path.write_text(original, encoding="utf-8")
            results.append({"label": m["label"], "file": m["file"],
                            "killed": rc != 0, "killed_by": failed})
    finally:
        shutil.rmtree(work, ignore_errors=True)

    survived = [r for r in results if not r["killed"]]
    return (0 if not survived else 1), {
        "schema_version": 1,
        "counts": {"mutations": len(results), "killed": len(results) - len(survived),
                   "survived": len(survived)},
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter to run the tests with")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    try:
        spec = json.loads(a.spec.read_text(encoding="utf-8"))
        rc, report = run(spec, a.python)
    except (OSError, ValueError, MutationError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(report, indent=2) + "\n",
                          encoding="utf-8")
    for r in report["results"]:
        mark = ("killed by " + " ".join(r["killed_by"][:2])) if r["killed"] \
            else "SURVIVED -- a missing test, or an equivalent mutant"
        print(f"  {r['label']:<44} {mark}")
    c = report["counts"]
    print(f"  {c['killed']}/{c['mutations']} killed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
