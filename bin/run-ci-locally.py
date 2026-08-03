#!/usr/bin/env python3
"""Run the CI workflow here, by reading it rather than repeating it.

    run-ci-locally.py [--event push|pull_request] [--job NAME] [--dry-run]
                      [--platform linux/amd64] [--json report.json]

CI stopped running when the account hit a billing wall, and work has to
continue. Docker is here, so the workflow's own steps can be executed on this
machine. Measured before any of this was written: the container job's three
steps, run by hand against a clean export of HEAD, gave 589 tests OK on
`linux/amd64` -- the architecture the runner uses.

**The risk is not failing. It is passing while meaning something else.** A
shell script that mirrors the workflow is a second implementation of it, and
"the same rule implemented twice" is the most repeated root cause in this
repository. Two copies agree until the day one is edited, and then "CI passed
locally" is false with nothing able to notice, because the real CI is down.

So this contains none of the workflow's commands. It parses
`.github/workflows/tests.yml`, resolves what the workflow itself computes --
including the matrix, by running the step that produces it -- and executes
what it finds. `tests/test_run_ci_locally.py` asserts that none of the
workflow's shell appears in this file, and that no job is named in its code.

**What it cannot reproduce, it says.** A local runner that skips quietly turns
"we did not check" into "we checked", which is worse than not having one. Every
`uses:` step, every difference from the runner image, and every unresolved
expression is either reported or refused.

Known differences, reported at the end of every run:

  - the host image is not GitHub's runner image. Steps that are not driving
    docker run in a python image matching `.python-version`, with git added,
    because the suite has tests that need it
  - `linux/amd64` here is emulated. Results have matched, but it is not the
    same silicon, and this project already says agreement across different
    hardware is not guaranteed
  - `uses:` steps are actions, not shell. Their intent is noted, not executed
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
EVENTS = ("push", "pull_request")

# The exit codes this tool speaks in, named so a check can reason about them
# rather than about bare numbers. The distinction that matters is between a
# verdict and the absence of one: a step that ran and failed is evidence that
# something is wrong (EXIT_STEP_FAILED); a run that could not be set up -- the
# runner image would not build, nothing ran at all -- is evidence of nothing
# either way (EXIT_NO_VERDICT), and a check that cannot tell the two apart will
# read a network outage as the property under test having failed.
EXIT_OK = 0
EXIT_STEP_FAILED = 1
EXIT_NO_VERDICT = 2

# A step that drives containers has to run where the daemon is, exactly as it
# does on the runner. Everything else has to run on Linux, because that is the
# whole reason this workflow exists: the machine it is developed on is not the
# machine it is deployed to. Decided by what the command does rather than by
# which job holds it, because a list of jobs goes stale the moment one is
# added.
CONTAINER_DRIVERS = ("docker", "podman")

EXPRESSION = re.compile(r"\$\{\{([^}]*)\}\}")

# The only condition this workflow uses. Anything else is refused rather than
# guessed: guessing runs a job CI would skip, or skips one CI would run, and
# either way the local result stops meaning what it claims.
EVENT_CONDITION = re.compile(
    r"^\s*github\.event_name\s*==\s*'([a-z_]+)'\s*$")


class CannotReproduce(Exception):
    """Refused. Never reported as a pass, and never silently skipped."""


@dataclass
class Job:
    name: str
    spec: dict
    applicable: bool
    reason: str = ""


@dataclass
class Report:
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def record(self, job: str, step: str, code: int, where: str) -> None:
        self.steps.append({"job": job, "step": step, "exit": code,
                           "ran_in": where})

    def note(self, what: str, why: str) -> None:
        self.notes.append({"what": what, "why": why})

    def failures(self) -> list:
        return [s for s in self.steps if s["exit"] != 0]

    def exit_code(self) -> int:
        """Zero only when something ran and nothing failed.

        Nothing having run is not a pass. Zero failures out of zero steps is
        exactly what a broken runner reports, and it is the most flattering
        possible lie.
        """
        if not self.steps:
            return EXIT_NO_VERDICT
        return EXIT_STEP_FAILED if self.failures() else EXIT_OK

    def as_dict(self) -> dict:
        return {"schema_version": 1,
                "counts": {"steps": len(self.steps),
                           "failed": len(self.failures()),
                           "not_reproduced": len(self.notes)},
                "steps": self.steps, "not_reproduced": self.notes}


def load_workflow(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise CannotReproduce(
            "PyYAML is not installed, so the workflow cannot be read. It is "
            "in requirements-tools.lock.txt") from None
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def applicable(spec: dict, event: str) -> tuple:
    """Whether a job runs for `event`, and why not when it does not."""
    cond = spec.get("if")
    if cond is None:
        return True, ""
    m = EVENT_CONDITION.match(str(cond))
    if not m:
        raise CannotReproduce(
            f"the condition {cond!r} is not one this understands. Guessing "
            "would run a job CI skips, or skip one CI runs")
    if m.group(1) == event:
        return True, ""
    return False, f"the workflow runs it only for {m.group(1)}"


def plan(doc: dict, event: str) -> list:
    """Every job, in an order that respects `needs`, each marked."""
    jobs, done, out = dict(doc["jobs"]), set(), []
    while jobs:
        ready = [n for n, s in jobs.items()
                 if set(_needs(s)) <= done]
        if not ready:
            raise CannotReproduce(
                f"these jobs need each other in a cycle: {sorted(jobs)}")
        for name in sorted(ready):
            spec = jobs.pop(name)
            ok, why = applicable(spec, event)
            out.append(Job(name=name, spec=spec, applicable=ok, reason=why))
            done.add(name)
    return out


def _needs(spec: dict) -> list:
    n = spec.get("needs") or []
    return [n] if isinstance(n, str) else list(n)


def substitute(text: str, values: dict) -> str:
    """Replace `${{ matrix.x }}` with what the plan resolved.

    An expression left in place would be handed to the shell, which would not
    complain -- it would run something that is not what CI runs.
    """
    def one(m):
        key = m.group(1).strip()
        short = key.split(".")[-1]
        if key.startswith("matrix.") and short in values:
            return str(values[short])
        raise CannotReproduce(
            f"nothing here resolves {m.group(0)!r}, and leaving it in the "
            "command would run something other than what CI runs")
    return EXPRESSION.sub(one, text)


def matrix_of(doc: dict, job: str, outputs: dict) -> list:
    """The matrix rows for `job`, taken from what the workflow computes.

    The values come from the output of the job that produces them, executed
    for real. Recomputing them here would be a second implementation of the
    discovery, and this repository has already had a CI matrix and a lock
    check disagree for exactly that reason.
    """
    strategy = (doc["jobs"][job].get("strategy") or {}).get("matrix")
    if not strategy:
        return [{}]
    rows = [{}]
    for key, value in strategy.items():
        resolved = value
        if isinstance(value, str) and EXPRESSION.search(value):
            resolved = _from_outputs(value, outputs)
        rows = [dict(r, **{key: v}) for r in rows for v in resolved]
    return rows


def _from_outputs(expr: str, outputs: dict):
    m = re.search(r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)", expr)
    if not m:
        raise CannotReproduce(f"cannot resolve the matrix expression {expr!r}")
    job, key = m.group(1), m.group(2)
    if job not in outputs or key not in outputs[job]:
        raise CannotReproduce(
            f"the matrix needs {job}.{key}, which has not been produced. Run "
            f"{job} first, or this would run against a matrix nobody computed")
    return json.loads(outputs[job][key])


def where(command: str) -> str:
    """Where a step has to run for it to mean the same thing.

    Returns `"host"` or `"image"`. Not `"container"`: the workflow has a job
    of that name, and a placement value spelled the same makes it impossible
    for any check to tell "this step runs in a container" from "this code
    branches on the container job".
    """
    first = command.strip().split()
    head = first[0] if first else ""
    if head in CONTAINER_DRIVERS or any(
            f"\n{d} " in f"\n{command}" for d in CONTAINER_DRIVERS):
        return "host"
    return "image"


def describe_uses(step: dict) -> str:
    """What an action step was for, since it cannot be executed here."""
    ref = str(step.get("uses", "")).split("@")[0]
    with_ = step.get("with") or {}
    detail = f" with {', '.join(sorted(with_))}" if with_ else ""
    return (f"`uses: {ref}`{detail} is a GitHub action, not shell. Its effect "
            "is provided differently here (the export of HEAD, and the "
            "interpreter in the image)")


def dirty_files(root: Path) -> list:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]


def submodule_paths(root: Path) -> list:
    """The working-tree paths of the pinned submodules, from git itself.

    Asked of git rather than parsed by hand so there is one answer; `git
    archive` of the superproject does not descend into them, so they are
    materialised separately below.
    """
    r = subprocess.run(["git", "config", "--file", ".gitmodules",
                        "--get-regexp", r"\.path$"],
                       cwd=root, capture_output=True, text=True)
    if r.returncode != 0:                       # no .gitmodules: no submodules
        return []
    return [line.split(None, 1)[1] for line in r.stdout.splitlines()
            if line.strip()]


def export_head(root: Path, dest: Path) -> None:
    """A clean tree at HEAD, which is what the runner checks out.

    Testing the working tree instead would make a green run a statement about
    code that is not in the repository. Submodules are materialised too: `git
    archive` stops at the submodule boundary, so without this the exported tree
    would have an empty `third_party/<sub>` and the run would test a tree with a
    hole where the pinned upstream should be -- exactly what CI checks out with
    `submodules: recursive`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"],
                             cwd=root, capture_output=True)
    if archive.returncode != 0:
        raise CannotReproduce("git archive failed; is this a checkout?")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)],
                         input=archive.stdout, capture_output=True)
    if tar.returncode != 0:
        raise CannotReproduce(f"could not unpack the export: {tar.stderr}")

    for sub in submodule_paths(root):
        src = root / sub
        if not (src / ".git").exists():
            raise CannotReproduce(
                f"submodule {sub} is not checked out, so the export would have "
                f"a hole there. Run: git submodule update --init {sub}")
        arc = subprocess.run(["git", "archive", "--format=tar", "HEAD"],
                             cwd=src, capture_output=True)
        if arc.returncode != 0:
            raise CannotReproduce(f"git archive failed for submodule {sub}")
        (dest / sub).mkdir(parents=True, exist_ok=True)
        t = subprocess.run(["tar", "-x", "-C", str(dest / sub)],
                           input=arc.stdout, capture_output=True)
        if t.returncode != 0:
            raise CannotReproduce(
                f"could not unpack submodule {sub}: {t.stderr}")


def image_for(tree: Path, platform: str) -> str:
    """A Linux image whose interpreter matches what the workflow pins.

    Not the runner's image, and the report says so. `git` is added because the
    suite has tests that need it and the runner has it; without it they would
    skip here and not there, which is a difference that hides.
    """
    version = (tree / ".python-version").read_text(encoding="utf-8").strip()
    tag = f"python:{version}-slim-bookworm"
    # The platform is in the name. Built for the host and then run as
    # `--platform linux/amd64`, docker refuses with exit 125 -- which arrives
    # as a bare number and looks like a failing test.
    name = f"ci-local-{version}-{platform.replace('/', '-')}"
    exists = subprocess.run(["docker", "image", "inspect", name],
                            capture_output=True)
    if exists.returncode == 0:
        return name
    df = f"FROM {tag}\nRUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*\n"
    build = subprocess.run(
        ["docker", "build", "-q", "--platform", platform, "-t", name, "-"],
        input=df.encode(), capture_output=True)
    if build.returncode != 0:
        raise CannotReproduce(
            f"could not build the local runner image: {build.stderr.decode()}")
    return name


def run_step(command: str, tree: Path, image: str, platform: str,
             place: str) -> int:
    if place == "host":
        r = subprocess.run(["bash", "-eo", "pipefail", "-c", command],
                           cwd=tree)
        return r.returncode
    r = subprocess.run(
        ["docker", "run", "--rm", "--platform", platform,
         "-v", f"{tree}:/work", "-w", "/work", image,
         "bash", "-eo", "pipefail", "-c", command])
    return r.returncode


def execute(doc: dict, event: str, only: str | None, platform: str,
            dry_run: bool, root: Path) -> Report:
    rep = Report()
    for f in dirty_files(root):
        rep.note(f"uncommitted: {f}",
                 "CI tests the pushed commit; this file is not in the export")

    trees: list = []
    tree = Path(tempfile.mkdtemp(prefix="ci-local-"))
    trees.append(tree)
    image = None
    outputs: dict = {}
    try:
        export_head(root, tree)
        for job in plan(doc, event):
            print(f"\n=== {job.name}")
            if not job.applicable:
                rep.note(f"job {job.name} not run", job.reason)
                print(f"  -- {job.reason}")
                continue

            # A job that publishes outputs is run first and in full, whatever
            # `--job` asked for: the matrix of everything after it is built
            # from what it produces, and a matrix nobody computed is a run
            # against nothing.
            if job.spec.get("outputs"):
                values, ran = _collect_outputs(job, tree)
                outputs[job.name] = values
                for name, code in ran:
                    rep.record(job.name, name, code, "host")
                    print(f"  -> [host] {name}")
                rep.note(f"{job.name}: its steps ran on this machine",
                         "the matrix has to be known before any image is "
                         "built, and the step run is the workflow's own")
                continue

            if only and job.name != only:
                print("  -- not selected")
                continue

            for row in matrix_of(doc, job.name, outputs):
                if row:
                    print(f"  # {', '.join(f'{k}={v}' for k, v in row.items())}")
                # **A fresh export per matrix row.** The runner gives every
                # matrix job its own checkout; reusing one tree let the venv
                # built for one method sit in the next method's run, where the
                # guards that scan the repository then read it. It showed up as
                # two methods reporting different numbers of tests from the
                # same suite -- found by running this, not by reasoning.
                tree = Path(tempfile.mkdtemp(prefix="ci-local-"))
                trees.append(tree)
                export_head(root, tree)
                for step in job.spec.get("steps", []):
                    if "uses" in step:
                        note = describe_uses(step)
                        rep.note(f"{job.name}: step not executed", note)
                        print(f"  -- not reproduced: {note}")
                        continue
                    command = substitute(str(step["run"]), row)
                    place = where(command)
                    name = step.get("name",
                                    command.strip().splitlines()[0])
                    print(f"  -> [{place}] {name}")
                    if dry_run:
                        continue
                    if place == "image" and image is None:
                        image = image_for(tree, platform)
                    code = run_step(command, tree, image, platform, place)
                    rep.record(f"{job.name}{row or ''}", name, code, place)
                    if code != 0:
                        print(f"  *** failed with {code}")
                        break
    finally:
        for t in trees:
            shutil.rmtree(t, ignore_errors=True)
    return rep


def _collect_outputs(job: Job, tree: Path) -> tuple:
    """Run the steps that publish a job's outputs, and read what they wrote.

    The workflow's steps append to `$GITHUB_OUTPUT`; the same commands run
    here with that variable pointing at a real file, so the values come from
    the workflow's own code rather than from a copy of it.

    These run on this machine rather than in an image, and the report says so.
    The matrix has to be known before there is an image to run anything in.
    """
    import os
    values, ran = {}, []
    for step in job.spec.get("steps", []):
        if "run" not in step:
            continue
        with tempfile.NamedTemporaryFile("w+", delete=False) as fh:
            out_path = fh.name
        env = {**os.environ, "GITHUB_OUTPUT": out_path}
        r = subprocess.run(["bash", "-eo", "pipefail", "-c", str(step["run"])],
                           cwd=tree, env=env, capture_output=True, text=True)
        ran.append((step.get("name", "step"), r.returncode))
        if r.returncode != 0:
            raise CannotReproduce(
                f"{job.name} could not produce its outputs: {r.stderr[-500:]}")
        for line in Path(out_path).read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                values[k] = v
        Path(out_path).unlink(missing_ok=True)
    return values, ran


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    ap.add_argument("--event", default="pull_request", choices=EVENTS)
    ap.add_argument("--job", default=None)
    ap.add_argument("--platform", default="linux/amd64",
                    help="the architecture the runner uses")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    try:
        doc = load_workflow(a.workflow)
        rep = execute(doc, a.event, a.job, a.platform, a.dry_run, ROOT)
    except CannotReproduce as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return EXIT_NO_VERDICT
    if a.json:
        a.json.write_text(json.dumps(rep.as_dict(), indent=2) + "\n",
                          encoding="utf-8")
    if a.dry_run:
        print("\n  dry run: nothing was executed")
        return EXIT_OK
    print("\n=== what could not be reproduced here")
    for n in rep.notes:
        print(f"  - {n['what']}: {n['why']}")
    print(f"  - the image is not GitHub's runner image, and {a.platform} "
          "is emulated on this machine")
    c = rep.as_dict()["counts"]
    print(f"\n  {c['steps'] - c['failed']}/{c['steps']} steps passed, "
          f"{c['not_reproduced']} things not reproduced")
    return rep.exit_code()


if __name__ == "__main__":
    sys.exit(main())
