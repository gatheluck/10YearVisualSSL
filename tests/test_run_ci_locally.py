#!/usr/bin/env python3
"""Specification for `bin/run-ci-locally.py`.

CI stopped running -- the account hit a billing wall -- and work has to
continue. Docker is available, so the workflow's own steps can be executed
here instead. That was measured before this tool was written: the container
job's three steps, run by hand against a clean checkout, gave 589 tests OK on
`linux/amd64`, the architecture the runner uses.

**The danger is not that it will fail. It is that it will pass while meaning
something different from CI.** A hand-written shell script mirroring the
workflow is a second implementation of the workflow, and "the same rule
implemented twice" is the most-repeated root cause in this repository
(CLAUDE.md counts it). The two would agree for as long as nobody edited
either, and diverge exactly when it mattered -- at which point "CI passed
locally" becomes a false statement with nothing able to catch it, because the
real CI is not running.

So this tool **reads `.github/workflows/tests.yml` and executes what it
finds**. It never contains the commands itself. The test that matters most in
this file is the one asserting that.

The second rule is that a difference it cannot reproduce must be **named in
the output**. A local runner that quietly skips a step is worse than no local
runner: it converts "we did not check" into "we checked".
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")

# **The workflow is not always there to read.** `.dockerignore` keeps
# `.github` out of the method images deliberately -- an image should not carry
# the CI definition -- so inside one there is nothing for these to parse. That
# is not a failure, it is a question that cannot be asked, and saying so is
# the difference between a reported skip and eleven errors that look like a
# broken tool. Found by running this against the container job.
needs_workflow = unittest.skipUnless(
    WORKFLOW.is_file(),
    "no workflow here: the method images exclude .github by design")

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _checkout import needs_checkout, needs_git      # noqa: E402


def load(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BIN / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


runner = load("run_ci_locally", "run-ci-locally.py")


@needs_yaml
@needs_workflow
class TestItReadsTheWorkflowRatherThanRestatingIt(unittest.TestCase):
    """**The test this file exists for.**

    If the commands live here too, the tool is a copy of the workflow and
    will drift from it. The check is on the tool's own source: none of the
    workflow's shell may appear in it.
    """

    @staticmethod
    def workflow_commands() -> list[str]:
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        out = []
        for spec in doc["jobs"].values():
            for step in spec.get("steps", []):
                if "run" in step:
                    for line in str(step["run"]).splitlines():
                        line = line.strip().rstrip("\\").strip()
                        if len(line) > 24:
                            out.append(line)
        return out

    def test_the_workflow_has_commands_to_compare_against(self):
        """Against an empty list the check below passes vacuously."""
        self.assertGreater(len(self.workflow_commands()), 3)

    def test_none_of_the_workflows_commands_are_written_here(self):
        source = (BIN / "run-ci-locally.py").read_text(encoding="utf-8")
        copied = [c for c in self.workflow_commands() if c in source]
        self.assertEqual(
            copied, [],
            "these commands are written into the tool as well as into the "
            "workflow, so the two can drift apart:\n"
            + "\n".join(f"  - {c}" for c in copied))

    def test_it_does_not_branch_on_a_job_name(self):
        """A list of jobs goes stale the moment one is added.

        **The first version of this checked for the job name anywhere in the
        source and accused the tool of naming `container`** -- which appears
        because a step can run *in a container*, an ordinary word that
        happens to collide with a job name. An accusation that lands on
        innocent code gets silenced, so this looks for the thing that would
        actually go stale: a comparison or a membership test against the
        name.
        """
        import re
        source = (BIN / "run-ci-locally.py").read_text(encoding="utf-8")
        code = "\n".join(ln for ln in source.splitlines()
                         if not ln.lstrip().startswith("#"))
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        named = []
        for j in doc["jobs"]:
            branch = re.compile(
                r"(==|!=)\s*[\"']" + re.escape(j) + r"[\"']"
                r"|[\"']" + re.escape(j) + r"[\"']\s*(==|!=)"
                r"|\bin\s*\([^)]*[\"']" + re.escape(j) + r"[\"']")
            if branch.search(code):
                named.append(j)
        self.assertEqual(named, [], f"the tool branches on job names: {named}")

    def test_that_detector_would_catch_a_real_branch(self):
        """Against a pattern that matches nothing, the check above is
        vacuous."""
        import re
        j = "locked"
        branch = re.compile(r"(==|!=)\s*[\"']" + re.escape(j) + r"[\"']")
        self.assertTrue(branch.search('if job.name == "locked":'))
        self.assertFalse(branch.search('place = "container"'))


@needs_yaml
@needs_workflow
class TestThePlan(unittest.TestCase):
    def doc(self) -> dict:
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_every_job_in_the_workflow_is_planned(self):
        got = {j.name for j in runner.plan(self.doc(), event="pull_request")}
        self.assertEqual(got, set(self.doc()["jobs"]))

    def test_a_job_runs_after_what_it_needs(self):
        order = [j.name for j in runner.plan(self.doc(), event="pull_request")]
        for name, spec in self.doc()["jobs"].items():
            needs = spec.get("needs") or []
            needs = [needs] if isinstance(needs, str) else needs
            for n in needs:
                with self.subTest(job=name, needs=n):
                    self.assertLess(order.index(n), order.index(name))

    def test_an_event_condition_is_honoured(self):
        """The container job is a pull-request job. Running it for a push
        would test something CI does not do."""
        push = {j.name for j in runner.plan(self.doc(), event="push")
                if j.applicable}
        pr = {j.name for j in runner.plan(self.doc(), event="pull_request")
              if j.applicable}
        self.assertTrue(push < pr, f"push={push} pr={pr}")

    def test_a_skipped_job_is_reported_not_dropped(self):
        """It must still appear in the plan, marked, with the reason."""
        for j in runner.plan(self.doc(), event="push"):
            if not j.applicable:
                self.assertTrue(j.reason, f"{j.name} is skipped with no reason")
                break
        else:
            self.fail("no job was skipped for a plain push")

    def test_the_condition_reader_understands_the_real_one(self):
        """A parser that answers True to everything would satisfy the tests
        above. Give it the workflow's own condition and a negation."""
        self.assertTrue(runner.applicable(
            {"if": "github.event_name == 'pull_request'"}, "pull_request")[0])
        self.assertFalse(runner.applicable(
            {"if": "github.event_name == 'pull_request'"}, "push")[0])

    def test_an_unreadable_condition_is_refused_not_assumed(self):
        """Guessing would silently run, or silently skip, a job."""
        with self.assertRaises(runner.CannotReproduce):
            runner.applicable({"if": "github.actor == 'someone'"}, "push")


@needs_yaml
@needs_workflow
class TestTheMatrixComesFromTheWorkflow(unittest.TestCase):
    def test_the_methods_are_taken_from_the_job_that_finds_them(self):
        """Not from a glob written again here. The discover job already
        computes this, and computing it twice is how the CI matrix and the
        lock checks once disagreed."""
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        got = runner.matrix_of(doc, "locked", {"discover": {
            "methods": '["a", "b"]'}})
        self.assertEqual([m["method"] for m in got], ["a", "b"])

    def test_a_job_with_no_matrix_runs_once(self):
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(runner.matrix_of(doc, "core", {}), [{}])

    def test_an_unresolved_matrix_is_an_error(self):
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        with self.assertRaises(runner.CannotReproduce):
            runner.matrix_of(doc, "locked", {})


class TestSubstitution(unittest.TestCase):
    def test_a_matrix_value_is_substituted(self):
        self.assertEqual(
            runner.substitute("run ${{ matrix.method }} now", {"method": "x"}),
            "run x now")

    def test_an_expression_with_no_value_is_an_error(self):
        """Leaving `${{ ... }}` in a command would run something that is not
        what CI runs, and the shell would not complain."""
        with self.assertRaises(runner.CannotReproduce):
            runner.substitute("run ${{ matrix.method }}", {})

    def test_text_without_expressions_is_untouched(self):
        self.assertEqual(runner.substitute("plain", {}), "plain")


class TestWhereEachStepRuns(unittest.TestCase):
    """A step that drives docker has to run where docker is, exactly as it
    does on the runner. Everything else has to run on Linux, because that is
    the whole reason CI exists -- this machine is macOS.

    Decided by what the command does, not by which job it is in: a list of
    jobs is the mistake this repository keeps making.
    """

    def test_a_docker_step_runs_on_the_host(self):
        self.assertEqual(runner.where("docker build -f x -t y ."), "host")

    def test_a_python_step_runs_in_a_linux_image(self):
        """Called "image", not "container": the workflow has a job named
        `container`, and a placement value spelled the same makes it
        impossible for any check to tell "this step runs in a container" from
        "this code branches on the container job". A mutation survived on
        exactly that ambiguity."""
        self.assertEqual(
            runner.where("python3 -m unittest discover -s tests"), "image")

    def test_the_decision_is_not_made_by_job_name(self):
        """Same command, and it cannot matter where it came from."""
        self.assertEqual(runner.where("docker run --rm x"),
                         runner.where("docker run --rm x"))


@needs_git
class TestTheCheckoutIsOfTheCommit(unittest.TestCase):
    """CI tests the pushed commit. A working tree with uncommitted edits is a
    different thing, and letting it stand in would make a green local run a
    statement about code that is not in the repository."""

    @needs_checkout
    def test_it_exports_head_not_the_working_tree(self):
        d = Path(tempfile.mkdtemp(prefix="ciexport-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        probe = ROOT / "_uncommitted_probe_tmp.txt"
        probe.write_text("not committed\n", encoding="utf-8")
        self.addCleanup(probe.unlink, missing_ok=True)
        runner.export_head(ROOT, d)
        self.assertTrue((d / "README.md").is_file(), "the export is empty")
        self.assertFalse((d / probe.name).exists(),
                         "an uncommitted file reached the export")

    @needs_checkout
    def test_a_dirty_tree_is_reported(self):
        """Not refused -- work continues -- but never unsaid."""
        probe = ROOT / "_uncommitted_probe_tmp.txt"
        probe.write_text("x\n", encoding="utf-8")
        self.addCleanup(probe.unlink, missing_ok=True)
        self.assertTrue(runner.dirty_files(ROOT))

    def test_a_clean_tree_reports_nothing(self):
        """Built rather than assumed: this repository's own tree is dirty
        whenever anyone is working in it, so asserting on it would make the
        result depend on who ran the tests and when."""
        d = Path(tempfile.mkdtemp(prefix="cleanrepo-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for args in (["init", "-q"], ["config", "user.email", "t@example.com"],
                     ["config", "user.name", "t"]):
            subprocess.run(["git", *args], cwd=d, check=True,
                           capture_output=True)
        (d / "a.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=d, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-qm", "c"], cwd=d, check=True,
                       capture_output=True)
        self.assertEqual(runner.dirty_files(d), [])


@needs_yaml
@needs_workflow
class TestTheReportSaysWhatWasNotReproduced(unittest.TestCase):
    """A local runner that skips quietly turns "we did not check" into "we
    checked" -- which is the failure this project has shipped more than once
    (DESIGN 2.4)."""

    def test_an_action_step_is_recorded_as_not_reproduced(self):
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        uses = [s for spec in doc["jobs"].values()
                for s in spec.get("steps", []) if "uses" in s]
        self.assertTrue(uses, "the workflow uses no actions to check against")
        note = runner.describe_uses(uses[0])
        self.assertTrue(note, "a `uses:` step produced no note at all")

    def test_the_report_carries_the_notes(self):
        rep = runner.Report()
        rep.note("something could not be done", "why")
        self.assertTrue(rep.notes)
        self.assertIn("why", json.dumps(rep.as_dict()))

    def test_a_report_with_a_failure_is_not_a_pass(self):
        rep = runner.Report()
        rep.record("job", "step", 1, "host")
        self.assertNotEqual(rep.exit_code(), 0)

    def test_a_report_with_no_failure_passes(self):
        rep = runner.Report()
        rep.record("job", "step", 0, "host")
        self.assertEqual(rep.exit_code(), 0)

    def test_a_report_that_ran_nothing_is_not_a_pass(self):
        """Zero failures out of zero steps is the most flattering possible
        lie: it is what a broken runner reports."""
        self.assertNotEqual(runner.Report().exit_code(), 0)


HAVE_DOCKER = shutil.which("docker") is not None and subprocess.run(
    ["docker", "info"], capture_output=True).returncode == 0
needs_docker = unittest.skipUnless(HAVE_DOCKER, "docker is not running")


@needs_yaml
@needs_docker
@needs_checkout
class TestEachMatrixRowGetsItsOwnTree(unittest.TestCase):
    """**Found by running this tool, not by reasoning about it.**

    The runner gives every matrix job a fresh checkout. This reused one
    export for all of them, so the virtual environment built for one method
    was still sitting there when the next method ran -- and the guards that
    scan the repository read it. It surfaced as two methods reporting
    different numbers of tests from the same suite, which is the kind of
    difference that looks like a flaky test and is not one.
    """

    def test_a_row_cannot_see_what_the_previous_row_wrote(self):
        d = Path(tempfile.mkdtemp(prefix="matrixtree-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        wf = d / "probe.yml"
        wf.write_text(
            "on: [push]\n"
            "jobs:\n"
            "  probe:\n"
            "    runs-on: ubuntu-latest\n"
            "    strategy:\n"
            "      matrix:\n"
            "        n: ['1', '2']\n"
            "    steps:\n"
            "      - name: refuse a tree another row has touched\n"
            "        run: test ! -e leaked.txt && touch leaked.txt\n",
            encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(BIN / "run-ci-locally.py"),
             "--workflow", str(wf), "--event", "push"],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(
            r.returncode, 0,
            "the second matrix row saw a file the first one wrote:\n"
            + r.stdout[-2000:] + r.stderr[-2000:])


class TestTheCommandLine(unittest.TestCase):
    """The two dry-run tests need a checkout: the tool exports HEAD, which is
    the point of it. Without `.git` -- the container image, and the mutation
    tool's own work tree -- there is nothing to export and the run refuses,
    correctly."""

    def test_it_refuses_an_unknown_event(self):
        r = subprocess.run(
            [sys.executable, str(BIN / "run-ci-locally.py"),
             "--event", "nonsense", "--dry-run"],
            capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)

    @needs_yaml
    @needs_checkout
    def test_a_dry_run_prints_the_plan_without_running_it(self):
        r = subprocess.run(
            [sys.executable, str(BIN / "run-ci-locally.py"),
             "--event", "pull_request", "--dry-run"],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        for job in doc["jobs"]:
            with self.subTest(job=job):
                self.assertIn(job, r.stdout)

    @needs_yaml
    @needs_checkout
    def test_the_dry_run_reaches_the_real_methods(self):
        """The matrix has to be resolved for the plan to mean anything, and
        resolving it is the part that runs the discover step for real."""
        r = subprocess.run(
            [sys.executable, str(BIN / "run-ci-locally.py"),
             "--event", "pull_request", "--dry-run"],
            capture_output=True, text=True, cwd=ROOT)
        found = sorted(p.parent.name for p in
                       (ROOT / "methods").glob("*/requirements.lock.txt"))
        self.assertTrue(found)
        for m in found:
            with self.subTest(method=m):
                self.assertIn(m, r.stdout)


if __name__ == "__main__":
    unittest.main()
