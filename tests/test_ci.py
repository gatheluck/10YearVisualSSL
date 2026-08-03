#!/usr/bin/env python3
"""The workflow has to run what it claims to run.

**A CI job that cannot fail is worse than no CI**: it converts "nobody
checked" into "something checked and it was fine", which is a false statement
nobody goes looking for. So the workflow is checked the same way the
Dockerfile is -- by reading it, mechanically.

Why it exists at all. The pre-commit hook is machinery, but machinery on one
machine: it needs `git config core.hooksPath .githooks` per clone,
`--no-verify` skips it, and it only ever exercised whatever platform the
committer had. Every end-to-end check up to now ran on macOS arm64, while the
deployment target is linux x86_64.

Two kinds of check below, deliberately:

- **Structural**, needing PyYAML. Thorough, and skipped where it is absent
- **Textual**, always run. Only the patterns that would silently swallow a
  failure -- those are worth catching even without a parser
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _checkout import needs_checkout        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")

# The tooling lock is the same for every method; the method lock comes from
# the matrix. Naming a method here would be the very list this file forbids.
TOOLING_LOCK = "requirements-tools.lock.txt"


def workflow_files() -> list[Path]:
    if not WORKFLOWS.is_dir():
        return []
    return sorted(p for p in WORKFLOWS.iterdir()
                  if p.suffix in (".yml", ".yaml"))


def parsed() -> dict:
    """The workflow, by name. `on:` is YAML's `True`, which is a trap."""
    out = {}
    for p in workflow_files():
        out[p.name] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return out


def runs_of(doc: dict, job: str) -> str:
    """The shell of one job.

    Joining every job's steps and searching that was how five mutations
    survived: a check satisfied by one job hid its removal from another.
    """
    spec = doc["jobs"][job]
    return "\n".join(str(st.get("run", "")) for st in spec.get("steps", []))


def discovery_script(doc: dict) -> str:
    """The python the discover job runs, lifted out of its heredoc."""
    text = runs_of(doc, "discover")
    # After the newline, not after the marker: the rest of that line is shell
    # redirection, and including it made the script a syntax error.
    start = text.index("\n", text.index("<<'PY'")) + 1
    return text[start:text.index("\nPY", start)]


def triggers(doc: dict):
    # PyYAML reads the bare key `on` as the boolean True.
    return doc.get("on", doc.get(True))


@needs_checkout
class TestItExists(unittest.TestCase):
    def test_there_is_a_workflow(self):
        self.assertTrue(workflow_files(),
                        "no CI: the suite runs only where somebody remembers")

    def test_the_scan_finds_it(self):
        """Against an empty listing every check below passes vacuously."""
        self.assertIn("tests.yml", [p.name for p in workflow_files()])


@needs_checkout
class TestNothingSwallowsAFailure(unittest.TestCase):
    """Textual, so these hold even without a parser. Each of these is a way
    to have a green tick over a failed command."""

    def test_no_step_continues_on_error(self):
        for p in workflow_files():
            with self.subTest(file=p.name):
                self.assertNotIn("continue-on-error", p.read_text())

    def test_no_command_is_forced_to_succeed(self):
        for p in workflow_files():
            text = p.read_text()
            with self.subTest(file=p.name):
                self.assertNotRegex(text, r"\|\|\s*true")
                self.assertNotRegex(text, r"\bexit\s+0\s*$")
                self.assertNotRegex(text, r"set\s+\+e")

    def test_no_step_runs_regardless_of_what_failed(self):
        """`if: always()` on a checking step reports on a build that died."""
        for p in workflow_files():
            with self.subTest(file=p.name):
                self.assertNotRegex(p.read_text(), r"if:\s*always\(\)")

    def test_the_detector_would_notice_those_patterns(self):
        """Guard the checks above: they must match what they describe."""
        for bad in ("continue-on-error: true", "pytest || true", "set +e",
                    "if: always()"):
            with self.subTest(pattern=bad):
                self.assertTrue(
                    "continue-on-error" in bad
                    or re.search(r"\|\|\s*true", bad)
                    or re.search(r"set\s+\+e", bad)
                    or re.search(r"if:\s*always\(\)", bad))


@needs_checkout
class TestWhenItRuns(unittest.TestCase):
    @needs_yaml
    def test_it_runs_on_push_and_on_pull_request(self):
        """Push alone leaves a merge unchecked; pull_request alone leaves a
        direct push unchecked."""
        for name, doc in parsed().items():
            with self.subTest(file=name):
                on = triggers(doc)
                self.assertIsNotNone(on, "no trigger at all")
                keys = set(on) if isinstance(on, (dict, list)) else {on}
                self.assertIn("push", keys)
                self.assertIn("pull_request", keys)

    @needs_yaml
    def test_a_plain_push_runs_the_whole_suite(self):
        """Not merely "some job runs".

        The first version asked only that *a* job be unconditional, which a
        single leftover job satisfies while the tests themselves have moved
        behind a pull-request condition. What has to hold is that a direct
        commit still runs the suite, both ways it is run.
        """
        for name, doc in parsed().items():
            on_push = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                if "if" not in spec for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("tests/run-tests.sh", on_push,
                              "a plain push does not run the suite")
                self.assertIn("unittest discover", on_push,
                              "a plain push never runs the dependent tests")

    @needs_yaml
    def test_a_conditional_job_says_what_it_waits_for(self):
        """A condition nobody can read is a job nobody knows runs."""
        for name, doc in parsed().items():
            for job, spec in doc["jobs"].items():
                if "if" not in spec:
                    continue
                with self.subTest(file=name, job=job):
                    self.assertIn("event_name", str(spec["if"]))


@needs_checkout
class TestWhatItRuns(unittest.TestCase):
    @needs_yaml
    def test_the_suite_is_run_by_its_own_script(self):
        """`run-tests.sh` is what the hook runs and what the documentation
        tells people to run. CI running something else would check a different
        thing from everybody else."""
        for name, doc in parsed().items():
            runs = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("tests/run-tests.sh", runs)

    @needs_yaml
    def test_the_whole_suite_is_run_with_the_dependencies_present(self):
        """The stdlib-only run skips every torch test and reports it. If that
        were the only job, the skips would never be filled in anywhere."""
        for name, doc in parsed().items():
            runs = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("unittest discover", runs)

    @needs_yaml
    def test_the_install_is_hash_checked_and_names_every_lock(self):
        """Both locks, with the method's one coming from the matrix.

        This used to name method 1's lock literally, which is how a second
        method arrived uncovered. It now requires the pair without naming
        either method.
        """
        for name, doc in parsed().items():
            installs = [str(s.get("run", ""))
                        for spec in doc["jobs"].values()
                        for s in spec.get("steps", [])
                        if "pip install" in str(s.get("run", ""))]
            with self.subTest(file=name):
                self.assertTrue(installs, "nothing is installed anywhere")
                for ins in installs:
                    self.assertIn("--require-hashes", ins)
                    self.assertIn(TOOLING_LOCK, ins)
                    self.assertIn("matrix.", ins,
                                  "the method lock is not from the matrix")
                    self.assertIn("requirements.lock.txt", ins)

    @needs_yaml
    def test_the_environment_is_verified_against_the_locks(self):
        """Installing from a lock and having installed it are different
        claims. CI makes the second one, or it is not worth making."""
        for name, doc in parsed().items():
            runs = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("verify-environment.py", runs)

    @needs_yaml
    def test_the_container_is_built_and_exercised(self):
        """Built, checked against the lock, and made to run the suite.

        It used to run a hand-written contract chain that only ever fitted
        one method. Running the suite inside the image covers the same chain
        -- each method's smoke tests are that chain -- and covers every
        method rather than the one the chain was written for.
        """
        for name, doc in parsed().items():
            runs = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("docker build", runs_of(doc, "container"))
                self.assertIn("verify-environment.py", runs_of(doc, "locked"))
                self.assertIn("unittest discover", runs_of(doc, "locked"))


@needs_checkout
class TestEveryMethodIsExercised(unittest.TestCase):
    """A method CI never installs is a method CI never tests.

    The `locked` job named one method's lock, so when the second method
    arrived **all twelve of its dependent tests skipped in CI** and the job
    still reported success. Skips are reported, but a green tick over twelve
    silent skips is exactly the "something checked and it was fine" that this
    file exists to prevent.

    Methods must be **discovered, not listed**: a hand-written matrix is a
    list that goes stale the moment a method is added, which is the mistake
    this repository has now made three times.
    """

    def method_locks(self) -> list[str]:
        return sorted(
            str(p.relative_to(ROOT))
            for p in (ROOT / "methods").glob("*/requirements.lock.txt"))

    def test_there_is_more_than_one_method_to_cover(self):
        """With one method a per-method matrix proves nothing."""
        self.assertGreater(len(self.method_locks()), 1)

    @needs_yaml
    def test_the_methods_are_discovered_rather_than_listed(self):
        """Any method name written into the workflow is a name that can rot."""
        for name, doc in parsed().items():
            text = (WORKFLOWS / name).read_text()
            for lock in self.method_locks():
                method = lock.split("/")[1]
                with self.subTest(file=name, method=method):
                    self.assertNotIn(
                        f"methods/{method}/", text,
                        f"{method} is named in the workflow; methods must be "
                        "discovered so a new one is covered without an edit")

    @needs_yaml
    def test_each_per_method_job_runs_over_the_matrix(self):
        """Checked job by job. A matrix on one of them is not a matrix on the
        other, and joining their text hid exactly that."""
        for name, doc in parsed().items():
            for job in ("locked", "container"):
                with self.subTest(file=name, job=job):
                    spec = doc["jobs"][job]
                    self.assertIn("matrix", str(spec.get("strategy", "")),
                                  f"{job} does not run over the matrix")
                    self.assertIn("matrix.", runs_of(doc, job),
                                  f"{job} never uses the matrix value")

    @needs_yaml
    def test_the_discovery_finds_methods_by_looking(self):
        """Run the script itself, against a tree it has never seen."""
        import subprocess, tempfile
        doc = parsed()["tests.yml"]
        script = discovery_script(doc)
        d = Path(tempfile.mkdtemp(prefix="discover-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        for m in ("aaa_first", "zzz_second"):
            (d / "methods" / m).mkdir(parents=True)
            (d / "methods" / m / "requirements.lock.txt").write_text("x==1\n")
        env = {**os.environ, "GITHUB_OUTPUT": str(d / "out.txt")}
        r = subprocess.run([sys.executable, "-c", script], cwd=d, env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("aaa_first", r.stdout)
        self.assertIn("zzz_second", r.stdout,
                      "the discovery found only some of the methods")

    @needs_yaml
    def test_the_discovery_refuses_to_find_nothing(self):
        """An empty matrix runs no jobs and reports success -- the quietest
        possible way for the suite to stop running."""
        import subprocess, tempfile
        script = discovery_script(parsed()["tests.yml"])
        d = Path(tempfile.mkdtemp(prefix="discover-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "methods").mkdir()
        r = subprocess.run([sys.executable, "-c", script], cwd=d,
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0,
                            "finding no methods was reported as success")

    @needs_yaml
    def test_the_matrix_comes_from_a_discovery_step(self):
        for name, doc in parsed().items():
            jobs = doc["jobs"]
            with self.subTest(file=name):
                matrixed = [j for j, spec in jobs.items()
                            if "matrix" in str(spec.get("strategy", ""))]
                self.assertTrue(matrixed, "no job runs over a matrix")
                for j in matrixed:
                    self.assertIn("needs", jobs[j],
                                  f"{j} has a matrix that nothing computes")

    @needs_yaml
    def test_the_container_job_builds_and_exercises_each_image(self):
        """All three, in the container job specifically."""
        for name, doc in parsed().items():
            runs = runs_of(doc, "container")
            with self.subTest(file=name):
                self.assertIn("docker build", runs)
                self.assertIn("verify-environment.py", runs,
                              "the image is never checked against its lock")
                self.assertIn("unittest discover", runs,
                              "the suite never runs inside the image")


@needs_checkout
class TestItIsReproducibleToo(unittest.TestCase):
    """The same discipline as everywhere else: pinned, and on the platform
    being claimed."""

    @needs_yaml
    def test_every_action_is_pinned_to_a_commit(self):
        """A tag moves. `actions/checkout@v4` is a different thing next month,
        and a run that cannot be repeated proves less than it looks."""
        for name, doc in parsed().items():
            for job, spec in doc["jobs"].items():
                for step in spec.get("steps", []):
                    uses = step.get("uses")
                    if not uses:
                        continue
                    with self.subTest(file=name, job=job, uses=uses):
                        self.assertRegex(
                            uses, r"@[0-9a-f]{40}$",
                            f"{uses} is pinned to a tag, not a commit")

    @needs_yaml
    def test_the_interpreter_comes_from_the_repository_pin(self):
        """Writing the version into the workflow would let it drift from
        `.python-version`, and the cp312 wheels would stop installing."""
        for name, doc in parsed().items():
            setups = [s for spec in doc["jobs"].values()
                      for s in spec.get("steps", [])
                      if "setup-python" in str(s.get("uses", ""))]
            with self.subTest(file=name):
                self.assertTrue(setups, "the interpreter is never set up")
                for s in setups:
                    with_ = s.get("with", {})
                    self.assertEqual(with_.get("python-version-file"),
                                     ".python-version",
                                     "the version is not taken from the pin")
                    self.assertNotIn("python-version", with_,
                                     "a hard-coded version can drift")

    @needs_yaml
    def test_it_runs_on_the_platform_we_claim_to_support(self):
        """The point of having CI here is linux x86_64: everything
        end-to-end so far ran on macOS arm64."""
        for name, doc in parsed().items():
            for job, spec in doc["jobs"].items():
                with self.subTest(file=name, job=job):
                    self.assertIn("ubuntu", str(spec.get("runs-on", "")))

    @needs_yaml
    def test_it_asks_for_no_more_access_than_it_needs(self):
        for name, doc in parsed().items():
            with self.subTest(file=name):
                self.assertEqual(doc.get("permissions", {}).get("contents"),
                                 "read")


@needs_yaml
class TestItDoesNotBurnMinutesTwice(unittest.TestCase):
    """**Written after the workflow stopped running for lack of minutes.**

    The merge of the third method failed CI with no failing step and no log:
    the jobs completed two seconds after starting, with zero steps, and the
    check annotation said the account had hit its spending limit. Nothing was
    wrong with the code, and the first instinct -- hunt the regression -- was
    the wrong one.

    The cost was measured rather than guessed. One full run is about 37
    minutes of billed runner time (`locked` about 6 minutes per method,
    `container` about 6 more), and `on: push:` with no branch filter meant
    **every commit on a branch with an open pull request started two runs**:
    one for the push and one for the pull request. Roughly 56 billed minutes
    per commit, for a suite that takes 15 seconds locally.

    It also scales with the thing this repository is for. Three methods cost
    37 minutes; thirty-seven would cost several hours per run. A design that
    cannot survive its own stated goal is a defect, not a surprise.

    None of this is a substitute for the account having minutes. It is the
    half that is ours.
    """

    def test_push_only_builds_the_default_branch(self):
        """The duplicate run, removed.

        Nothing loses coverage: a branch cannot reach the default branch
        except through a pull request, and `pull_request` covers that. What
        is dropped is the second, identical run of the same commit.
        """
        for name, doc in parsed().items():
            with self.subTest(file=name):
                push = triggers(doc).get("push")
                self.assertIsInstance(
                    push, dict,
                    "push has no branch filter, so every commit on every "
                    "branch runs the suite a second time")
                self.assertEqual(push.get("branches"), ["main"])

    def test_pull_requests_are_not_filtered_away(self):
        """The other half. Restricting both would leave nothing running."""
        for name, doc in parsed().items():
            with self.subTest(file=name):
                self.assertIn("pull_request", triggers(doc))

    def test_a_superseded_run_is_cancelled(self):
        """Pushing twice in a minute otherwise pays for both, and only the
        second one's answer is wanted."""
        for name, doc in parsed().items():
            with self.subTest(file=name):
                c = doc.get("concurrency")
                self.assertIsInstance(c, dict, "no concurrency group")
                self.assertIn("group", c)
                self.assertIn("cancel-in-progress", c)

    def test_the_default_branch_is_never_cancelled(self):
        """A cancelled run on the default branch destroys the record of
        whether that commit was good, which is the one record worth paying
        for."""
        for name, doc in parsed().items():
            with self.subTest(file=name):
                cancel = str(doc["concurrency"]["cancel-in-progress"])
                self.assertIn("github.ref", cancel,
                              "cancellation is unconditional, so a run on "
                              "the default branch can be thrown away")
                self.assertIn("main", cancel)

    def test_the_concurrency_group_separates_branches(self):
        """One group for everything would cancel unrelated work."""
        for name, doc in parsed().items():
            with self.subTest(file=name):
                self.assertIn("github.ref",
                              str(doc["concurrency"]["group"]))

    def test_the_expensive_job_caches_its_installation(self):
        """Most of `locked` is downloading torch, once per method per run.

        Keyed on the lock files, so a changed lock misses the cache and a
        run can never install something the lock does not describe.
        """
        for name, doc in parsed().items():
            steps = doc["jobs"]["locked"]["steps"]
            setup = [s for s in steps
                     if "setup-python" in str(s.get("uses", ""))]
            with self.subTest(file=name):
                self.assertTrue(setup, "locked does not set up python")
                w = setup[0].get("with") or {}
                self.assertEqual(w.get("cache"), "pip")
                self.assertIn("lock", str(w.get("cache-dependency-path", "")))


@needs_checkout
@needs_yaml
class TestThePinnedSubmoduleIsCheckedOut(unittest.TestCase):
    """A pinned submodule under `third_party/` is empty unless the checkout
    asks for it, and `actions/checkout` does not by default.

    Gated on a checkout, like the other workflow-reading classes here: the
    container image ships without `.github` (`.dockerignore` excludes it) and
    without git, so the workflow is not there to read and the class skips --
    rather than failing on an absent file, which is what an ungated version did
    inside the image.

    Every job here goes on to read the working tree -- the file-scans walk it,
    the method that imports the pinned upstream reads it, and even the no-deps
    `core` job needs it, because `local_modules` finds the upstream's `models`
    and `util` packages by looking in `third_party/<sub>` and would otherwise
    report the adapter's imports as undeclared. So every checkout must fetch
    submodules, or the job tests a tree with a hole where the upstream should be
    and reports it green.
    """

    def test_every_checkout_fetches_submodules(self):
        for name, doc in parsed().items():
            for job, spec in doc["jobs"].items():
                for st in spec.get("steps", []):
                    if "actions/checkout" not in str(st.get("uses", "")):
                        continue
                    with self.subTest(workflow=name, job=job):
                        got = str((st.get("with") or {}).get("submodules", ""))
                        self.assertEqual(
                            got, "recursive",
                            f"{job}: checkout does not fetch submodules "
                            "(with.submodules: recursive); third_party/ would "
                            "be empty and the job would test a tree with a hole")

    def test_there_is_a_checkout_to_check(self):
        """Against a workflow with no checkout the check above passes
        vacuously."""
        found = [st for doc in parsed().values()
                 for spec in doc["jobs"].values()
                 for st in spec.get("steps", [])
                 if "actions/checkout" in str(st.get("uses", ""))]
        self.assertTrue(found, "no actions/checkout step found to check")


if __name__ == "__main__":
    unittest.main()
