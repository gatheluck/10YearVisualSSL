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

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")

# The locks the documented install uses. Naming one and not the other is the
# mistake that once made a correct environment look broken.
LOCKS = ("methods/1_context_prediction/requirements.lock.txt",
         "requirements-tools.lock.txt")


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


def triggers(doc: dict):
    # PyYAML reads the bare key `on` as the boolean True.
    return doc.get("on", doc.get(True))


class TestItExists(unittest.TestCase):
    def test_there_is_a_workflow(self):
        self.assertTrue(workflow_files(),
                        "no CI: the suite runs only where somebody remembers")

    def test_the_scan_finds_it(self):
        """Against an empty listing every check below passes vacuously."""
        self.assertIn("tests.yml", [p.name for p in workflow_files()])


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
        for name, doc in parsed().items():
            installs = [str(s.get("run", ""))
                        for spec in doc["jobs"].values()
                        for s in spec.get("steps", [])
                        if "pip install" in str(s.get("run", ""))]
            with self.subTest(file=name):
                self.assertTrue(installs, "nothing is installed anywhere")
                for ins in installs:
                    self.assertIn("--require-hashes", ins)
                    for lock in LOCKS:
                        self.assertIn(lock, ins, f"{lock} is not installed")

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
        for name, doc in parsed().items():
            runs = " ".join(
                str(s.get("run", "")) for spec in doc["jobs"].values()
                for s in spec.get("steps", []))
            with self.subTest(file=name):
                self.assertIn("docker build", runs)
                self.assertIn("contract-test.py", runs)


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


if __name__ == "__main__":
    unittest.main()
