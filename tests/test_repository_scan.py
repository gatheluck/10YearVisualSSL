#!/usr/bin/env python3
"""The repository scan has one implementation, and it survives without git.

**Written immediately after the fourth occurrence of the same root cause.**
"Which files belong to this repository" was implemented twice. The two copies
answered identically everywhere git existed, so every local run agreed and
nothing looked wrong -- and diverged in the one environment that matters most,
the container image, which ships with no `.git` and no git binary because that
is what a published tree looks like. One copy degraded to a filesystem walk and
passed; the other raised `FileNotFoundError: 'git'` and errored three tests.

The tempting fix was to skip the scan where git is missing. That answers a red
CI by testing less, and it would have left the guard blind in precisely the
environment it most needs to see. The fix was one implementation.

So two properties are pinned here, and neither is a matter of remembering:

- the scan **works with no git on PATH**, checked by actually removing it
  rather than by reasoning that it would. Every previous attempt to verify
  this class of thing by reasoning was wrong, and one attempt to verify it by
  simulation was wrong too because the simulation still had git
- **no test module calls git directly unless it is gated**, so a new scan
  cannot quietly become the third implementation. `test_githooks` and
  `test_repo_hygiene` do call git directly -- they build repositories and
  drive hooks -- and are gated for it
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from _checkout import needs_checkout, needs_git            # noqa: E402
from _repo_files import _walk, git_available, repository_files  # noqa: E402,E501

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
GATES = ("needs_git", "needs_checkout")


def without_git(script: str) -> subprocess.CompletedProcess:
    """Run `script` in an environment where git does not exist.

    `PATH` points at an empty directory rather than being unset, because an
    unset `PATH` makes `shutil.which` fall back to a default path that still
    finds git. That is not a hypothetical: an earlier simulation of this did
    exactly that, reported success, and the real container failed anyway.
    """
    empty = Path(tempfile.mkdtemp(prefix="nogit-"))
    try:
        return subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT, text=True,
            capture_output=True, timeout=300,
            env={"PATH": str(empty), "HOME": str(empty),
                 "PYTHONDONTWRITEBYTECODE": "1"})
    finally:
        shutil.rmtree(empty, ignore_errors=True)


PROBE = (
    "import shutil, sys; sys.path.insert(0, 'tests')\n"
    "from pathlib import Path\n"
    "from _repo_files import repository_files, git_available\n"
    "assert shutil.which('git') is None, 'the simulation still has git'\n"
    "f = repository_files(Path('.'))\n"
    "print(len(f)); print(git_available(Path('.')))\n"
)


class TestTheScanSurvivesWithoutGit(unittest.TestCase):
    """The property that broke, proved by removing git rather than assuming."""

    def test_the_simulation_really_has_no_git(self):
        """A positive control. Without this the test below proves nothing.

        This is the exact step that was skipped last time: the check ran in an
        environment that still had git, passed, and the container failed.
        """
        r = without_git("import shutil; print(shutil.which('git'))")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "None",
                         "git is still reachable, so nothing here is tested")

    def test_the_scan_still_returns_files(self):
        r = without_git(PROBE)
        self.assertEqual(r.returncode, 0,
                         f"the scan failed with no git:\n{r.stderr}")
        count, available = r.stdout.split()
        self.assertGreater(int(count), 20,
                           "the scan fell back to nothing, which would make "
                           "every guard built on it vacuously pass")
        self.assertEqual(available, "False")

    def test_git_availability_is_reported_not_assumed(self):
        """`git_available` must answer False rather than raise."""
        r = without_git(
            "import sys; sys.path.insert(0, 'tests')\n"
            "from pathlib import Path\n"
            "from _repo_files import git_available\n"
            "print(git_available(Path('.')))\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "False")

    def test_every_module_that_could_need_git_runs_without_it(self):
        """The end the container reaches, for the tests that could reach it.

        This ran the **whole** suite for a while, and that was right in
        principle and wrong in cost: it spawns the suite inside the suite, so
        it grew with every method added and eventually exceeded its own
        timeout in CI -- 681 tests, ten minutes. A check that cannot survive
        this repository's own goal of thirty-seven methods is a defect, which
        is the same complaint made of the CI matrix.

        The set is narrowed to what can actually be affected -- modules that
        mention git or that import the shared scan -- and **derived, not
        listed**, so a new one joins by itself. It is not the earlier narrow
        version: that looked only at importers of the scan and would have
        missed the failure sitting in this very file. Both bugs the wide
        version caught are in modules this set contains, which was checked
        before narrowing it.

        Itself excluded: included, it would spawn this subprocess from inside
        the subprocess and never finish. Measured -- it times out.
        """
        mods = sorted(
            p.stem for p in TESTS.glob("test_*.py")
            if p.stem != Path(__file__).stem
            and ("git" in p.read_text(encoding="utf-8")
                 or "_repo_files" in p.read_text(encoding="utf-8")))
        self.assertGreater(len(mods), 3, "the discovery found nothing to run")
        r = without_git(
            "import sys, unittest; sys.path.insert(0, 'tests')\n"
            f"m = {mods!r}\n"
            "r = unittest.TextTestRunner(verbosity=0).run("
            "unittest.defaultTestLoader.loadTestsFromNames(m))\n"
            "sys.exit(0 if r.wasSuccessful() else 1)\n")
        self.assertEqual(
            r.returncode, 0,
            "the suite does not survive without git, which is what the "
            f"container image is:\n{r.stderr[-3000:]}")


@needs_checkout
class TestWhereGitExistsItsAnswerIsUsed(unittest.TestCase):
    """The other side of the fallback, and why it is a separate class.

    These assert that git *is* consulted, so their premise is a checkout --
    `needs_checkout`, not `needs_git`. The binary being installed is not
    enough and the difference is not academic: the mutation tool copies the
    tree to a temporary directory without `.git`, where git exists and answers
    "not a repository", and gating on the binary alone made the baseline red
    there. Inside the container it is not that these fail, it is that there is
    nothing to ask.

    Splitting them out was forced by the container simulation, which is the
    point of running it: written into the class above they made the whole
    class require git, and that class exists precisely to prove the scan works
    without it.
    """

    def test_git_is_what_answers_here(self):
        self.assertTrue(git_available(ROOT))

    def test_the_answer_is_narrower_than_a_plain_walk(self):
        """Otherwise the fallback has silently become the only path, and the
        `.venv` exclusion -- the reason any of this exists -- is gone."""
        walked = [p for p in ROOT.rglob("*")
                  if p.is_file() and not p.is_symlink()]
        self.assertLess(len(repository_files(ROOT)), len(walked))

    def test_gits_own_directory_is_never_included(self):
        self.assertNotIn(ROOT / ".git" / "config", repository_files(ROOT))


class TestTheFallbackItself(unittest.TestCase):
    """The fallback, checked on a directory built for the purpose.

    **Both of these were added because a mutation survived**, and neither
    survivor was an equivalent mutant -- each was a case nothing exercised:

    - emptying `FALLBACK_SKIP` survived, because the mutation tool copies the
      tree without `.git` and there was then no `.git` to walk into. The rule
      was only ever tested against whatever the ambient directory happened to
      contain
    - making `git_available` return True unconditionally survived, because
      with no git the function returns earlier and the mutated line is never
      reached. "Git is installed but this is not a repository" -- a `docker
      build` context, an unpacked tarball, the mutation tool's own work tree
      -- had no test at all

    So these build the directory instead of hoping to be run in one.
    """

    def tree(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="fallback-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "kept.py").write_text("x = 1\n", encoding="utf-8")
        (d / ".git" / "logs").mkdir(parents=True)
        (d / ".git" / "logs" / "HEAD").write_text("checkout: branch\n",
                                                  encoding="utf-8")
        return d

    def test_the_fallback_returns_ordinary_files(self):
        d = self.tree()
        self.assertIn(d / "kept.py", _walk(d))

    def test_an_installed_environment_is_not_our_text(self):
        """**The original CI failure, arriving by the back door.**

        The READMEs say to build the environment at `.venv/` inside the
        repository. With git, `.gitignore` excludes it. Without git, the walk
        read jinja2's bundled CJK and the language guard failed -- which is
        the exact failure this scan was written to prevent. Found by running
        the whole suite with no git in a job that had a `.venv`; no local run
        showed it, because every venv here is made in /tmp.
        """
        d = self.tree()
        env = d / ".venv" / "lib" / "pkg"
        env.mkdir(parents=True)
        (d / ".venv" / "pyvenv.cfg").write_text("home = /usr\n",
                                                encoding="utf-8")
        (env / "vendored.py").write_text("x = 1\n", encoding="utf-8")
        self.assertNotIn(env / "vendored.py", _walk(d))
        self.assertIn(d / "kept.py", _walk(d))

    def test_the_environment_is_found_by_its_marker_not_its_name(self):
        """`.venv`, `venv`, `env`, `.direnv` are conventions, and a list of
        conventions is correct until somebody picks a different one. PEP 405
        requires the marker, so an environment identifies itself."""
        d = self.tree()
        for name in ("venv", "env", "whatever-i-called-it"):
            (d / name).mkdir()
            (d / name / "pyvenv.cfg").write_text("home = /usr\n",
                                                 encoding="utf-8")
            (d / name / "vendored.py").write_text("x = 1\n", encoding="utf-8")
        walked = _walk(d)
        for name in ("venv", "env", "whatever-i-called-it"):
            with self.subTest(directory=name):
                self.assertNotIn(d / name / "vendored.py", walked)

    def test_a_directory_without_the_marker_is_still_ours(self):
        """The exclusion must not swallow real content. A directory is only
        skipped when it says what it is."""
        d = self.tree()
        (d / "environments").mkdir()
        (d / "environments" / "notes.py").write_text("x = 1\n",
                                                     encoding="utf-8")
        self.assertIn(d / "environments" / "notes.py", _walk(d))

    def test_the_fallback_never_walks_into_git(self):
        """`.git/logs/HEAD` quotes every branch name ever checked out, so the
        method-naming guard read it and reported the repository's own history
        as shared machinery naming a method."""
        d = self.tree()
        self.assertNotIn(d / ".git" / "logs" / "HEAD", _walk(d))

    def test_a_pinned_submodule_is_not_our_text(self):
        """Upstream code under a `third_party/<name>/` submodule is the authors',
        not ours. With git, `git ls-files` stops at the submodule boundary;
        without git, the walk must exclude the same paths, read from the tracked
        `.gitmodules`. Their `qsub`/HPC scripts tripped the platform-isolation
        guard and their non-English text the language guard when the walk
        descended into a real pinned submodule."""
        # A fabricated submodule with a fake upstream name, so this shared file
        # does not itself name a real method (the no-hard-coded-methods guard).
        d = self.tree()
        (d / ".gitmodules").write_text(
            '[submodule "third_party/vendorlib"]\n'
            "\tpath = third_party/vendorlib\n"
            "\turl = https://github.com/example/vendorlib\n", encoding="utf-8")
        up = d / "third_party" / "vendorlib" / "demo"
        up.mkdir(parents=True)
        (up / "run_demo.ipynb").write_text("qsub abci\n", encoding="utf-8")
        walked = _walk(d)
        self.assertNotIn(up / "run_demo.ipynb", walked)
        self.assertIn(d / "kept.py", walked)

    @needs_git
    def test_git_installed_but_not_a_repository_is_not_available(self):
        """The middle case: the binary answers, and its answer is no."""
        d = Path(tempfile.mkdtemp(prefix="notarepo-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        (d / "loose.py").write_text("x = 1\n", encoding="utf-8")
        self.assertFalse(git_available(d))
        self.assertEqual(repository_files(d), [d / "loose.py"])


class TestThereIsOnlyOneScan(unittest.TestCase):
    """A second implementation is how the divergence happened."""

    @staticmethod
    def modules() -> list[Path]:
        return sorted(TESTS.glob("test_*.py"))

    @staticmethod
    def runs_git(tree: ast.AST) -> bool:
        """Whether anything under `tree` puts "git" first in a command list."""
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and node.elts:
                first = node.elts[0]
                if isinstance(first, ast.Constant) and first.value == "git":
                    return True
        return False

    @classmethod
    def reaches_git(cls, tree: ast.Module, node: ast.ClassDef,
                    method: "ast.FunctionDef | None" = None) -> bool:
        """Whether `node` can reach git, directly or via a module helper.

        **Asking this of the whole module was too coarse and said so loudly.**
        `test_language.py` builds real repositories in one gated class, and a
        module-wide answer therefore accused all six of its other classes --
        which reach git through nothing but the shared scan, and run happily
        without it. An accusation that lands on innocent code gets silenced,
        and a silenced check protects nothing.
        """
        # **Scoped to the method when one is given.** A class-wide answer
        # accused a test that only checks argparse, because a *sibling* test
        # in the same class ran `git ls-tree`. An accusation that lands on
        # innocent code gets silenced, and a silenced guard protects nothing.
        # Helpers still count: a test that calls one inherits its reach.
        helpers = {f.name for f in tree.body
                   if isinstance(f, ast.FunctionDef) and cls.runs_git(f)}
        helpers |= {f.name for f in node.body
                    if isinstance(f, ast.FunctionDef)
                    and not f.name.startswith("test") and cls.runs_git(f)}
        scope = method if method is not None else node
        if cls.runs_git(scope):
            return True
        called = {c.func.id for c in ast.walk(scope)
                  if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        called |= {c.func.attr for c in ast.walk(scope)
                   if isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Attribute)}
        return bool(helpers & called)

    @classmethod
    def ungated(cls, path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loose = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if any(g in d for d in map(ast.unparse, node.decorator_list)
                   for g in GATES):
                continue
            for item in node.body:
                if not (isinstance(item, ast.FunctionDef)
                        and item.name.startswith("test")):
                    continue
                if not cls.reaches_git(tree, node, item):
                    continue
                marks = set(map(ast.unparse, item.decorator_list))
                if any(g in d for d in marks for g in GATES):
                    continue
                if any("skip" in d for d in marks):
                    continue
                loose.append(f"{node.name}.{item.name}")
        return loose

    def test_the_scan_lives_in_one_place(self):
        """No test module may implement the published-set rule itself.

        The rule is "tracked, plus untracked and not ignored", and the git
        flag naming that second half is what distinguishes it from merely
        listing tracked files -- which `test_repo_hygiene` legitimately does,
        for a different question, and is gated for.

        The needle is assembled from pieces rather than written out, because
        written out it appears in this file and the check accuses itself. The
        language guard spells its own alphabet in escapes for the same
        reason: a guard must not be the one thing exempt from itself.
        """
        needle = "--exclude-" + "standard"
        others = [p.name for p in self.modules()
                  if needle in p.read_text(encoding="utf-8")]
        self.assertEqual(
            others, [],
            "a second implementation of the repository scan. The two copies "
            "agreed everywhere git existed and diverged in the container:\n"
            + "\n".join(f"  - {x}" for x in others))

    def test_more_than_one_guard_shares_it(self):
        """With one consumer, sharing proves nothing."""
        users = [p.name for p in self.modules()
                 if "_repo_files" in p.read_text(encoding="utf-8")]
        self.assertGreater(len(users), 2, users)

    def test_some_module_still_calls_git_directly(self):
        """Against a check that has quietly stopped matching anything."""
        self.assertTrue([p for p in self.modules()
                         if self.runs_git(ast.parse(
                             p.read_text(encoding="utf-8")))])

    def test_no_direct_git_call_is_ungated(self):
        offenders = [f"{p.name}::{n}"
                     for p in self.modules() for n in self.ungated(p)]
        self.assertEqual(
            offenders, [],
            "these run git with no gate, so they fail where there is none:\n"
            + "\n".join(f"  - {x}" for x in offenders))

    def test_the_detector_sees_an_ungated_call(self):
        d = Path(tempfile.mkdtemp(prefix="scanspec-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = d / "test_sample.py"
        p.write_text('import unittest\n'
                     'class TestX(unittest.TestCase):\n'
                     '    def test_y(self):\n'
                     '        subprocess.run(["git", "status"])\n',
                     encoding="utf-8")
        self.assertEqual(self.ungated(p), ["TestX.test_y"])

    def test_a_test_that_reaches_git_through_a_helper_is_caught(self):
        """Scoping to the method must not lose the helper case: a test whose
        own body is clean but which calls something that runs git is just as
        broken where there is no git."""
        d = Path(tempfile.mkdtemp(prefix="scanspec-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = d / "test_sample.py"
        p.write_text('import unittest\n'
                     'class TestX(unittest.TestCase):\n'
                     '    def helper(self):\n'
                     '        subprocess.run(["git", "status"])\n'
                     '    def test_y(self):\n'
                     '        self.helper()\n',
                     encoding="utf-8")
        self.assertEqual(self.ungated(p), ["TestX.test_y"])

    def test_a_sibling_that_never_touches_git_is_left_alone(self):
        """The false positive that prompted the change: one test in a class
        ran git and every other test in it was accused."""
        d = Path(tempfile.mkdtemp(prefix="scanspec-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = d / "test_sample.py"
        p.write_text('import unittest\n'
                     'class TestX(unittest.TestCase):\n'
                     '    @needs_checkout\n'
                     '    def test_uses_git(self):\n'
                     '        subprocess.run(["git", "status"])\n'
                     '    def test_innocent(self):\n'
                     '        self.assertTrue(True)\n',
                     encoding="utf-8")
        self.assertEqual(self.ungated(p), [])

    def test_the_detector_accepts_a_gated_one(self):
        d = Path(tempfile.mkdtemp(prefix="scanspec-"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        p = d / "test_sample.py"
        p.write_text('import unittest\n'
                     '@needs_git\n'
                     'class TestX(unittest.TestCase):\n'
                     '    def test_y(self):\n'
                     '        subprocess.run(["git", "status"])\n',
                     encoding="utf-8")
        self.assertEqual(self.ungated(p), [])

    def test_the_detector_ignores_other_commands(self):
        self.assertFalse(self.runs_git(
            ast.parse('subprocess.run(["docker", "build", "."])')))


if __name__ == "__main__":
    unittest.main()
