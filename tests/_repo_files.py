"""Which files belong to this repository. **One implementation, shared.**

Three guards need this question answered -- the language scan, the platform
isolation scan, and the no-hard-coded-methods scan -- and it was answered
twice, differently. The two answers then diverged in the one place it mattered:

- `test_language.py` degraded to a filesystem walk when git was unavailable,
  so it ran inside the container image, which ships with no `.git` and no git
  binary
- `test_no_hard_coded_methods.py` raised, so three of its tests errored there
  and the pull request could not go green

That is the recurring root cause in this project, not a new one: **the same
rule implemented in two places, the scans disagreeing, and a false report
coming out of the difference.** The fix is not to skip the second scan where
git is missing -- that would answer a red CI by testing less. It is for there
to be one implementation and no second answer to diverge from.

The rule itself is git's own -- tracked, plus untracked and not ignored -- so
it is derived rather than listed and cannot go stale. It covers a file written
a moment ago and never added, which matters because the pre-commit hook runs
before anything is committed, and it excludes installed dependencies: the
READMEs say to build the environment at `.venv/` inside the repository, so a
plain walk reads jinja2, numpy, rich and torch's bundled headers, several of
which legitimately contain CJK. CI failed on exactly that.

Outside a work tree there is nothing to ask, so everything is a candidate.
That is the honest answer rather than a convenience: a shipped tree has no
repository to consult, and every file in it is one we published.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

LS_FILES = ["git", "ls-files", "--cached", "--others", "--exclude-standard",
            "-z"]

# What the fallback must not walk into. git's own directory is not part of
# what we publish, and its logs quote every branch name ever checked out --
# which made the no-hard-coded-methods guard report `.git/logs/HEAD` as
# shared machinery naming a method. Found by running the scan with no git on
# PATH, which is the only way it shows.
FALLBACK_SKIP = (".git",)


def git_available(root: Path) -> bool:
    """Whether git can answer for `root`.

    A missing binary raises `FileNotFoundError`; a tree that is not a checkout
    exits non-zero. Both mean the same thing here, and both must be handled --
    the container image is the first and a `docker build` context the second.
    """
    try:
        r = subprocess.run(LS_FILES, cwd=root, capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _walk(root: Path) -> list[Path]:
    """The answer where git cannot give one.

    It cannot consult `.gitignore`, so it is deliberately the **wider**
    answer: everything present except git's own directory. A guard that reads
    too much fails loudly and gets looked at; one that reads too little passes
    and says nothing, and this project has already shipped both -- the loud
    one was fixed in a day, the quiet one ran green for weeks.
    """
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink()
            and not set(p.relative_to(root).parts) & set(FALLBACK_SKIP)]


def repository_files(root: Path) -> list[Path]:
    """Every regular file belonging to `root`, symlinks excluded."""
    try:
        out = subprocess.run(LS_FILES, cwd=root, capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is None or out.returncode != 0:
        return _walk(root)
    paths = [root / rel for rel in out.stdout.split("\0") if rel]
    return sorted(p for p in paths if p.is_file() and not p.is_symlink())
