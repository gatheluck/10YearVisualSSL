"""Is this a working checkout, or just the code?

Some tests assert properties of the **repository**: that nothing generated is
tracked, that the layout in the README matches, that the pre-commit hook
behaves, that the workflow runs the right things. None of those are properties
of the *code*, and none can be asked where there is no repository.

The container is exactly that place. `.dockerignore` keeps `.git` and
`.github` out of the image deliberately -- an image should not carry the
history or the CI definition -- and the image has no `git` binary either. So
running the whole suite inside it failed on tests that were never about it.

**Skipping has to be conditional on a positive check, not on an exception.**
A test that skips whenever anything goes wrong is a test that never runs.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def has_git() -> bool:
    return shutil.which("git") is not None


def is_checkout(root: Path = ROOT) -> bool:
    """True when `root` is inside a git work tree and git can be run."""
    if not has_git():
        return False
    r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                       cwd=root, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


needs_checkout = unittest.skipUnless(
    is_checkout(),
    "not a git checkout: these assert properties of the repository, and the "
    "container image deliberately carries neither .git nor git")
