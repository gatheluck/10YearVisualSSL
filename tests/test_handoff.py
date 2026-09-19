"""Exercise the documented fresh-clone workflow against isolated local remotes."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from _checkout import needs_git

ROOT = Path(__file__).resolve().parent.parent


class TestHandoff(unittest.TestCase):
    def test_entrypoint_links_resolve(self):
        for rel in ("AGENTS.md", "docs/HANDOFF.md"):
            path = ROOT / rel
            self.assertTrue(path.is_file(), rel)
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text())
            self.assertTrue(links, rel)
            for target in links:
                if "://" not in target and not target.startswith("#"):
                    self.assertTrue((path.parent / target.split("#")[0]).is_file(), target)

    @needs_git
    def test_fresh_clone_restores_pins_hooks_and_preserves_existing_checkout(self):
        doc = ROOT / "docs/HANDOFF.md"
        self.assertTrue(doc.is_file(), "missing handoff instructions")
        match = re.search(r"<!-- handoff-fresh-clone -->\s*```sh\n(.*?)\n```",
                          doc.read_text(), re.S)
        self.assertIsNotNone(match, "missing executable fresh-clone recipe")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = os.environ.copy()
            for key in tuple(env):
                if key.startswith("GIT_"):
                    del env[key]
            env.update(HOME=tmp, GIT_CONFIG_NOSYSTEM="1",
                       GIT_CONFIG_GLOBAL=str(root / "gitconfig"),
                       GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
                       GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid",
                       GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.file.allow",
                       GIT_CONFIG_VALUE_0="always")

            def git(*args):
                return subprocess.run(["git", *map(str, args)], env=env,
                                      capture_output=True, text=True, check=True).stdout.strip()

            def repo(name):
                path = root / name
                git("init", "-b", "main", path)
                (path / "fixture.txt").write_text(name)
                git("-C", path, "add", ".")
                git("-C", path, "commit", "-m", "fixture")
                return path

            upstream = repo("upstream")
            port = repo("port-source")
            private = repo("private-source")
            git("-C", port, "submodule", "add", upstream, "third_party/example")
            (port / ".githooks").mkdir()
            (port / ".githooks/pre-commit").write_text("#!/bin/sh\nexit 0\n")
            git("-C", port, "add", ".")
            git("-C", port, "commit", "-m", "pinned dependency and hooks")
            env.update(PORT_REMOTE=str(port), CAPTURE_REMOTE=str(private),
                       PORT_DIR=str(root / "new port"), CAPTURE_DIR=str(root / "new private"))
            result = subprocess.run(["bash", "-eu", "-c", match[1]], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            destination = Path(env["PORT_DIR"])
            self.assertTrue((destination / "third_party/example/fixture.txt").is_file())
            self.assertEqual(git("-C", destination / "third_party/example", "rev-parse", "HEAD"),
                             git("-C", upstream, "rev-parse", "HEAD"))
            self.assertEqual(git("-C", destination, "config", "--get", "core.hooksPath"), ".githooks")
            self.assertEqual(git("-C", env["CAPTURE_DIR"], "rev-parse", "HEAD"),
                             git("-C", private, "rev-parse", "HEAD"))
            marker = destination / "do-not-overwrite.txt"
            marker.write_text("user work")
            retry = subprocess.run(["bash", "-eu", "-c", match[1]], env=env,
                                   capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(marker.read_text(), "user work")


if __name__ == "__main__":
    unittest.main()
