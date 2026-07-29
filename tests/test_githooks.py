#!/usr/bin/env python3
"""pre-commit フックの仕様を定義するテスト。

**文書では守られなかった。** 2026-07-29、テストが EXIT=1 のまま
commit して push した。ルールは CLAUDE.md・memory・DESIGN の3か所に
書いてあったが、それでも破られた。方針は機構にしないと守られない
（このプロジェクトが繰り返し学んできたこと）。
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK = ROOT / ".githooks" / "pre-commit"


class TestHookExists(unittest.TestCase):
    def test_hook_is_present(self):
        self.assertTrue(HOOK.is_file(), "pre-commit フックが無い")

    def test_hook_is_executable(self):
        """実行ビットが無いと git は黙って無視する。"""
        self.assertTrue(HOOK.stat().st_mode & stat.S_IXUSR,
                        "実行ビットが立っていない。git は黙って無視する")

    def test_hook_runs_the_suite(self):
        self.assertIn("tests/run-tests.sh", HOOK.read_text(encoding="utf-8"))


class TestHookBehaviour(unittest.TestCase):
    """偽のテストスクリプトを置いて、フックの分岐を実際に走らせる。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hooktest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q", str(self.tmp)],
                       check=True, capture_output=True)
        (self.tmp / "tests").mkdir()
        shutil.copy(HOOK, self.tmp / "pre-commit")
        os.chmod(self.tmp / "pre-commit", 0o755)

    def fake_suite(self, exit_code: int) -> None:
        p = self.tmp / "tests" / "run-tests.sh"
        p.write_text(f"#!/usr/bin/env bash\necho 偽のテスト\nexit {exit_code}\n")
        os.chmod(p, 0o755)

    def run_hook(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.tmp / "pre-commit")],
                              cwd=self.tmp, capture_output=True, text=True)

    def test_green_suite_allows_the_commit(self):
        self.fake_suite(0)
        self.assertEqual(self.run_hook().returncode, 0)

    def test_red_suite_blocks_the_commit(self):
        self.fake_suite(1)
        r = self.run_hook()
        self.assertNotEqual(r.returncode, 0, "赤なのにコミットを通している")
        self.assertIn("コミットを中止", r.stderr)

    def test_git_environment_is_cleared_before_running_the_suite(self):
        """**git はフックに GIT_DIR などを渡す。**

        残したままだと、テストが起動する git が fixture ではなく
        このリポジトリを触り、git add が exit 128 で落ちる。
        2026-07-29 に実際に 120 件エラーになった。
        """
        p = self.tmp / "tests" / "run-tests.sh"
        p.write_text(
            "#!/usr/bin/env bash\n"
            'for v in GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE; do\n'
            '  if [ -n "${!v:-}" ]; then echo "$v が残っている"; exit 1; fi\n'
            "done\nexit 0\n")
        os.chmod(p, 0o755)
        env = {**os.environ, "GIT_DIR": "/somewhere/.git",
               "GIT_WORK_TREE": "/somewhere", "GIT_INDEX_FILE": "/tmp/idx"}
        r = subprocess.run(["bash", str(self.tmp / "pre-commit")],
                           cwd=self.tmp, env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0,
                         f"git の環境変数を落としていない: {r.stderr}")

    def test_failure_output_is_shown(self):
        """止めた理由が見えないと、利用者は --no-verify に逃げる。"""
        self.fake_suite(1)
        self.assertIn("偽のテスト", self.run_hook().stderr)


if __name__ == "__main__":
    unittest.main()
