#!/usr/bin/env python3
"""ABCI 基盤の振る舞いを固定する。

この基盤は**任意の追加物**で、コアはこれを知らない
（分離そのものは `tests/test_platform_isolation.py` が守る）。
ここでは中身の正しさを見る。

**投入コマンドが無い環境でも走るテストにする。** 実際に投入したら
テストにならないし、共有計算機にジョブを撒くことになる。
"""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import platforms                                    # noqa: E402

abci = platforms.load_backend("abci")
JobSpec = platforms.JobSpec


def spec(**over) -> JobSpec:
    base = dict(name="job", command=["python3", "-m", "adapter"],
                env_name="py3.10_x", gpus=8, hours=24)
    base.update(over)
    return JobSpec(**base)


class TestResourceTranslation(unittest.TestCase):
    """必要量から資源への翻訳。**ここにしか翻訳表を置かない。**"""

    def test_known_amounts_translate(self):
        for gpus in (0, 1, 8):
            self.assertTrue(abci.resource_type(gpus))

    def test_unknown_amount_is_refused_not_rounded(self):
        """**勝手に丸めない。** 意図しない資源で走ると結果が変わる。"""
        with self.assertRaises(ValueError) as e:
            abci.resource_type(3)
        self.assertIn("3", str(e.exception))

    def test_the_error_says_what_is_available(self):
        with self.assertRaises(ValueError) as e:
            abci.resource_type(99)
        self.assertIn("8", str(e.exception), "使える値を教えていない")


class TestScriptRendering(unittest.TestCase):
    """副作用を持たないので単体で検証できる。"""

    def test_required_directives_are_present(self):
        s = abci.render_script(spec(), group="grp")
        for frag in ("#!/bin/bash", "walltime=24:00:00", "grp", "job"):
            self.assertIn(frag, s, f"{frag} が無い")

    def test_command_is_included(self):
        self.assertIn("python3 -m adapter",
                      abci.render_script(spec(), group="g"))

    def test_environment_is_exported_deterministically(self):
        """順序が実行ごとに変わると、生成物が毎回変わって差分が読めない。"""
        s1 = abci.render_script(spec(env={"B": "2", "A": "1"}), group="g")
        s2 = abci.render_script(spec(env={"A": "1", "B": "2"}), group="g")
        self.assertEqual(s1, s2, "環境変数の並びが安定していない")
        self.assertLess(s1.index('export A='), s1.index('export B='))

    def test_workdir_is_used_when_given(self):
        self.assertIn('cd "/w"',
                      abci.render_script(spec(workdir="/w"), group="g"))

    def test_unknown_gpu_count_propagates_as_an_error(self):
        with self.assertRaises(ValueError):
            abci.render_script(spec(gpus=3), group="g")


class TestAvailability(unittest.TestCase):
    def test_available_when_the_submit_command_exists(self):
        with mock.patch.object(abci.shutil, "which", return_value="/x/qsub"):
            self.assertTrue(abci.Backend("g").is_available())

    def test_unavailable_when_it_does_not(self):
        """**推測せず実際に確かめる。**"""
        with mock.patch.object(abci.shutil, "which", return_value=None):
            self.assertFalse(abci.Backend("g").is_available())


class TestSubmit(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="abcitest-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))

    def _submit(self, returncode=0, stdout="12345.pbs\n", stderr=""):
        b = abci.Backend("grp", script_dir=self.tmp)
        fake = types.SimpleNamespace(returncode=returncode, stdout=stdout,
                                     stderr=stderr)
        with mock.patch.object(abci.shutil, "which", return_value="/x/qsub"), \
             mock.patch.object(abci.subprocess, "run", return_value=fake):
            return b.submit(spec())

    def test_exit_status_is_unknown_not_zero(self):
        """**投入しただけでは結果は分からない。**

        0 は「成功した」という意味である。不明を成功と偽ると、
        呼び出し側は失敗したジョブを成功として扱う。
        変異テストでこの経路が未検証だと分かって追加した。
        """
        r = self._submit()
        self.assertIsNone(r.exit_status,
                          "投入しただけなのに終了コードを名乗っている")

    def test_job_id_comes_from_the_submitter(self):
        self.assertEqual(self._submit().job_id, "12345.pbs")

    def test_script_is_written(self):
        self._submit()
        self.assertTrue((self.tmp / "job.sh").is_file())

    def test_submission_failure_is_loud(self):
        with self.assertRaises(RuntimeError) as e:
            self._submit(returncode=1, stderr="だめ")
        self.assertIn("だめ", str(e.exception))

    def test_refuses_when_unavailable_and_says_what_to_do(self):
        b = abci.Backend("grp", script_dir=self.tmp)
        with mock.patch.object(abci.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError) as e:
                b.submit(spec())
        self.assertIn("local", str(e.exception),
                      "代わりに何を使えばよいか教えていない")

    def test_nothing_is_submitted_when_unavailable(self):
        b = abci.Backend("grp", script_dir=self.tmp)
        with mock.patch.object(abci.shutil, "which", return_value=None), \
             mock.patch.object(abci.subprocess, "run") as run:
            with self.assertRaises(RuntimeError):
                b.submit(spec())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
