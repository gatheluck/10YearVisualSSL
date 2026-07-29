#!/usr/bin/env python3
"""すべてのツールがテストの対象になっていることを機構で担保する。

survey-live.py は TDD 方針を採用する前に書かれ、そのまま
**テスト0件で運用に入っていた**（2026-07-29 に発覚）。
方針を文書に書くだけでは守られない。取り残しを機械が見つける。

「テストがあるか」は、tests/ 配下のどれかが
そのファイル名に言及しているかで判定する。
importlib で読み込む方式（load("x", "x.py")）と、
CLI を subprocess で叩く方式の両方を拾える。
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
TESTS = ROOT / "tests"
SELF = Path(__file__).name


def _corpus() -> dict[str, str]:
    """自分自身は除く。除かないと、下のツール一覧が自己満足してしまう。"""
    out: dict[str, str] = {}
    for p in sorted(TESTS.glob("test_*.py")):
        if p.name != SELF:
            out[p.name] = p.read_text(encoding="utf-8")
    for extra in ("run-tests.sh",):
        p = TESTS / extra
        if p.exists():
            out[extra] = p.read_text(encoding="utf-8")
    return out


class TestEveryToolIsUnderTest(unittest.TestCase):
    def test_no_tool_is_left_untested(self):
        corpus = _corpus()
        tools = sorted(p.name for p in BIN.glob("*.py"))
        tools += sorted(p.name for p in BIN.glob("*.sh"))
        self.assertTrue(tools, "bin/ にツールが見つからない")
        missing = [t for t in tools
                   if not any(t in body for body in corpus.values())]
        self.assertEqual(
            missing, [],
            "テストから一度も参照されていないツールがある。\n"
            "  tests/test_<name>.py を追加するか、e2e に組み込むこと:\n"
            + "".join(f"    - bin/{m}\n" for m in missing))

    def test_guard_itself_does_not_count_as_coverage(self):
        """このファイルを除外し忘れると、ガードが常に通ってしまう。"""
        self.assertNotIn(SELF, _corpus(),
                         "ガード自身が探索対象に入っている")


if __name__ == "__main__":
    unittest.main()
