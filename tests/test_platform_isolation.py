#!/usr/bin/env python3
"""ABCI への依存が `platforms/abci/` の外へ漏れないことを機構で担保する。

**ABCI 上での実行はオプショナル。** コアは ABCI を前提としない。
ABCI 対応は疎結合なモジュールとして分離し、そこからだけ呼ばれる。

方針を文書に書くだけでは守られない。今日の実例（Capture 側 DESIGN §5.26）:
厳格な TDD を3か所に書いても破られた。**機構で止める。**

このテストが守る性質:

- **ABCI 固有の語彙は `platforms/abci/` の中にしか現れない。**
  `#PBS`、`qsub`、`/groups/`、`rt_HF`、`module load` など
- **ABCI が無くても動く経路が存在する。** `platforms/local/` が必ずある
- **`platforms/abci` を `platforms/` の外から import しない。**
  import した時点で、コアが ABCI を知っていることになる
- **共通の界面に ABCI の語彙を入れない。** 界面が汚れたら疎結合は崩れる
"""

from __future__ import annotations

import re
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLATFORMS = ROOT / "platforms"
ABCI_DIR = PLATFORMS / "abci"

# ABCI 固有の語彙。実データ（Capture 済みのジョブ投入スクリプト）で
# 観測したものに基づく。推測で広げない。増えたらここに足す。
ABCI_MARKERS = (
    "#PBS", "qsub", "rt_HF", "rt_HG", "rt_HC", "gag51492",
    "/groups/", "module load", "abci",
)

# 走査対象。テスト自身と、ここを説明する文書は除く
def _scan_targets() -> list[Path]:
    out: list[Path] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        parts = rel.parts
        if parts[0] in (".git", "docs", "tests", ".githooks"):
            continue
        if p.suffix not in (".py", ".sh", ".yaml", ".yml", ".toml", ".cfg"):
            continue
        out.append(p)
    return out


class TestAbciVocabularyIsContained(unittest.TestCase):
    def test_markers_appear_only_under_platforms_abci(self):
        leaks: list[str] = []
        for p in _scan_targets():
            rel = p.relative_to(ROOT)
            if rel.parts[:2] == ("platforms", "abci"):
                continue
            text = p.read_text(encoding="utf-8", errors="replace").lower()
            for m in ABCI_MARKERS:
                if m.lower() in text:
                    leaks.append(f"{rel}: {m}")
        self.assertEqual(leaks, [],
                         "ABCI 固有の語彙が platforms/abci/ の外に漏れている:\n"
                         + "\n".join(f"  - {x}" for x in leaks))

    def test_the_scan_actually_looks_at_something(self):
        """走査対象がゼロだと、この検査は常に通ってしまう。"""
        self.assertTrue(_scan_targets(), "走査対象が1つも無い")


class TestAbciIsOptional(unittest.TestCase):
    """ABCI が無くても動く経路が存在すること。"""

    def test_local_platform_exists(self):
        self.assertTrue((PLATFORMS / "local").is_dir(),
                        "platforms/local/ が無い。ABCI が前提になっている")

    def test_base_interface_exists(self):
        self.assertTrue((PLATFORMS / "base.py").is_file(),
                        "共通の界面 platforms/base.py が無い")

    def test_base_interface_is_free_of_abci_vocabulary(self):
        """界面が汚れたら疎結合は崩れる。"""
        text = (PLATFORMS / "base.py").read_text(encoding="utf-8").lower()
        for m in ABCI_MARKERS:
            self.assertNotIn(m.lower(), text, f"界面に {m} が入っている")

    def test_nothing_outside_platforms_imports_the_abci_module(self):
        """import した時点で、コアが ABCI を知っていることになる。"""
        pat = re.compile(r"(?:from|import)\s+[\w.]*platforms\.abci")
        offenders: list[str] = []
        for p in _scan_targets():
            rel = p.relative_to(ROOT)
            if rel.parts[0] == "platforms":
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(rel))
        self.assertEqual(offenders, [],
                         f"platforms/ の外から abci を import している: {offenders}")

    def test_abci_module_exists_but_is_not_required(self):
        """あるが、無くても壊れないこと。"""
        self.assertTrue(ABCI_DIR.is_dir(), "platforms/abci/ が無い")
        base = (PLATFORMS / "base.py").read_text(encoding="utf-8")
        self.assertNotIn("abci", base.lower())


class TestBackendsShareOneInterface(unittest.TestCase):
    """両方が同じ界面を満たすことを、実際に読み込んで確かめる。

    パッケージとして import する。ファイルを個別に読み込むと**同じ
    ファイルが二重に読み込まれてクラスが別物になり**、issubclass が
    偽になる（Capture 側で except が効かなくなったのと同じ型）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

    def test_local_backend_implements_the_interface(self):
        import platforms
        self.assertTrue(issubclass(
            platforms.load_backend("local").Backend, platforms.Backend))

    def test_abci_backend_implements_the_interface(self):
        import platforms
        self.assertTrue(issubclass(
            platforms.load_backend("abci").Backend, platforms.Backend))

    def test_job_spec_is_expressed_in_generic_terms(self):
        """コアは必要量を一般語で言う。資源名への翻訳は platform の仕事。"""
        import platforms
        fields = set(platforms.JobSpec.__dataclass_fields__)
        for f in ("command", "env_name", "gpus", "hours", "name"):
            self.assertIn(f, fields, f"JobSpec に {f} が無い")


class TestBackendResolutionHasNoHardcodedTable(unittest.TestCase):
    """**コアは基盤の名前すら持たない。** 名前は利用者が渡す。"""

    @classmethod
    def setUpClass(cls) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

    def test_backends_are_discovered_not_listed(self):
        import platforms
        self.assertIn("local", platforms.available_backends())

    def test_unknown_backend_says_what_is_available(self):
        """「使えません」だけでは、利用者が次に何をすべきか分からない。"""
        import platforms
        with self.assertRaises(ValueError) as e:
            platforms.load_backend("nonexistent")
        self.assertIn("local", str(e.exception))

    def test_a_newly_added_backend_is_discovered_without_code_changes(self):
        """**挙動で確かめる。** 文字列で判定すると、docstring の使用例まで
        「直書き」と誤判定する（実際に誤判定するテストを書いた）。

        表を持っていれば、ディレクトリを置いても見つからない。
        """
        import platforms
        dummy = PLATFORMS / "_dummy_for_test"
        (dummy).mkdir(exist_ok=True)
        try:
            (dummy / "__init__.py").write_text("", encoding="utf-8")
            (dummy / "backend.py").write_text("", encoding="utf-8")
            self.assertIn("_dummy_for_test", platforms.available_backends(),
                          "置いただけでは見つからない。表を持っている")
        finally:
            shutil.rmtree(dummy, ignore_errors=True)
        self.assertNotIn("_dummy_for_test", platforms.available_backends(),
                         "消しても残っている。実体を見ていない")


if __name__ == "__main__":
    unittest.main()
