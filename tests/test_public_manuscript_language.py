"""Check the terminology encountered before opening an adapter or a protocol."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicManuscriptLanguage(unittest.TestCase):
    def test_figure_navigation_introduces_the_three_experimental_settings(self):
        intro = (ROOT / "README.md").read_text().split("| Figure |")[0]
        for term in ("ASIS", "CTRL", "Unified"):
            self.assertRegex(intro, rf"\b{term}\b")

    def test_method_index_uses_named_operations_and_settings(self):
        guide = (ROOT / "OPERATING_GUIDE.md").read_text()
        methods = guide.split("\n## Methods\n", 1)[1].split("\n## ", 1)[0]
        offending_rows = [
            line[:95] for line in methods.splitlines()
            if re.search(r"\bstep[ -]?[1-4]\b", line, re.I)
        ]
        self.assertEqual(offending_rows, [])
        for term in ("ASIS", "CTRL", "Unified"):
            self.assertIn(term, methods.split("| Directory |", 1)[0])

    def test_probe_metric_does_not_promise_unconditional_comparability(self):
        lines = (ROOT / "OPERATING_GUIDE.md").read_text().splitlines()
        row, = (line for line in lines if line.startswith("| `linear_probe` |"))
        condition = row.strip("|").split("|")[-1]
        for factor in ("dataset", "metric", "protocol"):
            self.assertIn(factor, condition)

    def test_figure_index_distinguishes_lora_and_the_asis_ctrl_comparison(self):
        rows = (ROOT / "README.md").read_text().splitlines()
        figure3, = (line for line in rows if line.startswith("| Figure 3 |"))
        self.assertIn("LoRA", figure3.split("|")[2])
        figure4, = (line for line in rows if line.startswith("| Figure 4 |"))
        for setting in ("ASIS", "CTRL"):
            self.assertIn(setting, figure4.split("|")[2])

    def test_public_guide_points_to_the_figure_scope_and_the_adapter_contract(self):
        text = (ROOT / "OPERATING_GUIDE.md").read_text()
        for dest in ("docs/CVPR_EXTENDED_COVERAGE.md", "bin/contract-test.py"):
            self.assertIn("](" + dest + ")", text)
            self.assertTrue((ROOT / dest).is_file())
