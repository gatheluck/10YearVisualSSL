"""Check the paths from package evaluation guidance to the current paper."""
from pathlib import Path
import re
import unittest


DOCS = Path(__file__).resolve().parents[1] / "docs"


class EvaluationGuidance(unittest.TestCase):
    def test_current_evaluation_guide_maps_each_setting_to_its_figure(self):
        guide = (DOCS / "EVALUATION.md").read_text()
        rows = [line.split("|")[1:4] for line in guide.splitlines() if line.startswith("|")]
        for setting, section, figure in (("ASIS", 4, 2), ("CTRL", 5, 4), ("Unified", 6, 3)):
            with self.subTest(setting=setting):
                matches = [row for row in rows if row[0].strip() == setting]
                self.assertEqual(len(matches), 1)
                self.assertIn(f"Section {section}", matches[0][1])
                self.assertIn(f"Figure {figure}", matches[0][1])
                target, = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", matches[0][2])
                self.assertTrue((DOCS / target).is_file())

    def test_active_training_guides_name_the_implementation_in_their_title(self):
        for guide in ("DINOV3_STEP4", "IDV2_COMPONENTS", "STEP1_WEIGHTS",
                      "STEP2_VIT_PORTING", "STEP3_PORTING_PLAN", "PORTING_ROADMAP"):
            with self.subTest(guide=guide):
                title = (DOCS / (guide + ".md")).read_text().splitlines()[0]
                self.assertNotRegex(title, r"(?i)\bstep[ -]?[1-4]\b")

    def test_basic_five_attribution_explains_both_sources(self):
        text = (DOCS / "BASIC5_V1_AUDIT.md").read_text()
        section = text.split("### Run attribution checked against the revised supplement\n", 1)[1].split("\n## ", 1)[0]
        for detail in ("Section 6", "LP/AP", "block 29", "t = 0.1", "K7", "22 cells"):
            self.assertIn(detail, section)

    def test_older_table_numbers_identify_the_source_manuscript(self):
        for path in DOCS.glob("*.md"):
            text = path.read_text()
            if re.search(r"(?i)\btables?\s+(?:3[3-9]|40)\b", text):
                with self.subTest(path=path.name):
                    self.assertRegex(text[:2400], r"(?i)(earlier submission|70-page|2026-09-25 manuscript)")

    def test_porting_instructions_use_descriptive_names_in_running_prose(self):
        for name in ("STEP2_VIT_PORTING", "STEP3_PORTING_PLAN", "PORTING_ROADMAP", "PAPER_REPRODUCTION_GAPS"):
            with self.subTest(name=name):
                text = (DOCS / (name + ".md")).read_text()
                prose = re.sub(r"`[^`\n]*`", "", text)
                self.assertNotRegex(prose, r"(?i)\bstep[ -][1-4]\b")
