"""One data-root convention for every method: `DATA_ROOT` is the dataset ROOT.

Methods used to disagree about what `DATA_ROOT` pointed at -- some wanted the
ImageNet root and joined `train/` themselves, others wanted the `train`
directory directly, and the same method could differ between step 1 and linear
evaluation. Passing the wrong one produced torchvision's opaque "Found no valid
file for the classes ..." error. The convention is now uniform:

    DATA_ROOT is the dataset root (it contains train/, and val/ for linear
    evaluation); a stage reads its split from a subdirectory (pretraining reads
    train/).

Resolution lives in one place -- `adapterlib.dataset_split_dir` -- and this test
keeps every step-1 config's documented contract uniform so it cannot drift as
methods are added. **Discover, never list:** it scans the step-1 configs and
lets a config *self-declare* an inherent exception (MNIST, or a recursive image
search) rather than hard-coding which methods are special.
"""

from __future__ import annotations

import glob
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"

# The one line every step-1 config carries to document the convention. Kept as
# an exact string so the docs stay uniform, not merely "mentions train".
CANONICAL_LINE = ("#   --set DATA_ROOT=<path>   the dataset root; pretraining "
                  "reads its train/ subdirectory")

# An inherent exception is one the config *declares*, so a new special method
# announces itself rather than being added to a list here. The two that remain:
# 02_vae trains on MNIST, and mar reads pre-encoded cached VAE latents (not an
# ImageNet image folder at all).
EXCEPTION_MARKERS = ("MNIST", "cached latents")


def step1_configs() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(
        str(METHODS_DIR / "*" / "configs" / "pretrain*.yaml")))


def declares_convention(text: str) -> bool:
    return CANONICAL_LINE in text


def declares_exception(text: str) -> bool:
    return any(m in text for m in EXCEPTION_MARKERS)


def offenders(configs: list[Path]) -> list[str]:
    """Step-1 configs that neither declare the convention nor a self-declared
    exception."""
    bad = []
    for c in configs:
        text = c.read_text(encoding="utf-8")
        if not declares_convention(text) and not declares_exception(text):
            try:
                bad.append(str(c.relative_to(ROOT)))
            except ValueError:
                bad.append(str(c))
    return bad


class TestEveryStep1DocumentsTheConvention(unittest.TestCase):
    def test_there_are_step1_configs_to_check(self):
        self.assertGreater(len(step1_configs()), 20,
                           "the scan found almost nothing -- it is not looking")

    def test_every_step1_config_declares_the_data_root_convention(self):
        bad = offenders(step1_configs())
        self.assertEqual(
            bad, [],
            "these step-1 configs do not document the unified DATA_ROOT "
            f"convention (and declare no exception): {bad}")


class TestTheDetectorFires(unittest.TestCase):
    """A guard that cannot fail is not a guard."""

    def test_a_config_without_the_line_or_an_exception_is_flagged(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        bad = tmp / "pretrain.yaml"
        bad.write_text("stage: pretrain\n# no data-root documentation here\n",
                       encoding="utf-8")
        self.assertTrue(offenders([bad]), "an undocumented config was not flagged")

    def test_the_canonical_line_silences_it(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        good = tmp / "pretrain.yaml"
        good.write_text(f"stage: pretrain\n{CANONICAL_LINE}\n", encoding="utf-8")
        self.assertEqual(offenders([good]), [])

    def test_a_self_declared_exception_silences_it(self):
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, ignore_errors=True)
        for marker in EXCEPTION_MARKERS:
            ex = tmp / f"{marker[:4]}.yaml"
            ex.write_text(f"stage: pretrain\n# uses {marker}\n", encoding="utf-8")
            self.assertEqual(offenders([ex]), [])


if __name__ == "__main__":
    unittest.main()
