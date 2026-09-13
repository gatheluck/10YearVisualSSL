"""The method-naming convention, mechanised.

Decision (see docs/METHOD_NAMING.md): the numeric prefix on a method directory
(`01_context_prediction` ... `38_clip`) is the *historical chronology* axis of the
"ten years of visual SSL", kept aligned with the numbered, fixed Capture
design-of-record. That roster is **closed**. Every method added afterwards
(Step 3 and beyond) is **unnumbered** (`eva02`, `sam3`, `cosmos3_super`, ...).

A policy in a document does not hold; this test is the machinery. It refuses a
*new* numbered method directory, so the "new methods are unnumbered" rule cannot
silently rot. It deliberately does NOT forbid the existing numbers (removing them
is a separate, deliberate migration; a physical rename, if ever, is deferred to
the publication/audit move and would edit FROZEN_NUMBERED here in the same change).

The guard is proven non-vacuous by a negative control: a decoy new-numbered name
must be judged as violating.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"

# The closed historical roster — the only numbered method directories allowed.
# Do not append to this to add a method: new methods are unnumbered. This set
# shrinks only if a numbered method is deliberately migrated to an unnumbered
# name (a separate change that also renames on disk, in docs, and in the bucket).
FROZEN_NUMBERED = frozenset({
    "01_context_prediction", "02_vae", "03_colorization", "04_context_encoder",
    "05_jigsaw_puzzle", "06_rotation_prediction", "07_deepcluster",
    "08_split_brain", "09_jigsaw_puzzle_pp", "10_inst_disc", "11_cpc", "12_cmc",
    "13_mocov1", "14_simclrv1", "15_mocov2", "16_simclrv2", "17_swav", "18_sela",
    "19_byol", "20_simsiam", "21_barlow_twins", "22_mocov3", "23_dino",
    "24_beit", "25_mae", "26_simmim", "27_ibot", "28_dinov2", "29_ijepa",
    "30_aim", "31_dinov3", "32_nepa", "33_pirl", "34_msn", "35_vjepa",
    "36_franca", "37_lejepa", "38_clip",
})

_NUMBERED = re.compile(r"^\d+_")


def _is_numbered(name: str) -> bool:
    return bool(_NUMBERED.match(name))


def _numbered_method_dirs() -> "list[str]":
    """Discover, never list: every directory under methods/ whose name is
    numeric-prefixed. Discovery is what makes a newly-added numbered method
    show up here without anyone remembering to register it."""
    return sorted(
        p.name for p in METHODS_DIR.iterdir()
        if p.is_dir() and _is_numbered(p.name)
    )


class TestMethodNaming(unittest.TestCase):
    def test_no_new_numbered_method_directories(self):
        """Every numbered method on disk is in the frozen roster: no new one
        was added. New methods must be unnumbered."""
        offenders = [d for d in _numbered_method_dirs() if d not in FROZEN_NUMBERED]
        self.assertEqual(
            offenders, [],
            "New numbered method directory found: {}. New methods are "
            "unnumbered (see docs/METHOD_NAMING.md); if this is a deliberate "
            "rename of an existing method, update FROZEN_NUMBERED in the same "
            "change.".format(offenders),
        )

    def test_the_frozen_roster_still_exists_on_disk(self):
        """Positive control: the frozen roster is not stale — each entry is a
        real directory. (Catches a rename/removal that forgot to update the
        roster, which would otherwise silently weaken the negative check.)"""
        missing = sorted(n for n in FROZEN_NUMBERED if not (METHODS_DIR / n).is_dir())
        self.assertEqual(
            missing, [],
            "FROZEN_NUMBERED names no directory on disk: {}. If a method was "
            "renamed/removed, update the roster.".format(missing),
        )

    def test_at_least_one_unnumbered_method_exists(self):
        """The unnumbered convention is live, not hypothetical: some method
        directory is unnumbered."""
        unnumbered = [
            p.name for p in METHODS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("_") and not _is_numbered(p.name)
        ]
        self.assertTrue(
            unnumbered,
            "No unnumbered method directory exists; the 'new methods are "
            "unnumbered' convention has no live example.",
        )

    def test_the_guard_fires_on_a_decoy_numbered_name(self):
        """Negative control: prove the guard is not vacuous. A decoy new-numbered
        name is not in the frozen roster and would be flagged by the same rule."""
        decoy = "39_decoy_method"
        self.assertTrue(_is_numbered(decoy))
        self.assertNotIn(decoy, FROZEN_NUMBERED)


if __name__ == "__main__":
    unittest.main()
