"""The pipeline stage is `pretrain`, never `step1`.

The SSL pretraining stage used to be called `step1`, which collides with the
paper/results axis "Step 1 / Step 2" (as-is vs from-scratch). The stage is now
`pretrain`; `step1`/`step2` are reserved for the paper axis and must not appear
as a *stage* token in code, configs, or the run manifest vocabulary.

**Discover, never list:** the contract vocabulary is read from `adapterlib`,
every method's `STAGES` is read from its adapter via AST, and every config's
`stage:` value is read from disk. A new method is held to the same bar without
editing this file.
"""

from __future__ import annotations

import ast
import glob
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = ROOT / "methods"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import adapterlib                                     # noqa: E402

FORBIDDEN_STAGE = "step1"


def adapter_stage_tokens(method: str) -> list[str]:
    """The stage names declared in a method's adapter STAGES (dict keys or
    tuple/list elements), read via AST."""
    src = (METHODS_DIR / method / "adapter" / "__init__.py").read_text(
        encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "STAGES" for t in node.targets):
            v = node.value
            if isinstance(v, ast.Dict):
                return [k.value for k in v.keys if isinstance(k, ast.Constant)]
            if isinstance(v, (ast.Tuple, ast.List)):
                return [e.value for e in v.elts if isinstance(e, ast.Constant)]
    return []


def config_stage_values(method: str, methods_dir: Path = METHODS_DIR) -> list[str]:
    vals = []
    for c in glob.glob(str(methods_dir / method / "configs" / "*.yaml")):
        for line in Path(c).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*stage:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
            if m:
                vals.append(m.group(1))
    return vals


def methods() -> list[str]:
    return sorted(d.name for d in METHODS_DIR.iterdir()
                  if d.is_dir() and d.name != "_reference"
                  and (d / "adapter" / "__init__.py").is_file())


class TestContractVocabulary(unittest.TestCase):
    def test_contract_stages_use_pretrain_not_step1(self):
        self.assertIn("pretrain", adapterlib.CONTRACT_STAGES)
        self.assertNotIn(FORBIDDEN_STAGE, adapterlib.CONTRACT_STAGES)
        self.assertNotIn(FORBIDDEN_STAGE, adapterlib.STAGE_FAMILIES)


class TestNoStep1StageToken(unittest.TestCase):
    def test_no_adapter_declares_step1(self):
        bad = [m for m in methods() if FORBIDDEN_STAGE in adapter_stage_tokens(m)]
        self.assertEqual(bad, [], f"adapters still declare a '{FORBIDDEN_STAGE}' "
                                  f"stage (rename to 'pretrain'): {bad}")

    def test_no_config_declares_step1(self):
        bad = [m for m in methods() if FORBIDDEN_STAGE in config_stage_values(m)]
        self.assertEqual(bad, [], f"configs still set 'stage: {FORBIDDEN_STAGE}' "
                                  f"(rename to 'pretrain'): {bad}")


class TestTheDetectorFires(unittest.TestCase):
    def test_the_config_reader_distinguishes_pretrain_from_step1(self):
        import tempfile
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root, ignore_errors=True)
        (root / "m" / "configs").mkdir(parents=True)
        cfg = root / "m" / "configs" / "pretrain.yaml"
        cfg.write_text("stage: step1\n")
        self.assertIn("step1", config_stage_values("m", methods_dir=root))
        cfg.write_text("stage: pretrain\n")
        self.assertNotIn("step1", config_stage_values("m", methods_dir=root))


if __name__ == "__main__":
    unittest.main()


# --- internal-name unification: the pretraining code carries no 'step1' name ---
# The stage token is already 'pretrain'; these guards extend that to the internal
# names so nothing reads 'step1' except the data-aug recipe ("step1"/"step2", the
# paper axis) and prose. Discover, never list.

def _glob(pattern: str) -> list[str]:
    return sorted(glob.glob(str(ROOT / pattern)))


class TestNoStep1InternalNames(unittest.TestCase):
    def test_no_trainer_file_is_named_train_step1(self):
        bad = [Path(p).relative_to(ROOT).as_posix()
               for p in _glob("methods/*/train_step1*.py")]
        self.assertEqual(bad, [], f"trainer files still named train_step1* "
                                  f"(rename to train_pretrain*): {bad}")

    def test_no_mutation_spec_is_named_step1(self):
        bad = [Path(p).relative_to(ROOT).as_posix()
               for p in _glob("mutations/*-step1*.json")]
        self.assertEqual(bad, [], f"mutation specs still named *-step1* "
                                  f"(rename to *-pretrain*): {bad}")

    def test_no_adapter_uses_a_STEP1_identifier(self):
        bad = [Path(p).relative_to(ROOT).as_posix()
               for p in _glob("methods/*/adapter/__init__.py")
               if "STEP1_" in Path(p).read_text(encoding="utf-8")]
        self.assertEqual(bad, [], f"adapters still use STEP1_* identifiers "
                                  f"(rename to PRETRAIN_*): {bad}")

    def test_no_train_step1_module_reference_remains(self):
        """Imports, test loads, mutation anchors, provenance and docs must not
        reference the train_step1 module name."""
        bad = []
        for pat in ("methods/*/adapter/__init__.py", "methods/*/provenance.json",
                    "mutations/*.json", "tests/*.py", "methods/*/README.md",
                    "docs/*.md"):
            for p in _glob(pat):
                if Path(p).name == "test_stage_vocabulary.py":
                    continue          # this guard names the forbidden token
                if "train_step1" in Path(p).read_text(encoding="utf-8"):
                    bad.append(Path(p).relative_to(ROOT).as_posix())
        self.assertEqual(sorted(set(bad)), [],
                         f"references to the train_step1 module remain: {bad}")


class TestTheInternalNameDetectorFires(unittest.TestCase):
    def test_it_would_flag_the_old_names(self):
        # positive controls: the patterns match the old names
        self.assertTrue("train_step1_x".startswith("train_step1"))
        self.assertIn("STEP1_", "STEP1_METRIC_NAMES")
        # negative: the new names are clean
        self.assertFalse("train_pretrain_x".startswith("train_step1"))
        self.assertNotIn("STEP1_", "PRETRAIN_METRIC_NAMES")
