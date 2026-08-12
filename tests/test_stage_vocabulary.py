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
