"""Public names, historical reader compatibility and publication order."""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_readme_methods_table import _table_dirs
from test_aggregate_seeds import _run_dir

ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "methods"
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


def invoke(tool, *args):
    return subprocess.run([sys.executable, str(ROOT / "bin" / tool), *args],
                          capture_output=True, text=True)


class TestMethodNaming(unittest.TestCase):
    def test_physical_names_and_exact_table_order(self):
        names = [old.split("_", 1)[1] for old in sorted(FROZEN_NUMBERED)]
        disk = {p.name for p in METHODS.iterdir() if p.is_dir() and not p.name.startswith("_")}
        self.assertEqual(len(names), 38)
        self.assertGreaterEqual(len(disk), 56)
        self.assertTrue({"dinov2", "vjepa2", "vjepa2_1", "vjepa2_ac"} <= disk)
        self.assertFalse([n for n in disk if re.match(r"^\d+_", n)])
        self.assertEqual(_table_dirs(), names + sorted(disk - set(names)))
        self.assertEqual(set(_table_dirs()), disk)

    def test_matrix_selection_and_new_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "matrix.json"
            proc = invoke("matrix-run.py", "--method", "dinov2", "--out", out)
            # No dataset was supplied, so the selected method records a failed cell.
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual([c["method"] for c in json.loads(out.read_text())["cells"]], ["dinov2"])
            for invalid in ("28_dinov2", "99_dinov2"):
                self.assertEqual(invoke("matrix-run.py", "--method", invalid).returncode, 2)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "requires PyYAML")
    def test_config_resolves_with_historical_encoder_path_verbatim(self):
        encoder = "/archive/02_vae/encoder.pt"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "config.json"
            args = ("--set", "DATA_ROOT=/images", "--set", "ENCODER=" + encoder, "--out", out)
            config = METHODS / "vae" / "configs" / "linear_eval.yaml"
            proc = invoke("resolve-config.py", "--config", config, *args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(out.read_text())["encoder"], encoder)
            old = METHODS / "02_vae" / "configs" / config.name
            self.assertNotEqual(invoke("resolve-config.py", "--config", old, *args).returncode, 0)

    def test_saved_manifests_mix_only_exact_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [_run_dir(root, i, top1=70, top5=90, method=m)
                    for i, m in enumerate(("28_dinov2", "dinov2", "28_dinov2"))]
            files = [d / "run_manifest.json" for d in runs]
            before = [p.read_bytes() for p in files]
            args = [x for d in runs for x in ("--run", d)]
            proc = invoke("aggregate-seeds.py", *args, "--out", root / "out")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((root / "out" / "aggregate.json").read_text())
            self.assertEqual(result["method"], "dinov2")
            self.assertEqual([p.read_bytes() for p in files], before)
            _run_dir(root, 0, top1=70, top5=90, method="99_dinov2")
            self.assertNotEqual(invoke("aggregate-seeds.py", *args, "--out", root / "bad").returncode, 0)

    def test_archived_encoders_and_venvs_keep_their_actual_paths(self):
        from test_extract_features import tool

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = root / "methods"
            provider = methods / "vae" / "feature_provider.py"
            provider.parent.mkdir(parents=True)
            provider.touch()
            saved = root / "02_vae" / "encoder.pt"
            saved.parent.mkdir()
            saved.write_bytes(b"saved input")
            extractor = tool()
            for mapping in ({}, {"02_vae": str(saved)}, {"vae": str(saved)}):
                rows = extractor.plan(methods, mapping, root)
                self.assertEqual([(r["method"], r["encoder"]) for r in rows], [("vae", str(saved))])
            with self.assertRaises(ValueError):
                extractor.plan(methods, {"vae": str(saved), "02_vae": str(saved)})
            python = saved.parent / "bin" / "python"
            python.parent.mkdir()
            python.touch()
            self.assertEqual(extractor.python_for("vae", root), str(python))
            self.assertEqual(saved.read_bytes(), b"saved input")

    def test_archived_matrix_paths_stay_readable(self):
        from test_matrix_audit import fabricate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = json.loads((METHODS / "dinov2" / "real_run_smoke.json").read_text())
            matrix = fabricate([("28_dinov2", spec)], root)
            before = matrix.read_bytes()
            out = root / "audit.json"
            proc = invoke("matrix-audit.py", "--matrix", matrix, "--out", out)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(json.loads(out.read_text())["cells"][0]["cell"], "dinov2/linear_eval")
            self.assertEqual(matrix.read_bytes(), before)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "scipy")),
                         "requires representation analysis dependencies")
    def test_representation_files_keep_paths_and_match_public_references(self):
        from downstream.representation_analysis import run
        from test_representation_analysis_cli import AnalysisCLI

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AnalysisCLI().fixture(root)
            config["models"][0]["id"] = "28_dinov2"
            config["image_reference"] = "dinov2"
            report = run(config, root / "analysis")
            self.assertEqual(report["references"]["image"], "dinov2")
            self.assertIn("dinov2", report["inputs"])
            self.assertEqual(config["models"][0]["id"], "28_dinov2")
            config["models"][2]["id"] = "dinov2"
            with self.assertRaisesRegex(ValueError, "unique"):
                run(config, root / "duplicate")

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "requires the workflow parser")
    @unittest.skipUnless((ROOT / ".github/workflows/tests.yml").is_file(), "workflow omitted from runtime image")
    def test_ci_discovers_all_locks_in_original_order(self):
        from test_ci import discovery_script, parsed

        proc = subprocess.run([sys.executable, "-c", discovery_script(parsed()["tests.yml"])],
                              cwd=ROOT, env={**os.environ, "GITHUB_EVENT_NAME": "push"},
                              text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        outputs = dict(line.split("=", 1) for line in proc.stdout.splitlines())
        expected = [name for name in _table_dirs() if (METHODS / name / "requirements.lock.txt").is_file()]
        self.assertEqual(json.loads(outputs["methods"]), expected)
        from test_ci import runs_of
        downstream = runs_of(parsed()["tests.yml"], "downstream")
        modules = re.findall(r"\btests\.(test_\w+)\b", downstream)
        self.assertTrue(modules)
        self.assertEqual([m for m in modules if not (ROOT / "tests" / (m + ".py")).is_file()], [])
        representatives = json.loads(outputs["locked_methods"])
        self.assertEqual(representatives, [name for name in expected if name in representatives])
