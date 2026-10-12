"""Distinct 300-epoch ImageFolder recipe and portable continuation."""
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_idv2_scaling as existing


def folder(root):
    from PIL import Image
    for label in ("zebra", "ant"):
        (root / label).mkdir(parents=True)
        for name in ("b.png", "a.png"):
            Image.new("RGB", (32, 32), (31, 45, 93)).save(root / label / name)


def config(root):
    return dict(existing.tiny_config(root), profile="in1k300_vitl",
                model="vit_large_patch16_224")


@existing.base.needs_deps
class FolderTests(unittest.TestCase):
    def setUp(self):
        self.reader = existing.base.load("imagefolder_scaling", existing.base.METHOD / "data/imagefolder_scaling.py")

    def test_order_labels_exact_basename_exclusions_and_readonly(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder(root / "train"); (root / "val").mkdir()
            (root / "val/b.png").touch()
            # A same stem with a different extension must remain in the IN-1K recipe.
            (root / "val/a.jpg").touch()
            Image.new("RGB", (32, 32)).save(root / "train/ant/ILSVRC2012_val_1.png")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            out = root / "membership"
            meta = self.reader.freeze(root / "train", root / "val", out)
            manifest = out / "TRAIN_MANIFEST.jsonl"
            rows = [json.loads(x) for x in manifest.read_text().splitlines()]
            self.assertEqual(rows, [{"instance_id": i, "relpath": f"{name}/a.png", "class_index": i, "class_name": name}
                                    for i, name in enumerate(("ant", "zebra"))])
            self.assertEqual((meta["n_train"], meta["n_excluded"], meta["n_classes"]), (2, 3, 2))
            self.assertEqual(meta["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertFalse(meta["content_overlap_checked"])
            data = self.reader.FrozenImageFolder(manifest, root / "train", lambda im: im.getpixel((0, 0)))
            self.assertEqual(len(data), 2)
            self.assertEqual(data[1], ((31, 45, 93), 1, 1))
            for p, value in before.items():
                self.assertEqual((root / p).read_bytes(), value)
            with self.assertRaises(FileExistsError):
                self.reader.freeze(root / "train", root / "val", out)

    def test_invalid_inputs_are_not_substituted_or_allowed_to_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder(root / "train"); (root / "val").mkdir()
            with self.assertRaises(FileNotFoundError):
                self.reader.freeze(root / "train", root / "absent", root / "out")
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.reader.freeze(root / "train", root / "val", root / "train/out")
            self.reader.freeze(root / "train", root / "val", root / "good")
            manifest = root / "good/TRAIN_MANIFEST.jsonl"
            rows = [json.loads(x) for x in manifest.read_text().splitlines()]
            data = self.reader.FrozenImageFolder(manifest, root / "train", lambda im: im)
            (root / "train/ant/a.png").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "unreadable"):
                data[0]
            for update in ({"instance_id": 2}, {"instance_id": True}, {"instance_id": False}, {"relpath": None}, {"relpath": ""}, {"relpath": "../escape.png"},
                           {"relpath": "/absolute.png"}, {"class_index": -1}, {"class_index": True}):
                bad = root / "bad.jsonl"; bad.write_text(json.dumps(dict(rows[0], **update)) + "\n")
                with self.subTest(update=update), self.assertRaises(ValueError):
                    self.reader.FrozenImageFolder(bad, root / "train", lambda im: im)
            bad.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                self.reader.FrozenImageFolder(bad, root / "train", lambda im: im)
            (root / "train/ant/a.png").unlink()
            (root / "train/ant/a.png").symlink_to(root / "val/a.png")
            with self.assertRaisesRegex(ValueError, "escape"):
                data[0]
            for name in ("a.png", "b.png"):
                (root / "val" / name).touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                self.reader.freeze(root / "train", root / "val", root / "empty")


@existing.base.needs_deps
@existing.base.needs_timm
class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.mod = existing.base.load("idv2_scaling", existing.base.METHOD / "idv2_scaling.py")

    def test_explicit_recipe_and_warmup_clock(self):
        cfg = config(Path("unused"))
        self.mod.validate(dict(cfg, epochs=300))
        for update in ({"profile": "unknown"}, {"epochs": 301}, {"model": "vit_base_patch16_224"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.mod.validate(dict(cfg, **update))
        for step in (0, 5, 19, 20, 310, 600):
            expected = .0006 * max((step + 1) / 20, 1e-8 / .0006) if step < 20 else .0003 * (1 + math.cos(math.pi * min((step - 20) / 580, 1)))
            self.assertAlmostEqual(self.mod.schedule(step, 600, warmup_steps=20), expected, places=15)

    def test_training_resume_and_100_epoch_export_keep_300_clock(self):
        import torch

        from downstream.contract import verify
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder(root / "source"); (root / "val").mkdir()
            cfg = config(root)
            with mock.patch.object(self.mod, "build_model", side_effect=existing.tiny_model), mock.patch.object(self.mod, "schedule", wraps=self.mod.schedule) as clock:
                for name, epochs, resume in (("full", 2, None), ("part", 1, None),
                    ("resumed", 2, str(root / "part/checkpoint_latest.pth"))):
                    path = root / f"{name}.json"; path.write_text(json.dumps(dict(cfg, epochs=epochs, resume=resume)))
                    self.assertEqual(self.mod.main(["--config", str(path), "--out", str(root / name)]), 0)
                    self.assertEqual(verify(root / name, path, 0), (True, []))
                self.assertEqual([c.args[:2] for c in clock.call_args_list],
                                 [(i, 600) for i in (0, 1, 2, 3, 0, 1, 2, 3)])
                self.assertTrue(all(c.kwargs == {"warmup_steps": 20} for c in clock.call_args_list))
                full = torch.load(root / "full/checkpoint_latest.pth", weights_only=False)
                resumed = torch.load(root / "resumed/checkpoint_latest.pth", weights_only=False)
                self.assertEqual(full["contract"]["schedule_epochs"], 300)
                for field in ("model", "bank"):
                    for key in full[field]:
                        torch.testing.assert_close(full[field][key], resumed[field][key], rtol=0, atol=0)
                for key, state in full["optimizer"]["state"].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)
                synthetic = copy.deepcopy(full); synthetic.update(epoch=98, global_step=198)
                checkpoint = root / "epoch99-fixture.pth"; torch.save(synthetic, checkpoint)
                out = root / "milestone"; out.mkdir()
                self.mod.run(dict(cfg, epochs=100, resume=str(checkpoint)), out)
                self.assertTrue((out / "checkpoint_epoch_100.pth").is_file())
                self.assertTrue((out / "encoder_epoch100.pt").is_file())
                self.assertFalse((out / "checkpoint_epoch_30.pth").exists())
                result = json.loads((out / "results.json").read_text())
                self.assertEqual(result["schedule_epochs"], 300)
                self.assertEqual(result["profile"], "in1k300_vitl")
                self.assertFalse(result["canonical_eligible"])
                for epoch in (200, 300):
                    synthetic.update(epoch=epoch - 2, global_step=(epoch - 1) * 2)
                    torch.save(synthetic, checkpoint)
                    target = root / f"milestone{epoch}"; target.mkdir()
                    self.mod.run(dict(cfg, epochs=epoch, resume=str(checkpoint)), target)
                    self.assertTrue((target / f"checkpoint_epoch_{epoch}.pth").is_file())
                    self.assertTrue((target / f"encoder_epoch{epoch}.pt").is_file())
                source = root / "bad.pth"
                wrong = copy.deepcopy(full); wrong["contract"]["schedule_epochs"] = 90; torch.save(wrong, source)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    self.mod.run(dict(cfg, epochs=3, resume=str(source)), root / "wrong")


@existing.base.needs_deps
@existing.base.needs_timm
class DistributedTests(unittest.TestCase):
    def test_two_rank_folder_continuation_and_collective_read_failure(self):
        import os
        import socket
        import subprocess

        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder(root / "source"); (root / "val").mkdir()
            Image.new("RGB", (32, 32)).save(root / "source/zebra/c.png")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
            proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
                f"--master-port={port}", "--nproc-per-node=2", str(Path(existing.__file__).resolve()), str(root), "in1k"],
                env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(existing.base.ROOT)),
                capture_output=True, text=True, timeout=100, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            result = json.loads((root / "full/results.json").read_text())
            self.assertEqual((result["profile"], result["schedule_epochs"]), ("in1k300_vitl", 300))
            self.assertEqual(result["global_batch_size"], 4)
            self.assertEqual(result["optimizer_steps_per_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
