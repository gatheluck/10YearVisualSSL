"""Real CUDA/BF16 diagnostic CLI matrix; CPU skips are not GPU evidence.

All weights and data are reduced synthetic fixtures. In particular, Omega's
synthetic LayerScale is 0.1, so the adapter diagnostic is not degenerate.
No published checkpoint or manuscript score is represented by these tests.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = (("imagenet", "lora"), ("ade20k", "lora"), ("ade20k", "dpt"),
          ("ssv2", "lora"), ("coco", "lora"), ("nyuv2", "lora"), ("nyuv2", "dpt"))


def _cuda_available():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available()


def _fixture_module(name):
    # Native upstream packages can shadow the repository's tests package.
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("adjacent fixture module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_checkpoints(root):
    """Reuse reduced provider builders and serialize complete local inputs."""
    import torch

    root = Path(root)
    root.mkdir(parents=True)
    hf = _fixture_module("test_cvpr_hf_bindings").HFBindings()
    native = _fixture_module("test_method_cvpr_native_bindings").NativeBindings()
    result = {}
    for family, kind in (("map_pool", "siglip2_g"), ("register_cls", "dinov3_hf")):
        hf.setUp()
        body = hf.backbone(family)
        path = root / kind
        body.model.save_pretrained(path)
        result[kind] = {"kind": kind, "arch": "fixture", "encoder": str(path), "img_size": 32, "patch_size": 16}
    native.setUp()
    body = native.vjepa()
    path = root / "video.pt"
    torch.save({"ema_encoder": body.encoder.state_dict()}, path)
    result["vjepa2_1"] = {"kind": "vjepa2_1", "arch": "vit_large", "encoder": str(path),
                              "img_size": 384, "patch_size": 16, "embed_dim": 48, "depth": 12, "num_heads": 4}
    native.setUp()
    body = native.omega()
    with torch.no_grad():
        for name, parameter in body.aggregator.named_parameters():
            if name.endswith(("ls1.gamma", "ls2.gamma")):
                parameter.fill_(0.1)
    path = root / "geometry.pt"
    torch.save({"aggregator." + n: p for n, p in body.aggregator.state_dict().items()}, path)
    result["vggt_omega"] = {"kind": "vggt_omega", "arch": "fixture", "encoder": str(path),
                               "img_size": 224, "patch_size": 16, "embed_dim": 64, "depth": 24, "num_heads": 4}
    tokenizer = _fixture_module("test_method_cvpr_rae_bindings").TokenizerBindings()
    tokenizer.setUp()
    state = tokenizer.body.model.state_dict()
    # The builder strips final affine norm; the complete checkpoint loader
    # validates those original keys before replacing that norm at runtime.
    state["norm.weight"] = torch.ones(32)
    state["norm.bias"] = torch.zeros(32)
    path = root / "tokenizer.pt"
    torch.save(state, path)
    result["raev2_k7"] = {"kind": "raev2_k7", "arch": "fixture", "encoder": str(path),
                             "img_size": 224, "patch_size": 16, "embed_dim": 32, "depth": 24, "num_heads": 4}
    return result


def build_data(root):
    """Create official-shaped tiny readers, including an actual encoded video."""
    import av
    import h5py
    import numpy as np
    from PIL import Image
    from scipy.io import savemat

    root = Path(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(73)
    for split in ("train", "val"):
        for label in ("a", "b"):
            folder = root / "imagenet" / split / label
            folder.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (237, 249, 3), dtype=np.uint8)).save(folder / "sample.png")
    for split in ("training", "validation"):
        folder = root / "ade20k" / "images" / split
        masks = root / "ade20k" / "annotations" / split
        folder.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(2):
            Image.fromarray(rng.integers(0, 256, (237, 249, 3), dtype=np.uint8)).save(folder / f"{index}.jpg")
            Image.fromarray(rng.integers(1, 4, (237, 249), dtype=np.uint8)).save(masks / f"{index}.png")
    y, x = np.indices((480, 640))
    rgb = np.stack(((3*x+y) % 256, (x+7*y) % 256, (x+y) % 256), -1).astype("uint8")
    depth = (.3 + (x+y)/200).astype("float32")
    folder = root / "nyuv2" / "labeled"
    folder.mkdir(parents=True)
    with h5py.File(folder / "nyu_depth_v2_labeled.mat", "w") as handle:
        images = handle.create_dataset("images", (1449, 3, 640, 480), dtype="uint8", chunks=(1, 3, 640, 480))
        depths = handle.create_dataset("depths", (1449, 640, 480), dtype="float32", chunks=(1, 640, 480))
        for index in (0, 1, 795, 796):
            images[index] = rgb.transpose(2, 1, 0)
            depths[index] = depth.T
    savemat(folder / "splits.mat", {"trainNdxs": np.arange(1, 796), "testNdxs": np.arange(796, 1450)})
    for split in ("train", "val"):
        folder = root / "coco" / "images" / f"{split}2017"
        folder.mkdir(parents=True)
        records, annotations = [], []
        for index in range(2):
            Image.fromarray(rgb[:47, :61]).save(folder / f"{index}.png")
            records.append({"id": index, "file_name": f"{index}.png", "width": 61, "height": 47})
            annotations.append({"id": index, "image_id": index, "category_id": 7, "bbox": [4, 5, 24, 25], "area": 600, "iscrowd": 0})
        folder = root / "coco" / "annotations"
        folder.mkdir(exist_ok=True)
        (folder / f"instances_{split}2017.json").write_text(json.dumps({"info": {}, "images": records,
            "categories": [{"id": 7, "name": "fixture"}], "annotations": annotations}))
    folder = root / "ssv2"
    (folder / "labels").mkdir(parents=True)
    (folder / "videos").mkdir()
    entries = [{"id": str(i), "template": f"Moving [object{i}]"} for i in range(2)]
    (folder / "labels" / "labels.json").write_text(json.dumps({f"Moving [object{i}]": str(i) for i in range(2)}))
    for split in ("train", "validation"):
        (folder / "labels" / f"{split}.json").write_text(json.dumps(entries))
    for index in range(2):
        with av.open(str(folder / "videos" / f"{index}.mp4"), "w") as output:
            stream = output.add_stream("mpeg4", rate=12)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for frame in range(20):
                pixels = np.empty((48, 64, 3), dtype=np.uint8)
                pixels[:] = [30 + frame*5, 60 + index*30, 110]
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return {task: root / task for task in ("imagenet", "ade20k", "ssv2", "coco", "nyuv2")}


@unittest.skipUnless(_cuda_available(), "CUDA unavailable: GPU/BF16 matrix not executed")
class RevisedCUDA(unittest.TestCase):
    def setUp(self):
        import torch

        self.assertTrue(torch.cuda.is_available(), "this matrix requires real CUDA")
        self.assertTrue(torch.cuda.is_bf16_supported(), "native CUDA BF16 is required")
        for name in ("transformers", "timm", "einops", "av", "decord", "h5py", "scipy", "pycocotools"):
            self.assertIsNotNone(importlib.util.find_spec(name), f"CUDA matrix dependency missing: {name}")
        torch.set_num_threads(1)

    def test_five_providers_seven_real_cuda_bf16_cli_routes(self):
        import torch

        from downstream import contract
        from downstream.revised_tasks import checkpoint_identity

        with tempfile.TemporaryDirectory(prefix="revised-cuda-") as tmp:
            root = Path(tmp)
            datasets = build_data(root / "data")
            providers = build_checkpoints(root / "weights")
            self.assertEqual(len(providers), 5)
            completed = 0
            for provider, backbone in providers.items():
                original = checkpoint_identity(backbone["encoder"])
                for task, track in ROUTES:
                    with self.subTest(provider=provider, task=task, track=track):
                        diagnostic = {"epochs": 1, "batch_size": 2, "max_steps": 1,
                                          "max_train_samples": 2, "max_val_samples": 2, "dpt_hidden": 4}
                        if task == "coco":
                            diagnostic.update(detection_size=32, detection_proposals=8)
                        config = {"profile": "b5x_lora_dpt_v2", "task": task, "track": track, "seed": 12,
                                      "device": "cuda", "data_root": str(datasets[task]), "backbone": backbone,
                                      "diagnostic": diagnostic}
                        label = f"{provider}-{task}-{track}"
                        config_file = root / f"{label}.json"
                        config_file.write_text(json.dumps(config))
                        out = root / label
                        result = subprocess.run([sys.executable, "-m", "downstream.revised_tasks",
                            "--config", str(config_file), "--out", str(out)], cwd=ROOT,
                            env=dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1"),
                            capture_output=True, text=True, check=False, timeout=240)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        valid, errors = contract.verify(out, config_file, 0)
                        self.assertTrue(valid, errors)
                        report = json.loads((out / "result.json").read_text())
                        self.assertEqual(report["precision"], "cuda_bf16")
                        self.assertEqual(report["updates"], 1)
                        self.assertEqual(report["used_populations"], [2, 2])
                        self.assertEqual(report["metrics"]["n_images" if task in ("coco", "nyuv2") else "n"], 2)
                        self.assertEqual(report["status"], "diagnostic")
                        self.assertFalse(report["record_value"])
                        self.assertFalse(report["historical_run_verified"])
                        state = torch.load(out / "trainable.pt", map_location="cpu", weights_only=True)
                        self.assertTrue(state)
                        self.assertTrue(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in state.values()))
                        if track == "lora":
                            self.assertTrue(any(t.abs().sum() > 0 for n, t in state.items() if n.endswith("lora_B")))
                        else:
                            self.assertTrue(any(t.abs().sum() > 0 for n, t in state.items()
                                                if n.startswith("head.norms.") and n.endswith(".bias")))
                        allowed = ("backbone.fpn.", "rpn.", "roi_heads.") if task == "coco" else ("head.",)
                        self.assertTrue(all(n.endswith(("lora_A", "lora_B")) or n.startswith(allowed) for n in state))
                        self.assertEqual(checkpoint_identity(backbone["encoder"]), original)
                        for artifact in out.glob("*.json"):
                            self.assertNotIn(tmp, artifact.read_text())
                        completed += 1
                        print("CUDA_CASE_PASS " + json.dumps({"provider": provider, "task": task, "track": track}), flush=True)
            self.assertEqual(completed, 35, "all provider/route combinations must execute")

    def test_cuda_completed_epoch_resume_matches_fresh_process_training(self):
        import copy

        import torch

        from downstream.revised_tasks import checkpoint_identity

        fixture = _fixture_module("test_revised_cli_resume").RevisedCLIResumeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = copy.deepcopy(fixture.cfg)
        config["device"] = "cuda"
        config["diagnostic"]["epochs"] = 2
        original = checkpoint_identity(config["backbone"]["encoder"])
        reports = {}
        for label, override in (("full", {}), ("first", {"stop_after_epoch": 1}),
                                ("resumed", {"resume": str(fixture.root / "first/resume.pt")})):
            cfg = dict(config, **override)
            path = fixture.root / f"{label}.json"
            path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable, "-m", "downstream.revised_tasks",
                "--config", str(path), "--out", str(fixture.root / label)], cwd=ROOT,
                env=dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1"),
                capture_output=True, text=True, check=False, timeout=240)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports[label] = json.loads((fixture.root / label / "result.json").read_text())
            self.assertEqual(reports[label]["precision"], "cuda_bf16")
        self.assertEqual(reports["first"]["updates"], 2)
        self.assertFalse(reports["first"]["training_complete"])
        self.assertEqual(reports["full"]["updates"], 4)
        for key in ("metrics", "loss_log", "updates", "epochs_completed", "recipe", "training_complete"):
            fixture.equal(reports["full"][key], reports["resumed"][key])
        # Includes optimizer moments, scheduler, sampler and per-rank RNG state.
        for name in ("resume.pt", "trainable.pt"):
            fixture.equal(torch.load(fixture.root / "full" / name, map_location="cpu", weights_only=True),
                          torch.load(fixture.root / "resumed" / name, map_location="cpu", weights_only=True))
        self.assertEqual(checkpoint_identity(config["backbone"]["encoder"]), original)
        for label in reports:
            for path in (fixture.root / label).glob("*.json"):
                self.assertNotIn(str(fixture.root), path.read_text())
        print("CUDA_RESUME_PASS fresh_process_single_gpu_exact", flush=True)

    def test_geometry_cuda_matches_float32_cpu_and_restores_precision(self):
        import numpy as np
        import torch

        from downstream.representation_neighbors import (
            agreement_matrix,
            nearest_neighbors,
            patch_vectors,
        )
        from downstream.revised_precision import precision_policy, runtime_precision

        generator = torch.Generator().manual_seed(901)
        images = [torch.randn(43, 21, generator=generator) for _ in range(3)]
        maps = [torch.randn(4, 13, 7, 7, generator=generator) for _ in range(3)]
        before = runtime_precision()
        with precision_policy(torch.device("cuda"), matmul_tf32=False):
            for centered in (False, True):
                cpu, cuda = [], []
                for features, patches in zip(images, maps, strict=True):
                    image_cpu = nearest_neighbors(features, k=10, search_k=20, chunk_size=11)
                    image_cuda = nearest_neighbors(features.cuda(), k=10, search_k=20, chunk_size=11)
                    torch.testing.assert_close(image_cpu, image_cuda.cpu(), rtol=0, atol=0)
                    for target, tensor in ((cpu, patches), (cuda, patches.cuda())):
                        vectors, owners = patch_vectors(tensor, center_positions=centered)
                        target.append(nearest_neighbors(vectors, image_ids=owners, k=10,
                                                        chunk_size=11, pre_normalized=True))
                for a, b in zip(cpu, cuda, strict=True):
                    torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)
                np.testing.assert_array_equal(agreement_matrix(cpu), agreement_matrix(cuda))
        self.assertEqual(runtime_precision(), before)


if __name__ == "__main__":
    unittest.main()
