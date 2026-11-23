"""Revised DINOv3 checkpoint and rotary storage at the first training input."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "requires Torch and DINOv3 transformers")
class DinoRevisedStorage(unittest.TestCase):
    def test_revised_cuda_load_preserves_native_rotary_frequencies(self):
        from downstream import revised_tasks
        from downstream.spatial_backbones import build_frozen_backbone

        class ObservedInput(Exception):
            pass

        class Input:
            def __len__(self):
                return 8

            def __getitem__(self, index):
                raise ObservedInput

        with tempfile.TemporaryDirectory() as root:
            config = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
                                      intermediate_size=96, num_register_tokens=2, image_size=32, patch_size=16)
            DINOv3ViTModel(config).save_pretrained(root)
            spec = dict(encoder=root, kind="dinov3_hf", arch="fixture", img_size=32, patch_size=16)
            reference = DINOv3ViTModel.from_pretrained(root, dtype=torch.bfloat16)
            observed = []

            def on_host(spec, device, **options):
                backbone = build_frozen_backbone(spec, torch.device("cpu"), **options)
                observed.append(backbone.model)
                return backbone

            context = SimpleNamespace(device=torch.device("cuda"), world=1, rank=0,
                                      call=lambda fn: fn(), agree=lambda value: None)
            with patch.object(revised_tasks, "build_frozen_backbone", side_effect=on_host), \
                 patch.object(revised_tasks, "prepare_datasets", return_value=(Input(), Input(), {})), \
                 self.assertRaises(ObservedInput):
                revised_tasks._execute(dict(task="nyuv2", track="dpt", seed=0, backbone=spec), None, context)

            actual, = observed
            self.assertEqual(next(actual.parameters()).dtype, torch.bfloat16)
            torch.testing.assert_close(actual.rope_embeddings.inv_freq, reference.rope_embeddings.inv_freq,
                                       rtol=0, atol=0)
            legacy = build_frozen_backbone(spec, torch.device("cpu"))
            self.assertEqual(next(legacy.parameters()).dtype, torch.float32)

    def test_lora_recomputes_each_block_only_on_recorded_tasks(self):
        from downstream.hf_vision import VisionBackbone
        from downstream.revised_recipe import resolve_recipe
        from downstream.revised_tasks import build_model

        config = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                 intermediate_size=48, image_size=32, patch_size=16)
        image = torch.randn(1, 3, 32, 32)
        for task in ('imagenet', 'ssv2', 'coco'):
            with self.subTest(task=task):
                torch.manual_seed(7)
                backbone = VisionBackbone(DINOv3ViTModel(config), family='register_cls')
                build_model(backbone, resolve_recipe(task, 'lora'), image, seed=0)
                blocks, = backbone.adaptation_block_groups()
                calls = [[] for _ in blocks]
                for block, log in zip(blocks, calls):
                    first = next(m for m in block.modules() if isinstance(m, torch.nn.LayerNorm))
                    first.register_forward_pre_hook(lambda *args, log=log: log.append(True))
                with torch.no_grad():
                    backbone.model(image)
                self.assertEqual([len(log) for log in calls], [1, 1])
                output = backbone.model(image).last_hidden_state
                output[..., 0].square().mean().backward()
                self.assertEqual([len(log) for log in calls], [2, 2] if task == 'imagenet' else [3, 3])
                self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)
                                    for p in backbone.parameters() if p.requires_grad))
