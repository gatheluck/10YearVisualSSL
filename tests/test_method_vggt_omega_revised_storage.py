"""Check loaded backbone values at the first revised training input."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class RevisedBackboneStorage(unittest.TestCase):
    def test_cuda_preparation_preserves_omega_checkpoint_values(self):
        import torch
        from downstream import revised_tasks
        from methods.vggt_omega.downstream_backbone import Backbone

        class ObservedInput(Exception):
            pass

        context = SimpleNamespace(device=torch.device("cuda"), world=1, rank=0,
                                  call=lambda fn: fn(), agree=lambda value: None)
        for provider, expected in ((Backbone, torch.float32), (torch.nn.Module, torch.bfloat16)):
            with self.subTest(provider=provider.__name__):
                backbone = provider.__new__(provider)
                torch.nn.Module.__init__(backbone)
                source = torch.tensor([1.0001])
                backbone.register_parameter("weight", torch.nn.Parameter(source.clone()))
                backbone.register_buffer("scale", source.clone())

                class Input:
                    def __len__(self):
                        return 8

                    def __getitem__(self, index):
                        raise ObservedInput

                dataset = Input()
                cfg = dict(task="nyuv2", track="dpt", seed=0, backbone={"kind": "vggt_omega"})
                with patch.object(revised_tasks, "prepare_datasets", return_value=(dataset, dataset, {})), \
                     patch.object(revised_tasks, "build_frozen_backbone", return_value=backbone), \
                     self.assertRaises(ObservedInput):
                    revised_tasks._execute(cfg, None, context)
                for tensor in (backbone.weight, backbone.scale):
                    self.assertEqual(tensor.dtype, expected)
                    torch.testing.assert_close(tensor, source.to(expected), rtol=0, atol=0)
