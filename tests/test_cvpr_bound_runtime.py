"""Bound adaptation lifecycle and frozen dense training behavior."""
import copy
import importlib.util
import unittest

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    from torch import nn

    from downstream import adaptation_runtime as runtime


@unittest.skipUnless(HAVE, 'torch required')
class BoundRuntime(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(32)

    def test_dense_probe_freezes_backbone_and_updates_only_head(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Conv2d(3, 4, 1)
                self.dropout = nn.Dropout(.8)
            def forward_dpt_features(self, x):
                return [self.dropout(self.layer(x))] * 4
        backbone = Encoder()
        model = runtime.FrozenDPTModel(backbone, [4]*4, out_channels=2, hidden=4)
        reference_head = copy.deepcopy(model.head)
        before = copy.deepcopy(backbone.state_dict())
        x = torch.randn(2, 3, 2, 3, requires_grad=True)
        model.train()
        self.assertTrue(model.head.training)
        self.assertFalse(any(m.training for m in backbone.modules()))
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        previous = copy.deepcopy(model.head.state_dict())
        output = model(x, (9, 11))
        with torch.no_grad():
            expected = reference_head(backbone.forward_dpt_features(x), (9, 11))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.square().mean().backward()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, model.head.state_dict()[k]) for k, v in previous.items()))
        for k, v in before.items():
            torch.testing.assert_close(v, backbone.state_dict()[k], rtol=0, atol=0)
        model.eval()
        restored = copy.deepcopy(model)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x, (9, 11)), restored(x, (9, 11)), rtol=0, atol=0)

    def test_dense_constructor_failure_does_not_freeze_or_change_mode(self):
        backbone = nn.Linear(3, 4)
        with self.assertRaisesRegex(TypeError, 'four-tap'):
            runtime.FrozenDPTModel(backbone, [4]*4, out_channels=2)
        self.assertTrue(backbone.training)
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
        backbone.forward_dpt_features = lambda x: [x]*4
        with self.assertRaises(ValueError):
            runtime.FrozenDPTModel(backbone, [4]*3, out_channels=2)
        self.assertTrue(backbone.training)
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))

    def test_lora_setup_enables_grad_after_success_only(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                block = nn.Module()
                block.attn = nn.ModuleDict({'qkv': nn.Linear(3, 3), 'proj': nn.Linear(3, 3)})
                self.blocks = nn.ModuleList([block])
                self.enabled = False
            def adaptation_block_groups(self):
                return [self.blocks]
            def enable_adaptation_grad(self):
                self.enabled = True
                self.eval()
        backbone = Encoder()
        with self.assertRaises(ValueError):
            runtime.install_backbone_lora(backbone, rank=0)
        self.assertFalse(backbone.enabled)
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
        adapters = runtime.install_backbone_lora(backbone, rank=2, alpha=4)
        self.assertTrue(backbone.enabled)
        self.assertFalse(backbone.training)
        self.assertEqual(len(adapters), 2)
        self.assertEqual({p.scaling for p in adapters.values()}, {2.})
        before = copy.deepcopy(backbone.state_dict())
        with self.assertRaises(ValueError):
            runtime.install_backbone_lora(backbone)
        for k, v in before.items():
            torch.testing.assert_close(v, backbone.state_dict()[k], rtol=0, atol=0)

    def test_unsupported_lora_provider_is_unchanged(self):
        backbone = nn.Linear(3, 4)
        with self.assertRaisesRegex(ValueError, 'adaptation binding'):
            runtime.install_backbone_lora(backbone)
        self.assertTrue(backbone.training)
        self.assertTrue(all(p.requires_grad for p in backbone.parameters()))
