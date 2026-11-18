"""Native encoder adaptation bindings retain model-specific readouts."""
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    __import__("einops")
    __import__("timm")
    from downstream import spatial_backbones as sb
    from downstream.lora import assert_adapter_usage, inject_attention_lora
    from provider_support import prepare_upstream
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAVE, 'torch and native encoder dependencies required')
class NativeBindings(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(79)

    def vjepa(self):
        provider = sb._load_provider(sb.discover_providers()['vjepa2_1'])
        return provider.build({"arch": 'vit_large', "encoder": '', "img_size": 384,
                                   "patch_size": 16, "embed_dim": 48, "depth": 12, "num_heads": 4})

    def omega(self):
        prepare_upstream(ROOT / 'third_party/vggt-omega', ('vggt_omega',))
        from vggt_omega.models.aggregator import Aggregator
        from vggt_omega.models.layers.vision_transformer import init_weights_vit
        model = Aggregator(embed_dim=64, depth=24, num_heads=4,
                           register_attention_block_indices=[0])
        for block in list(model.frame_blocks) + list(model.inter_frame_blocks):
            for name, module in block.named_modules():
                init_weights_vit(module, name)
        provider = sb._load_provider(sb.discover_providers()['vggt_omega'])
        return provider.Backbone(model)

    def test_lora_freezes_base_keeps_eval_and_updates_both_native_families(self):
        for build in (self.vjepa, self.omega):
            with self.subTest(build=build.__name__):
                body = build()
                groups = body.adaptation_block_groups()
                expected = ([body.encoder.blocks] if hasattr(body, 'encoder') else
                            [body.aggregator.frame_blocks, body.aggregator.inter_frame_blocks])
                self.assertEqual(groups, expected)
                adapters = inject_attention_lora(body, groups, rank=2, alpha=2)
                frozen = {n: p.detach().clone() for n, p in body.named_parameters() if not p.requires_grad}
                body.enable_adaptation_grad()
                body.train()
                self.assertFalse(any(m.training for m in body.modules()))
                optimizer = torch.optim.AdamW([p for p in body.parameters() if p.requires_grad], lr=.01)
                x = torch.randn(1, 3, 32, 32)
                before = body.forward_features(x).detach()
                for _ in range(3):
                    optimizer.zero_grad()
                    output = body.forward_features(x)
                    (output * torch.randn_like(output)).mean().backward()
                    self.assertTrue(all(layer.lora_B.grad is not None for layer in adapters.values()))
                    self.assertTrue(any(layer.lora_B.grad.abs().sum() > 0 for layer in adapters.values()))
                    optimizer.step()
                assert_adapter_usage(adapters)
                self.assertFalse(torch.equal(before, body.forward_features(x)))
                for name, parameter in body.named_parameters():
                    if name in frozen:
                        self.assertFalse(parameter.requires_grad)
                        self.assertIsNone(parameter.grad)
                        torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)

    def test_vjepa_four_raw_taps_keep_single_image_geometry_and_cleanup(self):
        body = self.vjepa()
        x = torch.randn(2, 3, 32, 48)
        captured = []
        handles = [body.encoder.blocks[i].register_forward_hook(
            lambda module, args, out: captured.append(out[0] if isinstance(out, (tuple, list)) else out)) for i in (2, 5, 8, 11)]
        body.encoder(x.unsqueeze(2))
        for handle in handles:
            handle.remove()
        expected = [t.transpose(1, 2).reshape(2, 48, 2, 3) for t in captured]
        actual = body.forward_dpt_features(x)
        self.assertEqual(len(actual), 4)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertFalse(left.requires_grad)
        self.assertTrue(all(not block._forward_hooks for block in body.encoder.blocks))
        body.enable_adaptation_grad()
        differentiable = x.clone().requires_grad_()
        sum(t.square().mean() for t in body.forward_dpt_features(differentiable)).backward()
        self.assertGreater(float(differentiable.grad.abs().sum()), 0)

    def test_vjepa_dpt_rejects_non_images_and_ambiguous_unaligned_grid(self):
        body = self.vjepa()
        for shape in ((1, 3, 31, 32), (1, 2, 3, 32, 32), (1, 4, 32, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                body.forward_dpt_features(torch.randn(shape))
        with (mock.patch.object(body.encoder.blocks[2], 'forward', side_effect=RuntimeError('fixture failure')),
              self.assertRaisesRegex(RuntimeError, 'fixture failure')):
            body.forward_dpt_features(torch.randn(1, 3, 32, 32))
        self.assertTrue(all(not block._forward_hooks for block in body.encoder.blocks))

    def test_omega_taps_use_cached_layer_four_not_even_layer_five(self):
        body = self.omega()
        x = torch.randn(1, 3, 32, 48)
        mean = x.new_tensor((.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        caches, prefix = body.aggregator((x * std + mean).unsqueeze(1))
        actual = body.forward_dpt_features(x)
        self.assertEqual(len(actual), 4)
        for value, index in zip(actual, (4, 11, 17, 23)):
            expected = caches[index][:, 0, prefix:].transpose(1, 2).reshape(1, 128, 2, 3)
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
            self.assertFalse(value.requires_grad)
        body.enable_adaptation_grad()
        x.requires_grad_()
        sum(t.square().mean() for t in body.forward_dpt_features(x)).backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_omega_dpt_refuses_wrong_cache_set_missing_layer_and_prefix(self):
        body = self.omega()
        x = torch.randn(1, 3, 32, 32)
        caches = [None] * 24
        for i in (4, 11, 17, 23):
            caches[i] = torch.randn(1, 1, 21, 128)
        body.aggregator.cached_layer_indices = {5, 11, 17, 23}
        with mock.patch.object(body.aggregator, 'forward', return_value=(caches, 17)) as forward:
            with self.assertRaisesRegex(ValueError, 'cached'):
                body.forward_dpt_features(x)
            forward.assert_not_called()
        body.aggregator.cached_layer_indices = {4, 11, 17, 23}
        wrong_prefix = [value[:, :, :-1] if value is not None else None for value in caches]
        for outputs in ((caches, 16), (wrong_prefix, 16), ([None] * 24, 17)):
            with (mock.patch.object(body.aggregator, 'forward', return_value=outputs),
                  self.assertRaises(ValueError)):
                body.forward_dpt_features(x)
        with self.assertRaises(ValueError):
            body.forward_dpt_features(torch.randn(1, 2, 3, 32, 32))
