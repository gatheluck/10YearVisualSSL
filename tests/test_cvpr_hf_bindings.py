"""Real reduced vision towers exercise the revised adaptation boundary."""
import copy
import unittest

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import (
        DINOv3ViTConfig,
        DINOv3ViTModel,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "torch and current transformers required")
class HFBindings(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def backbone(self, family, *, trainable=False):
        from downstream.hf_vision import VisionBackbone
        cfg = {'hidden_size': 32, 'intermediate_size': 48, 'num_hidden_layers': 6,
               'num_attention_heads': 4, 'image_size': 32, 'patch_size': 16}
        if family == 'map_pool':
            model = SiglipVisionModel(SiglipVisionConfig(**cfg))
        else:
            model = DINOv3ViTModel(DINOv3ViTConfig(
                **cfg, num_register_tokens=2, pos_embed_rescale=None))
        return VisionBackbone(model, family=family, trainable=trainable)

    def require_api(self, bb):
        for name in ('adaptation_block_groups', 'enable_adaptation_grad', 'forward_dpt_features'):
            self.assertTrue(callable(getattr(bb, name, None)), name + ' is missing')

    def test_blocks_are_complete_and_ambiguous_or_other_family_is_rejected(self):
        for family in ('map_pool', 'register_cls'):
            bb = self.backbone(family)
            self.require_api(bb)
            blocks = bb.adaptation_block_groups()
            expected = next(m for n, m in bb.model.named_modules() if n.endswith('encoder.layers' if family == 'map_pool' else 'layer') and isinstance(m, nn.ModuleList))
            self.assertEqual(len(blocks), 1)
            self.assertIs(blocks[0], expected)
            bb.model.decoy = nn.ModuleList([nn.Identity() for _ in range(5)])
            self.assertIs(bb.adaptation_block_groups()[0], expected)
            bb.model.ambiguous = nn.ModuleList([nn.Identity() for _ in range(6)])
            with self.assertRaisesRegex(ValueError, 'unique|one|ambiguous'):
                bb.adaptation_block_groups()
            del bb.model.ambiguous
            bb.model.config.num_hidden_layers = 99
            with self.assertRaisesRegex(ValueError, 'unique|one|blocks'):
                bb.adaptation_block_groups()
        bb = self.backbone('map_pool')
        bb.family = 'projected_cls'
        with self.assertRaisesRegex(ValueError, 'family|binding'):
            bb.adaptation_block_groups()
        with self.assertRaises(ValueError):
            bb.enable_adaptation_grad()
        with self.assertRaises(ValueError):
            bb.forward_dpt_features(torch.randn(1, 3, 32, 32))

    def test_lora_keeps_eval_and_base_frozen_but_updates_adapters(self):
        from downstream.adaptation_runtime import check_adapter_effect
        from downstream.lora import assert_adapter_usage, inject_attention_lora
        for family in ('map_pool', 'register_cls'):
            bb = self.backbone(family)
            self.require_api(bb)
            x = torch.randn(2, 3, 32, 32)
            baseline = bb.classification_features(x, adaptation='frozen')
            base = [(p, p.detach().clone()) for p in bb.parameters()]
            adapters = inject_attention_lora(bb, bb.adaptation_block_groups(), rank=2, alpha=2)
            bb.enable_adaptation_grad()
            bb.train()
            self.assertFalse(any(m.training for m in bb.modules()))
            self.assertFalse(any(p.requires_grad for p, _ in base))
            def forward(bb=bb, x=x):
                return bb.classification_features(x, adaptation='frozen')
            torch.testing.assert_close(forward(), baseline, rtol=0, atol=0)
            check_adapter_effect(forward, adapters)
            optimizer = torch.optim.AdamW([p for p in bb.parameters() if p.requires_grad], lr=.01)
            target = torch.randn_like(baseline)
            for _ in range(3):
                optimizer.zero_grad()
                (forward() * target).sum().backward()
                for layer in adapters.values():
                    self.assertIsNotNone(layer.lora_B.grad)
                    self.assertGreater(layer.lora_B.grad.abs().sum().item(), 0)
                optimizer.step()
            assert_adapter_usage(adapters)
            self.assertGreater((forward() - baseline).abs().max().item(), 1e-6)
            for p, before in base:
                torch.testing.assert_close(p, before, rtol=0, atol=0)
                self.assertIsNone(p.grad)
            with torch.no_grad():
                self.assertFalse(forward().requires_grad)

    def test_dpt_raw_taps_match_independent_hooks_and_input_gradients(self):
        for family in ('map_pool', 'register_cls'):
            bb = self.backbone(family)
            self.require_api(bb)
            bb.enable_adaptation_grad()
            reference = copy.deepcopy(bb.model)
            x = torch.randn(2, 3, 33, 49, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            blocks = next(m for n, m in reference.named_modules() if n.endswith('encoder.layers' if family == 'map_pool' else 'layer') and isinstance(m, nn.ModuleList))
            outputs, handles = [], []
            for i in (1, 2, 4, 5):
                handles.append(blocks[i].register_forward_hook(
                    lambda m, args, out, outputs=outputs: outputs.append(out[0] if isinstance(out, tuple) else out)))
            if family == 'map_pool':
                raw = y * y.new_tensor((.229, .224, .225))[None, :, None, None] + y.new_tensor((.485, .456, .406))[None, :, None, None]
                reference(pixel_values=(raw - .5) / .5, interpolate_pos_encoding=True)
                h, w, prefix = 2, 3, 0
            else:
                reference(pixel_values=F.pad(y, (0, 15, 0, 15)))
                h, w, prefix = 3, 4, 3
            for handle in handles:
                handle.remove()
            expected = [v[:, prefix:].transpose(1, 2).reshape(2, 32, h, w) for v in outputs]
            actual = bb.forward_dpt_features(x)
            self.assertEqual(len(actual), 4)
            weights = [torch.randn_like(v) for v in expected]
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            sum((v * w).sum() for v, w in zip(actual, weights)).backward()
            sum((v * w).sum() for v, w in zip(expected, weights)).backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            self.assertFalse(any(m._forward_hooks for m in bb.model.modules()))

    def test_frozen_dpt_and_existing_finetune_remain_distinct(self):
        for family in ('map_pool', 'register_cls'):
            bb = self.backbone(family)
            self.require_api(bb)
            self.assertTrue(all(not v.requires_grad for v in bb.forward_dpt_features(torch.randn(1, 3, 32, 32))))
            ft = self.backbone(family, trainable=True)
            ft.train()
            self.assertTrue(ft.training)
            ft.enable_adaptation_grad()
            ft.train()
            self.assertFalse(ft.training)
            if family == 'register_cls':
                self.assertFalse(ft.model.rope_embeddings.training)
            self.assertTrue(all(p.requires_grad for p in ft.parameters()))

    def test_public_composition_updates_only_dense_head_and_keeps_positive_depth(self):
        from downstream.adaptation_runtime import FrozenDPTModel, install_backbone_lora
        for family in ('map_pool', 'register_cls'):
            bb = self.backbone(family)
            x = torch.randn(2, 3, 32, 32)
            model = FrozenDPTModel(bb, [32] * 4, out_channels=1, hidden=4, positive=True)
            before = {n: p.clone() for n, p in model.named_parameters()}
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001)
            model.train()
            self.assertTrue(model.head.training)
            self.assertFalse(any(m.training for m in bb.modules()))
            out = model(x, (9, 11))
            self.assertEqual(out.shape, (2, 1, 9, 11))
            self.assertTrue(bool((out > .001).all()))
            out.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.head.parameters()))
            self.assertFalse(any(p.grad is not None for p in bb.parameters()))
            optimizer.step()
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith('head.')))
            for n, p in model.named_parameters():
                if n.startswith('backbone.'):
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            adapted = self.backbone(family)
            adapters = install_backbone_lora(adapted, rank=2, alpha=2)
            adapted.classification_features(x, adaptation='frozen').sum().backward()
            self.assertEqual(len(adapters), 24)
            self.assertTrue(all(layer.lora_B.grad is not None for layer in adapters.values()))


if __name__ == '__main__':
    unittest.main()
