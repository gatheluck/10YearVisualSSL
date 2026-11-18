"""Tokenizer-specific adaptation bindings, distinct from generator readouts."""
import copy
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    from downstream.lora import assert_adapter_usage, inject_attention_lora
    from methods.raev2.downstream_backbone import Backbone
    from provider_support import prepare_upstream
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAVE, "torch required")
class TokenizerBindings(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)
        prepare_upstream(ROOT / "third_party/dinov3", ("dinov3",))
        from dinov3.models.vision_transformer import DinoVisionTransformer
        model = DinoVisionTransformer(img_size=224, patch_size=16, embed_dim=32,
                                     depth=24, num_heads=4, n_storage_tokens=4,
                                     pos_embed_rope_rescale_coords=2.)
        model.init_weights()
        model.norm = nn.LayerNorm(32, elementwise_affine=False)
        self.body = Backbone(model)
        self.images = torch.randn(2, 3, 31, 47)

    def require_api(self, name):
        self.assertTrue(callable(getattr(self.body, name, None)), name + " is missing")

    def test_four_normalized_taps_match_official_readout_not_k7(self):
        self.require_api("forward_dpt_features")
        padded = F.pad(self.images, (0, 1, 0, 1))
        with torch.no_grad():
            expected = self.body.model.get_intermediate_layers(
                padded, n=[5, 11, 17, 23], reshape=True,
                return_class_token=False, norm=True)
            raw = self.body.model.get_intermediate_layers(
                padded, n=[5, 11, 17, 23], reshape=True,
                return_class_token=False, norm=False)
        observed = self.body.forward_dpt_features(self.images.requires_grad_())
        self.assertEqual(len(observed), 4)
        for actual, wanted, unnormalized in zip(observed, expected, raw):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            self.assertEqual(actual.shape, (2, 32, 2, 3))
            self.assertFalse(actual.requires_grad)
            self.assertFalse(torch.allclose(actual, unnormalized))
        self.assertFalse(torch.allclose(observed[-1], self.body.forward_features(self.images)))

    def test_adapter_gradients_updates_and_eval_do_not_unfreeze_base(self):
        self.require_api("adaptation_block_groups")
        self.require_api("enable_adaptation_grad")
        before = self.body.forward_features(self.images)
        groups = self.body.adaptation_block_groups()
        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0], self.body.model.blocks)
        adapters = inject_attention_lora(self.body, groups, rank=2, alpha=2)
        self.assertEqual(len(adapters), 48)
        base = {n: p.detach().clone() for n, p in self.body.named_parameters()
                if not p.requires_grad}
        self.body.enable_adaptation_grad()
        self.body.train()
        self.assertTrue(all(not m.training for m in self.body.modules()))
        self.assertFalse(self.body.trainable)
        output = self.body.forward_features(self.images)
        torch.testing.assert_close(output, before, rtol=0, atol=0)
        assert_adapter_usage(adapters)
        parameters = [p for p in self.body.parameters() if p.requires_grad]
        self.assertEqual(len(parameters), 96)
        optimizer = torch.optim.AdamW(parameters, lr=.003)
        target = torch.randn_like(output)
        for _ in range(3):
            optimizer.zero_grad()
            (self.body.forward_features(self.images) - target).square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in parameters))
            optimizer.step()
        self.assertTrue(all(torch.count_nonzero(a.lora_B) for a in adapters.values()))
        self.assertFalse(torch.equal(self.body.forward_features(self.images), before))
        for n, p in self.body.named_parameters():
            if n in base:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
                torch.testing.assert_close(p, base[n], rtol=0, atol=0)

        self.body.trainable = True
        self.body.train()
        self.assertTrue(all(not m.training for m in self.body.modules()))

    def test_tap_gradients_match_direct_upstream(self):
        self.require_api("forward_dpt_features")
        self.require_api("enable_adaptation_grad")
        reference = copy.deepcopy(self.body.model)
        self.body.enable_adaptation_grad()
        x = self.images.clone().requires_grad_()
        y = self.images.clone().requires_grad_()
        actual = self.body.forward_dpt_features(x)
        expected = reference.get_intermediate_layers(
            F.pad(y, (0, 1, 0, 1)), n=[5, 11, 17, 23], reshape=True,
            return_class_token=False, norm=True)
        target = [torch.randn_like(t) for t in expected]
        sum((a - t).square().mean() for a, t in zip(actual, target)).backward()
        sum((a - t).square().mean() for a, t in zip(expected, target)).backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_invalid_images_and_wrong_tap_outputs_fail(self):
        self.require_api("forward_dpt_features")
        for images in (torch.randn(2, 4, 16, 16), torch.randn(2, 3, 0, 16),
                       torch.randn(3, 16, 16)):
            with self.subTest(shape=images.shape), self.assertRaisesRegex(ValueError, "images"):
                self.body.forward_dpt_features(images)
        for outputs in ([torch.zeros(2, 32, 2, 3)] * 3,
                        [torch.zeros(2, 32, 2, 4)] * 4):
            with mock.patch.object(self.body.model, "get_intermediate_layers", return_value=outputs), \
                    self.assertRaisesRegex(ValueError, "four|grid"):
                self.body.forward_dpt_features(self.images)


if __name__ == "__main__":
    unittest.main()
