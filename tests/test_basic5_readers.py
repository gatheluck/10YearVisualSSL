"""Executable AP/FT building-block contracts, not full recipe conformance."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TIMM = HAVE_TORCH and importlib.util.find_spec("timm") is not None
if HAVE_TORCH:
    import torch
    from torch import nn
    from downstream import spatial_backbones as sb

SPEC = dict(kind="vit", arch="vit_tiny_patch16_224", img_size=32,
            patch_size=16, embed_dim=32, depth=1, num_heads=2, encoder="")


@unittest.skipUnless(HAVE_TIMM, "Basic5 backbone tests need torch and timm")
class TestCheckpointAndTrainability(unittest.TestCase):
    def test_missing_checkpoint_weights_are_refused(self):
        model = sb.build_frozen_backbone(SPEC, torch.device("cpu"))
        state = model.vit.state_dict()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "encoder.pt"
            for bad in ({}, {k: v for k, v in state.items()
                             if k != "patch_embed.proj.weight"}):
                torch.save(bad, p)
                with self.subTest(keys=len(bad)), self.assertRaisesRegex(
                        RuntimeError, "missing.*weights"):
                    sb.build_frozen_backbone({**SPEC, "encoder": str(p)},
                                             torch.device("cpu"))

    def test_complete_checkpoint_matches_and_only_classifier_is_ignored(self):
        model = sb.build_frozen_backbone(SPEC, torch.device("cpu"))
        x = torch.randn(2, 3, 32, 32)
        state = model.vit.state_dict()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "encoder.pt"
            state["head.weight"] = torch.zeros(1000, 32)
            state["head.bias"] = torch.zeros(1000)
            torch.save(state, p)
            restored = sb.build_frozen_backbone({**SPEC, "encoder": str(p)},
                                                torch.device("cpu"))
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
            state["head.unrelated"] = torch.zeros(1)
            torch.save(state, p)
            with self.assertRaisesRegex(RuntimeError, "keys"):
                sb.build_frozen_backbone({**SPEC, "encoder": str(p)},
                                         torch.device("cpu"))

    def test_explicit_finetune_updates_backbone_and_preserves_spatial_readout(self):
        builder = getattr(sb, "build_trainable_backbone", None)
        self.assertTrue(callable(builder), "explicit FT builder is required")
        frozen = sb.build_frozen_backbone(SPEC, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "encoder.pt"
            torch.save(frozen.vit.state_dict(), p)
            ft = builder({**SPEC, "encoder": str(p)}, torch.device("cpu"))
        x = torch.randn(2, 3, 32, 32)
        ft.eval()
        torch.testing.assert_close(ft(x), frozen(x), rtol=0, atol=0)
        before = ft.vit.patch_embed.proj.weight.detach().clone()
        head = nn.Conv2d(32, 3, 1)
        model = nn.Sequential(ft, head).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        model(x).square().mean().backward()
        self.assertGreater(ft.vit.patch_embed.proj.weight.grad.abs().sum(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, ft.vit.patch_embed.proj.weight))
        self.assertTrue(all(p.requires_grad for p in ft.parameters()))

    def test_unknown_trainability_is_not_silently_unfrozen(self):
        builder = getattr(sb, "build_trainable_backbone", None)
        self.assertTrue(callable(builder), "explicit FT builder is required")
        from types import SimpleNamespace
        from unittest import mock
        # Capabilities, not the historical assumption that every provider is frozen.
        for flag, factory in ((False, lambda spec: nn.Linear(2, 2)),
                              (True, None), (1, lambda spec: nn.Linear(2, 2))):
            provider = SimpleNamespace(TRAINABLE=flag, build_trainable=factory)
            with mock.patch.object(sb, "_PROVIDERS", {"fixture": Path("unused")}), \
                 mock.patch.object(sb, "_load_provider", return_value=provider):
                with self.assertRaisesRegex(NotImplementedError, "trainable"):
                    builder({**SPEC, "kind": "fixture"}, torch.device("cpu"))
        with self.assertRaisesRegex(NotImplementedError, "trainable"):
            builder({**SPEC, "kind": "not_a_provider"}, torch.device("cpu"))
        expected = nn.Linear(2, 2)
        provider = SimpleNamespace(TRAINABLE=True, build_trainable=lambda spec: expected)
        with mock.patch.object(sb, "_PROVIDERS", {"fixture": Path("unused")}), \
             mock.patch.object(sb, "_load_provider", return_value=provider):
            self.assertIs(builder({**SPEC, "kind": "fixture"}, torch.device("cpu")), expected)

    def test_finetune_batchnorm_updates_but_frozen_batchnorm_does_not(self):
        cls = getattr(sb, "ViTSpatialBackbone", None)
        self.assertTrue(callable(cls), "trainability-aware spatial wrapper is required")
        class Tokens(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch = nn.Conv2d(3, 8, 2, stride=2)
                self.norm = nn.BatchNorm2d(8)
            def forward_features(self, x):
                return self.norm(self.patch(x)).flatten(2).transpose(1, 2)
        for trainable in (False, True):
            with self.subTest(trainable=trainable):
                model = cls(Tokens(), 2, 0, 8, trainable=trainable)
                model.train()
                before = model.vit.norm.running_mean.clone()
                y = model(torch.randn(2, 3, 8, 8) + 3)
                self.assertEqual(y.requires_grad, trainable)
                self.assertEqual(not torch.equal(before, model.vit.norm.running_mean),
                                 trainable)
                model.eval()
                before = model.vit.norm.running_mean.clone()
                model(torch.randn(2, 3, 8, 8))
                torch.testing.assert_close(before, model.vit.norm.running_mean)


@unittest.skipUnless(HAVE_TORCH, "Basic5 attention tests need torch")
class TestReaders(unittest.TestCase):
    def readers(self):
        spec = importlib.util.find_spec("downstream.attention")
        self.assertIsNotNone(spec, "shared attention readers are required")
        from downstream import attention
        return attention

    def test_query_reader_topology_and_tokens_affect_output(self):
        cls = self.readers().QueryReader
        model = cls(24).eval()
        self.assertEqual(tuple(model.queries.shape), (32, 512))
        attn = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
        self.assertEqual(len(attn), 1)
        self.assertEqual((attn[0].embed_dim, attn[0].num_heads, attn[0].dropout),
                         (512, 8, 0.0))
        self.assertTrue(any(isinstance(m, nn.Linear) and m.out_features == 2048
                            for m in model.modules()))
        x = torch.randn(2, 7, 24, requires_grad=True)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 512))
        out.square().mean().backward()
        self.assertGreater(x.grad.abs().sum(), 0)
        self.assertGreater(model.queries.grad.abs().sum(), 0)
        changed = x.detach().clone()
        changed[:, 0] += 4
        self.assertFalse(torch.allclose(out.detach(), model(changed)))
        # No temporal position policy is specified: do not claim order sensitivity.
        torch.testing.assert_close(model(x.flip(1)), out, rtol=1e-5, atol=1e-6)

    def test_query_reader_matches_explicit_pre_norm_residual_equations(self):
        model = self.readers().QueryReader(24).eval()
        x = torch.randn(2, 5, 24)
        q = model.queries.unsqueeze(0).expand(2, -1, -1)
        context = model.input_projection(x)
        block = model.block
        kv = block.context_norm(context)
        attended = block.attention(block.query_norm(q), kv, kv,
                                   need_weights=False)[0]
        residual = q + attended
        expected = (residual + block.mlp(block.mlp_norm(residual))).mean(1)
        torch.testing.assert_close(model(x), expected)

    def test_query_initialization_matches_reference_width_scaling(self):
        cls = self.readers().QueryReader
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            model = cls(24)
            torch.manual_seed(91)
            nn.Linear(24, 512)
            expected = torch.randn(32, 512) / (512 ** .5)
            torch.testing.assert_close(model.queries, expected, rtol=0, atol=0)

    def test_spatial_projections_match_reference_pointwise_convolutions(self):
        model = self.readers().SpatialAdapter(24)
        for layer in (model.input_projection, model.output_projection):
            self.assertIsInstance(layer, nn.Conv2d)
            self.assertEqual(layer.kernel_size, (1, 1))

    def test_query_reader_rejects_global_and_empty_inputs(self):
        model = self.readers().QueryReader(24)
        for shape in ((2, 24), (2, 0, 24), (2, 3, 25)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "tokens"):
                model(torch.randn(*shape))

    def test_spatial_adapter_is_identity_then_learns_with_shared_scales(self):
        model = self.readers().SpatialAdapter(24)
        x = torch.randn(2, 24, 3, 5)
        y = torch.randn(2, 24, 2, 2)
        torch.testing.assert_close(model(x), x, rtol=0, atol=0)
        torch.testing.assert_close(model(y), y, rtol=0, atol=0)
        attn = [m for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
        self.assertEqual(len(attn), 1)
        self.assertEqual((attn[0].embed_dim, attn[0].num_heads), (256, 8))
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        (model(x).square().mean() + model(y).square().mean()).backward()
        opt.step()
        self.assertFalse(torch.equal(model(x), x))
        opt.zero_grad()
        model(x).square().mean().backward()
        self.assertGreater(model.input_projection.weight.grad.abs().sum(), 0)

    def test_spatial_adapter_refuses_global_or_incompatible_channels(self):
        model = self.readers().SpatialAdapter(24)
        for shape in ((2, 24), (2, 25, 2, 2), (2, 24, 0, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "spatial"):
                model(torch.randn(*shape))

    def test_frozen_backbone_state_survives_parent_train_and_optimizer(self):
        attention = self.readers()
        class Backbone(nn.Module):
            out_channels = 24
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Conv2d(3, 24, 1),
                                            nn.BatchNorm2d(24), nn.Dropout(.8))
            def forward_features(self, x):
                return self.layers(x)
        original = Backbone()
        model = attention.AttentiveSpatialBackbone(original)
        before = {k: v.clone() for k, v in original.state_dict().items()}
        parent = nn.Sequential(model, nn.Conv2d(24, 2, 1)).train()
        self.assertFalse(original.training)
        self.assertTrue(model.adapter.training)
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        optimizer = torch.optim.SGD(parent.parameters(), lr=.01)
        parent(x).square().mean().backward()
        optimizer.step()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad
                            for p in original.parameters()))
        for k, v in original.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertGreater(model.adapter.output_projection.weight.grad.abs().sum(), 0)

    @unittest.skipUnless(HAVE_TIMM, "integration needs timm")
    def test_public_attentive_builder_drives_a_spatial_head(self):
        builder = getattr(sb, "build_attentive_backbone", None)
        self.assertTrue(callable(builder), "public AP builder is required")
        model = nn.Sequential(builder(SPEC, torch.device("cpu")), nn.Conv2d(32, 2, 1))
        model.train()
        model(torch.randn(2, 3, 32, 32)).square().mean().backward()
        self.assertGreater(model[0].adapter.output_projection.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model[0].backbone.parameters()))

    @unittest.skipUnless(HAVE_TIMM, "GPU integration needs timm")
    def test_cuda_attention_and_finetune_update_and_reload(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA integration requires a GPU")
        attention = self.readers()
        builder = getattr(sb, "build_trainable_backbone", None)
        self.assertTrue(callable(builder), "explicit FT builder is required")
        device = torch.device("cuda")
        for build in (builder, sb.build_attentive_backbone):
            model = nn.Sequential(build(SPEC, device), nn.Conv2d(32, 2, 1).to(device))
            model.train()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            x = torch.randn(2, 3, 32, 32, device=device)
            opt = torch.optim.SGD(model.parameters(), lr=.01)
            opt.zero_grad()
            loss = model(x).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            self.assertTrue(any(not torch.equal(v, before[k])
                                for k, v in model.state_dict().items()))
            model.eval()
            expected = model(x).detach()
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "state.pt"
                torch.save(model.state_dict(), p)
                restored = nn.Sequential(build(SPEC, device),
                                         nn.Conv2d(32, 2, 1).to(device)).eval()
                restored.load_state_dict(torch.load(p, weights_only=True))
                torch.testing.assert_close(restored(x), expected)
        reader = attention.QueryReader(32).to(device)
        output = reader(torch.randn(2, 16, 32, device=device))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(reader.queries.grad).all())
