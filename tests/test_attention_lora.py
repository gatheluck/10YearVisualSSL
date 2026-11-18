"""Behavioral checks for explicit attention-projection adapters."""
import copy
import importlib.util
import unittest

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    from torch import nn
    from downstream import lora


@unittest.skipUnless(HAVE, 'torch required')
class AttentionLoRA(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_initial_output_and_base_freezing(self):
        base = nn.Linear(4, 3)
        x = torch.randn(2, 4)
        expected = base(x).detach()
        adapted = lora.LowRankLinear(base, rank=2, alpha=4)
        self.assertFalse(any(p.requires_grad for p in base.parameters()))
        torch.testing.assert_close(adapted(x), expected, rtol=0, atol=0)
        self.assertIs(adapted.weight, base.weight)
        self.assertIs(adapted.bias, base.bias)
        self.assertEqual((adapted.in_features, adapted.out_features), (4, 3))
        self.assertEqual(adapted.lora_A.dtype, torch.float32)
        self.assertEqual(adapted.lora_B.dtype, torch.float32)
        self.assertEqual(adapted.calls, 1)

    def test_outputs_gradients_and_three_updates_match_equation(self):
        adapted = lora.LowRankLinear(nn.Linear(4, 3), rank=2, alpha=6)
        weight = adapted.weight.detach().clone()
        bias = adapted.bias.detach().clone()
        a = nn.Parameter(adapted.lora_A.detach().clone())
        b = nn.Parameter(adapted.lora_B.detach().clone())
        optimizer = torch.optim.AdamW([adapted.lora_A, adapted.lora_B], lr=.03)
        reference = torch.optim.AdamW([a, b], lr=.03)
        for _ in range(3):
            x = torch.randn(2, 5, 4, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            optimizer.zero_grad(); reference.zero_grad()
            actual = adapted(x)
            expected = torch.nn.functional.linear(y, weight, bias) + 3 * ((y @ a.T) @ b.T)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward(); expected.square().mean().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            for first, second in ((adapted.lora_A, a), (adapted.lora_B, b)):
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
            optimizer.step(); reference.step()
            torch.testing.assert_close(adapted.lora_A, a, rtol=0, atol=0)
            torch.testing.assert_close(adapted.lora_B, b, rtol=0, atol=0)
        torch.testing.assert_close(adapted.weight, weight, rtol=0, atol=0)
        self.assertIsNone(adapted.weight.grad)
        self.assertGreater(torch.count_nonzero(adapted.lora_B).item(), 0)

    def test_low_precision_base_keeps_float32_adapters(self):
        for dtype in (torch.float16, torch.bfloat16):
            adapted = lora.LowRankLinear(nn.Linear(4, 3, bias=False).to(dtype), rank=2, alpha=2)
            x = torch.randn(2, 4).to(dtype)
            self.assertEqual(adapted(x).dtype, dtype)
            self.assertEqual(adapted.lora_A.dtype, torch.float32)
            self.assertEqual(adapted.lora_B.dtype, torch.float32)
            self.assertIsNone(adapted.bias)
            adapted(x).float().sum().backward()
            self.assertIsNotNone(adapted.lora_B.grad)

    def test_invalid_rank_scale_and_target_leave_base_untouched(self):
        for rank, alpha in ((0, 1), (-1, 1), (True, 1), (1.5, 1), (2, 0), (2, -1), (2, float('nan')), (2, float('inf')), (2, True)):
            base = nn.Linear(4, 3)
            with self.subTest(rank=rank, alpha=alpha), self.assertRaises(ValueError):
                lora.LowRankLinear(base, rank=rank, alpha=alpha)
            self.assertTrue(base.weight.requires_grad)
        with self.assertRaises(TypeError):
            lora.LowRankLinear(nn.Identity())

    def model(self, counts=(2, 4)):
        model = nn.Module()
        model.stem = nn.Linear(4, 4)
        model.blocks = nn.ModuleList()
        for count in counts:
            block = nn.Module()
            block.attention = nn.Module()
            names = ('qkv', 'proj') if count == 2 else ('q_proj', 'k_proj', 'v_proj', 'out_proj')[:count]
            for name in names:
                setattr(block.attention, name, nn.Linear(4, 4))
            block.mlp = nn.Module(); block.mlp.proj = nn.Linear(4, 4)
            model.blocks.append(block)
        return model

    def test_projection_selection_and_trainable_scope(self):
        model = self.model()
        original = copy.deepcopy(model.state_dict())
        adapters = lora.inject_attention_lora(model, [model.blocks])
        self.assertEqual(len(adapters), 6)
        self.assertEqual(set(adapters), {'blocks.0.attention.qkv', 'blocks.0.attention.proj', 'blocks.1.attention.q_proj', 'blocks.1.attention.k_proj', 'blocks.1.attention.v_proj', 'blocks.1.attention.out_proj'})
        self.assertIsInstance(model.blocks[0].mlp.proj, nn.Linear)
        self.assertFalse(model.stem.weight.requires_grad)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {path + suffix for path in adapters for suffix in ('.lora_A', '.lora_B')})
        for path, module in adapters.items():
            self.assertEqual(module.lora_A.shape, (16, 4))
            torch.testing.assert_close(module.weight, original[path+'.weight'], rtol=0, atol=0)

    def test_bad_membership_counts_and_repeat_are_atomic(self):
        for variant in ('late_bad_count', 'foreign', 'duplicate', 'empty'):
            model = self.model((2, 3) if variant == 'late_bad_count' else (2, 4))
            groups = [model.blocks]
            if variant == 'foreign': groups = [self.model().blocks]
            if variant == 'duplicate': groups = [model.blocks, model.blocks]
            if variant == 'empty': groups = []
            before = copy.deepcopy(model.state_dict())
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                lora.inject_attention_lora(model, groups)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertEqual(set(model.state_dict()), set(before))
            for name, value in before.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        model = self.model(); adapters = lora.inject_attention_lora(model, [model.blocks])
        with self.assertRaises(ValueError): lora.inject_attention_lora(model, [model.blocks])
        self.assertEqual(len(adapters), 6)

    def test_usage_guard_detects_weight_bypass_and_resets(self):
        model = self.model((2,))
        adapters = lora.inject_attention_lora(model, [model.blocks])
        x = torch.randn(2, 4)
        modules = list(adapters.values())
        modules[0](x)
        torch.nn.functional.linear(x, modules[1].weight, modules[1].bias)
        with self.assertRaisesRegex(RuntimeError, 'blocks.0.attention.proj'):
            lora.assert_adapter_usage(adapters)
        modules[1](x)
        lora.assert_adapter_usage(adapters, reset=True)
        self.assertTrue(all(m.calls == 0 for m in adapters.values()))
        with self.assertRaises(RuntimeError): lora.assert_adapter_usage(adapters)
        with self.assertRaises(ValueError): lora.assert_adapter_usage({})

    def test_state_roundtrip_preserves_adapted_output(self):
        first = self.model((2,)); second = copy.deepcopy(first)
        one = lora.inject_attention_lora(first, [first.blocks]); two = lora.inject_attention_lora(second, [second.blocks])
        with torch.no_grad():
            for module in one.values(): module.lora_B.fill_(.25)
        second.load_state_dict(first.state_dict(), strict=True)
        x = torch.randn(2, 4)
        for name in one:
            torch.testing.assert_close(one[name](x), two[name](x), rtol=0, atol=0)

    def test_shared_projection_is_rejected_before_mutation(self):
        model = self.model((2, 2))
        model.blocks[1].attention.qkv = model.blocks[0].attention.qkv
        with self.assertRaises(ValueError):
            lora.inject_attention_lora(model, [model.blocks])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertIsInstance(model.blocks[0].attention.qkv, nn.Linear)
