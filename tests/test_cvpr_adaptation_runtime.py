"""Training-facing adaptation behavior, without downloading model weights."""
import copy
import importlib.util
import unittest

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    from torch import nn

    from downstream.adaptation_runtime import (
        adapter_parameter_groups,
        check_adapter_effect,
    )
    from downstream.feature_taps import collect_block_taps, even_taps, tokens_to_grid
    from downstream.lora import LowRankLinear


@unittest.skipUnless(HAVE, 'torch required')
class AdaptationRuntime(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_separate_rates_and_decay_update_only_selected_parameters(self):
        model = nn.Module()
        model.encoder = LowRankLinear(nn.Linear(4, 4), rank=2, alpha=4)
        model.head = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2))
        groups = adapter_parameter_groups(model.named_parameters(), weight_decay=.05,
                                          head_lr=.001, adapter_lr=.0001)
        self.assertEqual([(g['kind'], g['weight_decay'], g['lr'], g['base_lr']) for g in groups],
                         [('head', .05, .001, .001), ('head', 0., .001, .001),
                          ('lora', .05, .0001, .0001)])
        parameters = [p for g in groups for p in g['params']]
        self.assertEqual({id(p) for p in parameters},
                         {id(p) for p in model.parameters() if p.requires_grad})
        base = model.encoder.weight.detach().clone()
        before = model.encoder.lora_B.detach().clone()
        optimizer = torch.optim.AdamW(groups)
        model.head(model.encoder(torch.randn(3, 4))).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(model.encoder.weight, base, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.encoder.lora_B, before))

    def test_group_refusals_and_norm_rule(self):
        p = nn.Parameter(torch.ones(2, 2))
        for names in ([('encoder.weight', p)], [('encoder.not_lora_A', p)],
                      [('headless.weight', p)], [('head.weight', p), ('head.alias', p)], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                adapter_parameter_groups(names, weight_decay=.05, head_lr=.01, adapter_lr=.001)
        groups = adapter_parameter_groups([('encoder.norm.lora_A', p)], weight_decay=.05,
                                          head_lr=.01, adapter_lr=.001)
        self.assertEqual(groups[0]['weight_decay'], 0.)
        for kwargs in ({'weight_decay': -1}, {'head_lr': 0}, {'adapter_lr': float('nan')}):
            settings = {'weight_decay': .05, 'head_lr': .01, 'adapter_lr': .001}
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                adapter_parameter_groups([('head.weight', p)], **settings)

    def test_effect_probe_restores_weights_rng_and_counters(self):
        layer = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        x = torch.randn(2, 4)
        layer.calls = 7
        state = copy.deepcopy(layer.state_dict())
        rng = torch.get_rng_state().clone()
        result = check_adapter_effect(lambda: layer(x), {'attention.qkv': layer})
        self.assertGreater(result['delta_perturbed'], 0.)
        self.assertEqual(result['delta_restored'], 0.)
        self.assertEqual(result['n_wrapped'], 1)
        for key, value in state.items():
            torch.testing.assert_close(layer.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(layer.calls, 7)

    def test_called_but_discarded_adapter_and_bypass_are_refused(self):
        layer = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        x = torch.randn(2, 4)
        def discarded():
            layer(x)
            return layer.base(x)
        for forward in (discarded, lambda: layer.base(x), lambda: layer(x) * float('nan')):
            with self.subTest(forward=forward), self.assertRaises(RuntimeError):
                check_adapter_effect(forward, {'attention.qkv': layer})
        with self.assertRaises(ValueError):
            check_adapter_effect(lambda: x, {})

    def test_partial_bypass_is_not_hidden_by_an_effective_adapter(self):
        first = LowRankLinear(nn.Linear(4, 4), rank=2, alpha=4)
        second = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        x = torch.randn(2, 4)
        with self.assertRaisesRegex(RuntimeError, 'second'):
            check_adapter_effect(lambda: second.base(first(x)), {'first': first, 'second': second})

    def test_empty_adapter_check_has_no_forward_side_effect(self):
        calls = []
        def forward():
            calls.append(True)
            return torch.ones(1)
        with self.assertRaises(ValueError):
            check_adapter_effect(forward, {})
        self.assertEqual(calls, [])

    def test_nonrepeatable_feature_is_not_evidence_of_adapter_effect(self):
        layer = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        calls = []
        def forward():
            calls.append(True)
            return layer(torch.ones(2, 4)) + len(calls)
        with self.assertRaisesRegex(RuntimeError, 'reliable effect'):
            check_adapter_effect(forward, {'attention': layer})

    def test_effect_probe_restores_after_perturbed_forward_error(self):
        layer = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        before = layer.lora_B.detach().clone()
        rng = torch.get_rng_state().clone()
        def forward():
            if torch.count_nonzero(layer.lora_B):
                raise RuntimeError('simulated forward failure')
            return layer(torch.ones(2, 4))
        with self.assertRaisesRegex(RuntimeError, 'simulated forward failure'):
            check_adapter_effect(forward, {'attention.qkv': layer})
        torch.testing.assert_close(layer.lora_B, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_effect_probe_restores_rng_consumed_by_forward(self):
        layer = LowRankLinear(nn.Linear(4, 3), rank=2, alpha=4)
        x = torch.randn(2, 4)
        rng = torch.get_rng_state().clone()
        def forward():
            return layer(x) + torch.rand(()) * 0
        check_adapter_effect(forward, {'attention.qkv': layer})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_four_taps_keep_order_gradients_and_remove_hooks(self):
        blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(8)])
        self.assertEqual(even_taps(8), [1, 3, 5, 7])
        self.assertEqual(even_taps(9), [2, 4, 6, 8])
        x = torch.randn(1, 4, 3, requires_grad=True)
        def forward():
            y = x
            for block in blocks:
                y = block(y)
            return y
        taps = collect_block_taps(blocks, [1, 3, 5, 7], forward)
        expected = []
        y = x
        for i, block in enumerate(blocks):
            y = block(y)
            if i in (1, 3, 5, 7):
                expected.append(y)
        self.assertEqual(len(taps), 4)
        for actual, reference in zip(taps, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        sum(t.sum() for t in taps).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(not b._forward_hooks for b in blocks))

    def test_tap_failure_does_not_leak_hooks_or_reuse_stale_values(self):
        blocks = nn.ModuleList([nn.Identity() for _ in range(4)])
        for indices in ([0, 1, 1, 3], [0, 1, 2, 4], [0, 1, 2], [-1, 0, 1, 2]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                collect_block_taps(blocks, indices, lambda: None)
        for n in (0, 3, True, 4.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                even_taps(n)
        with self.assertRaises(RuntimeError):
            collect_block_taps(blocks, [0, 1, 2, 3], lambda: blocks[0](torch.ones(1)))
        self.assertTrue(all(not b._forward_hooks for b in blocks))
        def broken():
            raise RuntimeError('forward failed')
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            collect_block_taps(blocks, [0, 1, 2, 3], broken)
        self.assertTrue(all(not b._forward_hooks for b in blocks))

    def test_grid_conversion_removes_prefix_preserves_patch_order(self):
        x = torch.arange(15.).reshape(1, 5, 3)
        actual = tokens_to_grid(x, 2, 2, prefix=1)
        torch.testing.assert_close(actual, x[:, 1:].transpose(1, 2).reshape(1, 3, 2, 2))
        for tensor, h, w, prefix in ((x, 3, 2, 1), (x[0], 2, 2, 1), (x, 2, 3, -1)):
            with self.subTest(h=h, w=w, prefix=prefix), self.assertRaises(ValueError):
                tokens_to_grid(tensor, h, w, prefix=prefix)

    def test_tuple_taps_and_existing_hooks_are_preserved(self):
        class TupleBlock(nn.Module):
            def forward(self, x):
                return x * 2, None
        blocks = nn.ModuleList([TupleBlock() for _ in range(4)])
        observed = []
        existing = blocks[0].register_forward_hook(lambda m, a, o: observed.append(o[0]))
        def forward():
            x = torch.ones(1)
            for block in blocks:
                x = block(x)[0]
        result = collect_block_taps(blocks, [0, 1, 2, 3], forward)
        self.assertEqual([x.item() for x in result], [2., 4., 8., 16.])
        self.assertEqual(len(observed), 1)
        self.assertEqual(len(blocks[0]._forward_hooks), 1)
        existing.remove()

    def test_fractional_grid_is_rejected_explicitly(self):
        with self.assertRaises(ValueError):
            tokens_to_grid(torch.ones(1, 4, 3), 2., 2)
