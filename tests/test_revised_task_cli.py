"""Behavioral contracts for the explicit revised task entry point."""
import copy
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset


@unittest.skipUnless(HAVE, 'torch required')
class RevisedCLI(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.assertIsNotNone(importlib.util.find_spec('downstream.revised_tasks'),
                             'revised task CLI is not implemented')
        self.api = importlib.import_module('downstream.revised_tasks')

    def config(self, task='imagenet', track='lora'):
        return {'profile': 'b5x_lora_dpt_v2', 'task': task, 'track': track, 'seed': 7,
                    'device': 'cpu', 'data_root': 'private-data',
                    'backbone': {'kind': 'fixture', 'arch': 'fixture'},
                    'diagnostic': {'epochs': 1, 'batch_size': 2, 'max_steps': 1, 'dpt_hidden': 4}}

    def backbone(self):
        class Backbone(nn.Module):
            out_channels = global_channels = 4
            def __init__(self):
                super().__init__()
                block = nn.Module()
                block.attn = nn.ModuleDict({'qkv': nn.Linear(3, 4), 'proj': nn.Linear(4, 4)})
                self.blocks = nn.ModuleList([block])
                self.dropout = nn.Dropout(.8)
                self.enabled = False
            def adaptation_block_groups(self): return [self.blocks]
            def enable_adaptation_grad(self): self.enabled = True; self.eval()
            def forward_features(self, x):
                block = self.blocks[0].attn
                y = block['proj'](torch.tanh(block['qkv'](x.permute(0, 2, 3, 1))))
                return self.dropout(y.permute(0, 3, 1, 2))
            def classification_features(self, x, *, adaptation):
                assert adaptation == 'frozen'
                return nn.functional.normalize(self.forward_features(x).mean((2, 3)), dim=-1)
            def forward_dpt_features(self, x): return [self.forward_features(x)]*4
        return Backbone()

    def data(self, task):
        x = torch.randn(4, 3, 4, 4)
        labels = torch.tensor([0, 1, 0, 1]) if task == 'imagenet' else torch.arange(64).reshape(4, 4, 4) % 3
        ds = TensorDataset(x, labels)
        ds.class_to_idx = {'a': 0, 'b': 1}
        return ds, ds

    def test_config_rejects_unknown_or_implicit_recipe_changes(self):
        cfg = self.config()
        self.api.validate_config(cfg)
        for change in ({'profile': 'legacy'}, {'track': 'finetune'}, {'task': 'unknown'},
                       {'epochs': 1}, {'seed': True}, {'diagnostic': {'batch_size': 0}},
                       {'diagnostic': {'unknown': 1}}, {'device': 'imaginary'},
                       {'data_root': ''}, {'backbone': None}, {'workers': -1},
                       {'diagnostic': {'epochs': 101}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.api.validate_config(dict(cfg, **change))
        with self.assertRaises(ValueError): self.api.validate_config(self.config('imagenet', 'dpt'))

    def test_checkpoint_identity_tracks_bytes_without_exposing_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'config.json').write_text('{"hidden":4}')
            weights = root/'private-name.safetensors'; weights.write_bytes(b'original')
            identity = self.api.checkpoint_identity(str(root))
            self.assertEqual(identity, self.api.checkpoint_identity(str(root)))
            self.assertEqual(len(identity), 64)
            weights.write_bytes(b'changed')
            self.assertNotEqual(identity, self.api.checkpoint_identity(str(root)))
            with self.assertRaises(ValueError): self.api.checkpoint_identity(str(root/'missing'))

    def test_heads_preserve_global_readout_and_adapter_gradients(self):
        from downstream.revised_recipe import resolve_recipe
        x = torch.randn(2, 3, 4, 4)
        for task in ('imagenet', 'ade20k'):
            bb = self.backbone()
            old = {k: v.clone() for k, v in bb.state_dict().items()}
            model, adapters = self.api.build_model(bb, resolve_recipe(task, 'lora'), x, seed=7)
            model.train()
            self.assertFalse(bb.training)
            self.assertTrue(model.head.training)
            features = (bb.classification_features(x, adaptation='frozen') if task == 'imagenet'
                        else bb.forward_features(x))
            expected = model.head(features.float())
            actual = model(x) if task == 'imagenet' else model(x, (4, 4))
            torch.testing.assert_close(actual, expected)
            self.assertGreater(model.head.weight.std().item(), .005)
            torch.testing.assert_close(model.head.bias, torch.zeros_like(model.head.bias))
            actual.square().mean().backward()
            self.assertTrue(all(layer.lora_B.grad.abs().sum() > 0 for layer in adapters.values()))
            self.assertTrue(all(p.grad is None for p in bb.parameters() if not p.requires_grad))
            self.assertEqual(len(old), 4)

    def test_classification_initialization_matches_explicit_source_seed(self):
        from downstream.revised_recipe import resolve_recipe
        model, _ = self.api.build_model(self.backbone(), resolve_recipe('imagenet', 'lora'),
                                        torch.randn(1, 3, 4, 4), seed=7)
        torch.manual_seed(7)
        expected = nn.Linear(4, 1000)
        nn.init.normal_(expected.weight, std=.01)
        nn.init.zeros_(expected.bias)
        torch.testing.assert_close(model.head.weight, expected.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.head.bias, expected.bias, rtol=0, atol=0)

    def test_source_loader_seed_drop_last_and_fixed_schedule_horizon(self):
        from downstream import revised_engine
        original = revised_engine.train_epoch
        for task in ('imagenet', 'ade20k'):
            cfg = self.config(task)
            x = torch.randn(5, 3, 4, 4)
            y = torch.arange(5)%2 if task == 'imagenet' else torch.arange(80).reshape(5, 4, 4)%3
            ds = TensorDataset(x, y); ds.class_to_idx = {'a': 0, 'b': 1}
            observed = []
            def inspect(model, loader, optimizer, recipe, step, steps_per_epoch, device, observed=observed, **kw):
                observed.append((loader.drop_last, len(loader), loader.generator.initial_seed(),
                                 recipe.epochs, loader.sampler.__class__.__name__))
                return original(model, loader, optimizer, recipe, step, steps_per_epoch, device, **kw)
            with tempfile.TemporaryDirectory() as td, \
                 patch.object(self.api, 'build_frozen_backbone', return_value=self.backbone()), \
                 patch.object(self.api, 'build_datasets', return_value=(ds, ds)), \
                 patch.object(revised_engine, 'train_epoch', side_effect=inspect):
                self.api.run(cfg, Path(td)/'out')
            self.assertEqual(observed, [(True, 2, 7000 if task == 'imagenet' else 7,
                                         100 if task == 'imagenet' else 20,
                                         'DistributedSampler' if task == 'imagenet' else 'RandomSampler')])

    def test_three_routes_write_truthful_results_and_never_overwrite(self):
        for task, track in [('imagenet', 'lora'), ('ade20k', 'lora'), ('ade20k', 'dpt')]:
            cfg = self.config(task, track)
            with tempfile.TemporaryDirectory() as td:
                out = Path(td)/'out'
                with patch.object(self.api, 'build_frozen_backbone', return_value=self.backbone()), \
                     patch.object(self.api, 'build_datasets', return_value=self.data(task)):
                    result = self.api.run(cfg, out)
                self.assertEqual(result['updates'], 1)
                self.assertFalse(result['record_value'])
                self.assertFalse(result['training_complete'])
                self.assertEqual(result['status'], 'diagnostic')
                saved = json.loads((out/'result.json').read_text())
                self.assertEqual(saved, result)
                self.assertNotIn('private-data', (out/'result.json').read_text())
                state = torch.load(out/'trainable.pt', weights_only=True)
                self.assertTrue(any(k.startswith('head.') for k in state))
                self.assertTrue(all(k.startswith('head.') or k.endswith(('lora_A', 'lora_B')) for k in state))
                before = (out/'result.json').read_bytes()
                with self.assertRaises(FileExistsError): self.api.run(cfg, out)
                self.assertEqual(before, (out/'result.json').read_bytes())

    def test_reference_run_rejects_fixture_population_and_bad_class_mapping(self):
        cfg = self.config(); del cfg['diagnostic']
        with tempfile.TemporaryDirectory() as td, \
             patch.object(self.api, 'build_datasets', return_value=self.data('imagenet')), \
             self.assertRaisesRegex(ValueError, 'reference|population|released'):
                self.api.run(cfg, Path(td)/'out')
        cfg = self.config()
        train, val = self.data('imagenet'); val = copy.copy(val); val.class_to_idx = {'b': 0, 'a': 1}
        with tempfile.TemporaryDirectory() as td, \
             patch.object(self.api, 'build_datasets', return_value=(train, val)), \
             self.assertRaisesRegex(ValueError, 'class mapping'):
                self.api.run(cfg, Path(td)/'out')
