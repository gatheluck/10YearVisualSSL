"""Exact portable continuation and preflight integrity for revised task runners."""

import copy
import importlib
import importlib.util
import json
import os
import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) for name in ('torch', 'numpy'))


def distributed_resume_rank(rank, directory):
    from datetime import timedelta

    import numpy as np
    import torch
    import torch.distributed as dist

    from downstream.extended_resume import _cpu, _rng

    os.environ.update(WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/meeting',
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        fixture = RevisedResumeTests()
        fixture.root, fixture.serial = Path(directory), 0
        def start():
            value = fixture.fresh(batchnorm=False)
            random.seed(43 + rank); np.random.seed(43 + rank); torch.manual_seed(43 + rank)
            value.loader.generator.manual_seed(29 + rank)
            return value
        full = fixture.advance(start(), 3)
        first = start(); fixture.advance(first, 1)
        resumed = fixture.fresh(cfg={'resume': str(first.path)}, batchnorm=False)
        fixture.equal(full, fixture.advance(resumed, 3))
        blob = torch.load(first.path, weights_only=True)
        blob['rng'][1]['state']['torch'] = torch.zeros(1, dtype=torch.uint8)
        bad = Path(directory) / 'corrupt-rank.pt'
        if rank == 0: torch.save(blob, bad)
        dist.barrier()
        current = fixture.fresh(batchnorm=False)
        state, optimizer, rng = (_cpu(current.model.state_dict()), _cpu(current.opt.state_dict()),
                                 _cpu(_rng(current.loader, current.context.device)))
        try:
            current.load(str(bad))
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        else:
            raise AssertionError('corrupt other-rank RNG was accepted')
        fixture.equal(state, current.model.state_dict())
        fixture.equal(optimizer, current.opt.state_dict())
        fixture.equal(rng, _rng(current.loader, current.context.device))
        Path(directory, f'rank{rank}.json').write_text(json.dumps({'exact': True, 'error': error}))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(HAVE, 'Torch and NumPy required')
class RevisedResumeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0

    def api(self):
        return importlib.import_module('downstream.revised_resume')

    def fresh(self, *, cfg=None, membership=None, adapters=True, algorithm='AdamW', batchnorm=True,
              immutable_identity='a' * 64, drop_last=False):
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from downstream.adaptation_runtime import adapter_parameter_groups
        from downstream.extended_distributed import Session
        from downstream.lora import LowRankLinear
        from downstream.revised_recipe import resolve_recipe

        random.seed(31); np.random.seed(31); torch.manual_seed(31)
        model = nn.Module()
        model.backbone = (LowRankLinear(nn.Linear(2, 2), rank=2, alpha=2)
                          if adapters else nn.Linear(2, 2).requires_grad_(False))
        model.head = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2) if batchnorm else nn.Identity())
        model.head.register_parameter('unused', nn.Parameter(torch.ones(2)))
        registered = {'': model.backbone} if adapters else {}
        recipe = resolve_recipe('imagenet', 'lora')
        groups = adapter_parameter_groups(model.named_parameters(), head_lr=recipe.head_lr,
                                          adapter_lr=recipe.adapter_lr, weight_decay=recipe.weight_decay)
        opt = (torch.optim.AdamW(groups, betas=recipe.betas) if algorithm == 'AdamW'
               else torch.optim.SGD(groups, momentum=.9))
        loader = DataLoader(TensorDataset(torch.arange(8).float().reshape(4, 2)),
                            batch_size=2, shuffle=True, drop_last=drop_last,
                            generator=torch.Generator().manual_seed(17))
        self.serial += 1
        out = self.root / str(self.serial); out.mkdir(exist_ok=True)
        config = {'profile': 'b5x_lora_dpt_v2', 'task': 'imagenet', 'track': 'lora',
                      'device': 'cpu', 'workers': 0, 'seed': 31, 'diagnostic': {'epochs': 3},
                      'backbone': {'kind': 'fixture'}, 'continuation': True}
        config.update(cfg or {})
        value = self.api().RevisedContinuation(
            config, membership or {'train': 'fixture-train', 'val': 'fixture-val'},
            model, opt, recipe, 2, loader, Session(torch.device('cpu')), out,
            registered, backbone=model.backbone, immutable_identity=immutable_identity)
        return value

    def equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left: self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right): self.equal(a, b)
        else: self.assertEqual(left, right)

    def advance(self, value, target):
        import numpy as np
        import torch

        from downstream.revised_recipe import set_learning_rates
        for epoch in range(value.start_epoch, target):
            losses = []
            for (data,) in value.loader:
                set_learning_rates(value.opt, value.recipe, value.step, 2)
                data = data + random.random() + float(np.random.rand()) + torch.rand_like(data)
                loss = value.model.head(value.model.backbone(data)).square().mean()
                value.opt.zero_grad(set_to_none=True); loss.backward()
                if value.context.world > 1:
                    for p in value.model.parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad)
                            p.grad.div_(value.context.world)
                value.opt.step()
                scalar = loss.detach().clone()
                if value.context.world > 1:
                    torch.distributed.all_reduce(scalar)
                    scalar.div_(value.context.world)
                value.step += 1; losses.append(float(scalar))
            value.losses.append({'epoch': epoch + 1, 'loss': sum(losses) / 2, 'updates': 2})
            value.save(epoch + 1, value.step, value.losses)
        return torch.load(value.path, weights_only=True)

    def test_exact_continuation_preserves_adapters_buffers_optimizer_and_all_rng(self):
        import torch
        for adapters, algorithm in [(True, 'AdamW'), (False, 'AdamW'), (True, 'SGD')]:
            with self.subTest(adapters=adapters, algorithm=algorithm):
                full = self.advance(self.fresh(adapters=adapters, algorithm=algorithm), 3)
                initial = self.fresh(adapters=adapters, algorithm=algorithm)
                first = self.advance(initial, 1)
                self.assertEqual(first['step'], 2)
                self.assertEqual(first['epoch'], 1)
                self.assertTrue(any('running_mean' in key for key in first['model']))
                self.assertFalse(any('.base.' in key for key in first['model']))
                if adapters:
                    self.assertIn('backbone.lora_A', first['model'])
                    self.assertIn('backbone.lora_B', first['model'])
                restored = self.fresh(cfg={'resume': str(initial.path)}, adapters=adapters,
                                      algorithm=algorithm)
                self.assertEqual((restored.start_epoch, restored.step), (1, 2))
                actual = self.advance(restored, 3)
                self.equal(full, actual)
                self.assertFalse(restored.report()['legacy_native_compatible'])
                self.assertIsNotNone(restored.report()['source_sha256'])
                resumed_same = self.fresh(cfg={'resume': str(restored.path), 'stop_after_epoch': 3},
                                          adapters=adapters, algorithm=algorithm)
                with patch.object(resumed_same.opt, 'step', side_effect=AssertionError('repeated update')):
                    resumed_same.save(3, resumed_same.step, resumed_same.losses)
                self.equal(actual, torch.load(resumed_same.path, weights_only=True))

    def test_identity_binds_membership_frozen_weights_and_registered_adapter_exclusion(self):
        original = self.fresh()
        payload = self.advance(original, 1)
        changed = self.fresh(membership={'train': 'changed', 'val': 'fixture-val'})
        with self.assertRaisesRegex(ValueError, 'identity'): changed.load(str(original.path))
        current = self.fresh()
        self.assertEqual(current.identity, original.identity)
        with __import__('torch').no_grad(): current.model.backbone.base.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'identity'): current.load(str(original.path))
        self.assertEqual(payload['identity'], original.identity)
        changed_bytes = self.fresh(immutable_identity='b' * 64)
        with self.assertRaisesRegex(ValueError, 'identity'): changed_bytes.load(str(original.path))
        with self.assertRaisesRegex(ValueError, 'checkpoint identity'): self.fresh(immutable_identity=None)
        decoy = self.fresh()
        decoy.backbone.register_buffer('lora_A_decoy', __import__('torch').ones(2))
        with self.assertRaisesRegex(ValueError, 'identity'): decoy.load(str(original.path))

    def test_model_implementation_bytes_are_bound_without_persisting_paths(self):
        source = self.root / 'provider.py'
        source.write_text('implementation version one')
        with patch('inspect.getsourcefile', return_value=str(source)):
            original = self.fresh(); self.advance(original, 1)
            source.write_text('implementation version two')
            changed = self.fresh()
            with self.assertRaisesRegex(ValueError, 'identity'):
                changed.load(str(original.path))
        self.assertNotIn(str(source), str(changed.identity_base))

    def test_actual_loader_policy_changes_are_not_hidden_by_identical_config(self):
        original = self.fresh(); self.advance(original, 1)
        changed = self.fresh(drop_last=True)
        with self.assertRaisesRegex(ValueError, 'identity'): changed.load(str(original.path))

    def test_partial_boundary_workers_and_step_caps_are_rejected(self):
        for cfg in [{'workers': 1}, {'diagnostic': {'epochs': 3, 'max_steps': 1}},
                    {'stop_after_epoch': 0}, {'stop_after_epoch': 4}]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError): self.fresh(cfg=cfg)
        value = self.fresh()
        for epoch, step, losses in [(0, 0, []), (1, 1, [{'epoch': 1, 'loss': 1., 'updates': 1}]),
                                    (1, 2, []), (1, 2, [{'epoch': 1, 'loss': float('nan'), 'updates': 2}])]:
            with self.subTest(epoch=epoch, step=step), self.assertRaises(ValueError):
                value.save(epoch, step, losses)
        self.assertFalse(value.path.exists())

    def test_corrupt_payload_is_rejected_before_any_live_state_mutation(self):
        import torch

        from downstream.extended_resume import _cpu, _rng
        original = self.fresh(); blob = self.advance(original, 1)
        def missing_adapter(b): b['model'].pop('backbone.lora_A')
        def frozen_tensor(b): b['model']['backbone.base.weight'] = torch.ones(2, 2)
        def nonfinite_tensor(b): b['model']['head.0.weight'].fill_(float('nan'))
        def wrong_lr(b): b['optimizer']['param_groups'][0]['lr'] = .3
        def nonfinite_slot(b): next(iter(b['optimizer']['state'].values()))['exp_avg'].fill_(float('inf'))
        def wrong_rng(b): b['rng'][0]['state']['torch'] = torch.zeros(1, dtype=torch.uint8)
        def wrong_rank(b): b['rng'][0]['rank'] = 1
        def wrong_slot(b): next(iter(b['optimizer']['state'].values()))['step'].fill_(3)
        def wrong_count(b): b['step'] = 3
        def bool_world(b): b['world'] = True
        def float_horizon(b): b['horizon'] = 100.
        def float_steps(b): b['steps_per_epoch'] = 2.
        def bool_epoch(b): b['losses'][0]['epoch'] = True
        def bool_loss(b): b['losses'][0]['loss'] = True
        def float_updates(b): b['losses'][0]['updates'] = 2.
        def empty_losses(b): b['losses'] = []
        def legacy_format(b): b['format'] = 'native_resume'
        def extra_field(b): b['unexpected'] = 1
        def forged_count_and_rates(b):
            from downstream.revised_recipe import recipe_learning_rate
            b['step'] = 3
            for group in b['optimizer']['param_groups']:
                group['lr'] = recipe_learning_rate(original.recipe, 2, 2, group['base_lr'])
        for corrupt in [missing_adapter, frozen_tensor, nonfinite_tensor, wrong_lr,
                        nonfinite_slot, wrong_rng, wrong_rank, wrong_slot, wrong_count,
                        bool_world, float_horizon, float_steps, bool_epoch, bool_loss, float_updates,
                        empty_losses, legacy_format, extra_field, forged_count_and_rates]:
            with self.subTest(corrupt=corrupt.__name__):
                current = self.fresh(); modified = copy.deepcopy(blob); corrupt(modified)
                path = self.root / 'bad.pt'; torch.save(modified, path)
                state = _cpu(current.model.state_dict()); opt = _cpu(current.opt.state_dict())
                rng = _cpu(_rng(current.loader, current.context.device))
                with self.assertRaisesRegex(ValueError, 'checkpoint|RNG'): current.load(str(path))
                self.equal(current.model.state_dict(), state)
                self.equal(current.opt.state_dict(), opt)
                self.equal(_rng(current.loader, current.context.device), rng)

    def test_atomic_failure_keeps_previous_checkpoint_and_cleans_temporary(self):
        import torch
        value = self.fresh(); self.advance(value, 1)
        before = value.path.read_bytes()
        real = torch.save
        def fail(blob, path):
            real(blob, path); raise OSError('interrupted write')
        with patch('downstream.extended_resume.torch.save', side_effect=fail), self.assertRaises(OSError):
            value.save(1, value.step, value.losses)
        self.assertEqual(before, value.path.read_bytes())
        self.assertFalse(value.path.with_suffix('.tmp').exists())

    def test_ordered_membership_binds_file_bytes_labels_and_membership(self):
        first = self.root / 'a'; second = self.root / 'b'
        first.write_bytes(b'a'); second.write_bytes(b'b')
        rows = [{'files': [first], 'metadata': {'label': 1}},
                {'files': [second], 'metadata': {'label': 2}}]
        digest = self.api().hash_membership(rows)
        self.assertEqual(digest['samples'], 2)
        self.assertNotIn(str(self.root), json.dumps(digest))
        self.assertNotEqual(digest, self.api().hash_membership(rows[::-1]))
        rows[0]['metadata']['label'] = 2
        self.assertNotEqual(digest, self.api().hash_membership(rows))
        rows[0]['metadata']['label'] = 1; first.write_bytes(b'changed')
        self.assertNotEqual(digest, self.api().hash_membership(rows))
        with self.assertRaises(FileNotFoundError):
            self.api().hash_membership([{'files': [self.root / 'missing'], 'metadata': 0}])

    def test_two_rank_continuation_and_other_rank_corruption_are_atomic(self):
        import torch
        if not torch.distributed.is_gloo_available(): self.skipTest('Gloo unavailable')
        processes = torch.multiprocessing.spawn(distributed_resume_rank, args=(str(self.root),),
                                                nprocs=2, join=False)
        try:
            deadline = time.monotonic() + 60
            while not processes.join(timeout=1):
                if time.monotonic() > deadline: self.fail('rank continuation exceeded 60 seconds')
        finally:
            for process in processes.processes:
                if process.is_alive(): process.terminate()
            for process in processes.processes: process.join(timeout=5)
        for rank in range(2):
            result = json.loads(Path(self.root, f'rank{rank}.json').read_text())
            self.assertTrue(result['exact'])
            self.assertIn('RNG', result['error'])


if __name__ == '__main__':
    unittest.main()
