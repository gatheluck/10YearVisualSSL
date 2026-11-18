"""Behavioral contracts for the explicitly versioned adaptation recipes."""

import importlib
import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace


class RevisedRecipeTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('downstream.revised_recipe')

    def test_imagenet_lora_canonical_values_and_immutable_identity(self):
        recipe = self.api().resolve_recipe('imagenet', 'lora')
        self.assertEqual(recipe.profile, 'b5x_lora_dpt_v2')
        self.assertEqual(recipe.source_protocol_id, 'B5X_LORA_LP_v2')
        self.assertEqual((recipe.task, recipe.track), ('imagenet', 'lora'))
        self.assertEqual((recipe.epochs, recipe.effective_batch,
                          recipe.reference_batch), (100, 256, 256))
        self.assertEqual((recipe.head_lr, recipe.adapter_lr), (.001, .000125))
        self.assertEqual(recipe.warmup_epochs, 5)
        self.assertEqual(recipe.weight_decay, .05)
        self.assertEqual(recipe.betas, (.9, .999))
        self.assertEqual(recipe.clip_norm, 1.)
        self.assertEqual((recipe.rank, recipe.alpha), (16, 16.))
        self.assertEqual((recipe.start_lr, recipe.min_lr), (1e-6, 1e-6))
        self.assertIsNone(recipe.dpt_hidden)
        with self.assertRaises(FrozenInstanceError):
            recipe.epochs = 1

    def test_dense_tracks_have_distinct_head_rates_and_trainables(self):
        lora = self.api().resolve_recipe('ade20k', 'lora')
        dpt = self.api().resolve_recipe('ade20k', 'dpt')
        for recipe in (lora, dpt):
            self.assertEqual((recipe.epochs, recipe.effective_batch,
                              recipe.reference_batch), (20, 8, 8))
            self.assertEqual(recipe.warmup_epochs, 1)
        self.assertEqual((lora.head_lr, lora.adapter_lr), (.001, .0001))
        self.assertEqual((dpt.head_lr, dpt.adapter_lr), (.0005, None))
        self.assertEqual(dpt.source_protocol_id, 'B5X_DPT_LP_v2')
        self.assertEqual(dpt.dpt_hidden, 512)
        self.assertEqual((dpt.rank, dpt.alpha), (None, None))

    def test_batch_scaling_preserves_reference_identity_and_absolute_floor(self):
        for task, track in [('imagenet', 'lora'), ('ade20k', 'lora'),
                            ('ade20k', 'dpt')]:
            canonical = self.api().resolve_recipe(task, track)
            recipe = self.api().resolve_recipe(
                task, track, effective_batch=canonical.reference_batch * 2)
            self.assertEqual(recipe.head_lr, 2 * canonical.head_lr)
            if canonical.adapter_lr is not None:
                self.assertEqual(recipe.adapter_lr, 2 * canonical.adapter_lr)
            self.assertEqual(recipe.reference_batch, canonical.reference_batch)
            self.assertEqual(recipe.epochs, canonical.epochs)
            self.assertEqual(recipe.min_lr, 1e-6)
            self.assertEqual(recipe.start_lr, 1e-6)

    def test_unconfirmed_routes_and_invalid_batches_fail_closed(self):
        for task, track in [('imagenet', 'dpt'), ('nyuv2', 'dpt'),
                            ('ssv2', 'lora'), ('ade20k', 'LORA'),
                            ('ade20k', 'finetune')]:
            with self.subTest(task=task, track=track), self.assertRaises(ValueError):
                self.api().resolve_recipe(task, track)
        for batch in (0, -1, True, 1.5, '8'):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.api().resolve_recipe('ade20k', 'lora', effective_batch=batch)

    def test_cosine_uses_zero_based_updates_and_absolute_endpoints(self):
        cosine = self.api().cosine_lr
        self.assertEqual(cosine(0, 100, 5, .001), 1e-6)
        self.assertAlmostEqual(cosine(2, 100, 5, .001), .0004006)
        self.assertAlmostEqual(cosine(5, 100, 5, .001), .001)
        self.assertAlmostEqual(cosine(50, 100, 5, .001),
                               1e-6 + .5 * (.001 - 1e-6)
                               * (1 + math.cos(math.pi * 45 / 95)))
        self.assertGreater(cosine(99, 100, 5, .001), 1e-6)
        self.assertEqual(cosine(100, 100, 5, .001), 1e-6)
        self.assertEqual(cosine(101, 100, 5, .001), 1e-6)
        self.assertAlmostEqual(cosine(0, 100, 0, .001), .001)
        self.assertAlmostEqual(cosine(0, 100, 5, .001, 2e-6, 3e-6), 3e-6)

    def test_scheduler_applies_each_groups_own_rate_before_update(self):
        api = self.api()
        recipe = api.resolve_recipe('imagenet', 'lora')
        groups = [{'base_lr': recipe.head_lr, 'lr': 10},
                  {'base_lr': recipe.adapter_lr, 'lr': 20}]
        optimizer = SimpleNamespace(param_groups=groups)
        api.set_learning_rates(optimizer, recipe, step=0, steps_per_epoch=2)
        self.assertEqual([g['lr'] for g in groups], [1e-6, 1e-6])
        api.set_learning_rates(optimizer, recipe, step=5, steps_per_epoch=2)
        self.assertEqual([g['lr'] for g in groups], [.0005005000000000001, .000063])
        api.set_learning_rates(optimizer, recipe, step=10, steps_per_epoch=2)
        self.assertAlmostEqual(groups[0]['lr'], recipe.head_lr)
        self.assertAlmostEqual(groups[1]['lr'], recipe.adapter_lr)
        api.set_learning_rates(optimizer, recipe, step=199, steps_per_epoch=2)
        self.assertGreater(groups[0]['lr'], 1e-6)
        self.assertGreater(groups[1]['lr'], 1e-6)
        api.set_learning_rates(optimizer, recipe, step=200, steps_per_epoch=2)
        self.assertEqual([g['lr'] for g in groups], [1e-6, 1e-6])
        self.assertEqual([g['base_lr'] for g in groups], [.001, .000125])

    def test_warmup_matches_reference_float_operation_order(self):
        cosine = self.api().cosine_lr
        self.assertEqual(cosine(1, 100, 5, .000125), 2.58e-05)
        self.assertEqual(cosine(3, 100, 5, .001), .0006004000000000002)

    def test_invalid_schedule_is_rejected_without_partially_mutating_groups(self):
        api = self.api()
        recipe = api.resolve_recipe('ade20k', 'lora')
        for bad_group in ({'lr': 7}, {'base_lr': float('nan'), 'lr': 7},
                          {'base_lr': -1., 'lr': 7}):
            optimizer = SimpleNamespace(param_groups=[{'base_lr': .001, 'lr': 3},
                                                      bad_group])
            with self.assertRaises((ValueError, KeyError)):
                api.set_learning_rates(optimizer, recipe, step=0, steps_per_epoch=2)
            self.assertEqual([g['lr'] for g in optimizer.param_groups], [3, 7])
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                api.set_learning_rates(SimpleNamespace(param_groups=[]), recipe,
                                       step=0, steps_per_epoch=steps)

    def test_invalid_scalar_schedule_arguments_are_rejected(self):
        cosine = self.api().cosine_lr
        for step, total, warmup, base, floor, start in [
            (-1, 100, 5, .001, 1e-6, 1e-6),
            (0, 0, 0, .001, 1e-6, 1e-6),
            (0, 100, -1, .001, 1e-6, 1e-6),
            (0, 100, 100, .001, 1e-6, 1e-6),
            (0, 100, 5, float('inf'), 1e-6, 1e-6),
            (0, 100, 5, .001, float('nan'), 1e-6),
            (0, 100, 5, .001, 1e-6, -1),
            (True, 100, 5, .001, 1e-6, 1e-6),
            (0, 100.5, 5, .001, 1e-6, 1e-6),
        ]:
            with self.subTest(args=(step, total, warmup)), self.assertRaises(ValueError):
                cosine(step, total, warmup, base, floor, start)


if __name__ == '__main__':
    unittest.main()
