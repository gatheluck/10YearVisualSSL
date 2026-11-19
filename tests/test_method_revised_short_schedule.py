"""Author-declared short schedules are explicit and provider-scoped."""
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE, 'torch required for provider discovery')
class ShortScheduleTests(unittest.TestCase):
    def backbone(self):
        return {'kind': 'dinov3_hf', 'arch': 'released', 'encoder': '/local/weights'}

    def resolve(self, task, schedule, **kwargs):
        from downstream.revised_recipe import resolve_recipe
        return resolve_recipe(task, 'lora', schedule=schedule, backbone=kwargs or self.backbone())

    def test_author_schedule_and_cosine_span(self):
        from downstream.revised_recipe import recipe_learning_rate
        for task, epochs, schedule in (
                ('imagenet', 50, 'B5X_LORA_LP_50EP_v2'),
                ('ssv2', 24, 'B5X_LORA_LP_24EP_v2')):
            recipe = self.resolve(task, schedule)
            self.assertEqual((recipe.epochs, recipe.source_protocol_id), (epochs, schedule))
            self.assertEqual(recipe.warmup_epochs, 5)
            self.assertEqual(recipe_learning_rate(recipe, epochs * 3, 3, recipe.head_lr), 1e-6)
            self.assertGreater(recipe_learning_rate(recipe, epochs * 3 // 2, 3, recipe.head_lr), 1e-6)

    def test_other_provider_track_task_or_fixture_is_not_admitted(self):
        from downstream.revised_recipe import resolve_recipe
        for change in ({'kind': 'siglip2_g'}, {'arch': 'fixture'}, {'encoder': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.resolve('imagenet', 'B5X_LORA_LP_50EP_v2', **(self.backbone() | change))
        for task, track, schedule in (('ade20k', 'lora', 'B5X_LORA_LP_50EP_v2'),
                                      ('ssv2', 'lora', 'B5X_LORA_LP_50EP_v2'),
                                      ('ade20k', 'dpt', 'B5X_LORA_LP_24EP_v2'),
                                      ('imagenet', 'lora', 'unknown')):
            with self.subTest(task=task, track=track), self.assertRaises(ValueError):
                resolve_recipe(task, track, schedule=schedule, backbone=self.backbone())

    def test_cli_accepts_named_schedule_and_enforces_selected_horizon(self):
        from downstream.revised_tasks import validate_config
        cfg = {'profile': 'b5x_lora_dpt_v2', 'task': 'imagenet', 'track': 'lora',
               'seed': 0, 'device': 'cpu', 'data_root': '/data', 'backbone': self.backbone(),
               'schedule': 'B5X_LORA_LP_50EP_v2', 'continuation': True, 'stop_after_epoch': 50}
        validate_config(cfg)
        with self.assertRaisesRegex(ValueError, 'schedule'):
            validate_config(cfg | {'schedule': None})
        with self.assertRaisesRegex(ValueError, 'epoch|horizon'):
            validate_config(cfg | {'stop_after_epoch': 51})
        with self.assertRaisesRegex(ValueError, 'epochs'):
            validate_config(cfg | {'diagnostic': {'epochs': 51}})

    def test_default_recipe_unchanged(self):
        from downstream.revised_recipe import resolve_recipe
        for task, epochs in (('imagenet', 100), ('ssv2', 50)):
            recipe = resolve_recipe(task, 'lora')
            self.assertEqual((recipe.epochs, recipe.source_protocol_id), (epochs, 'B5X_LORA_LP_v2'))

    def test_resume_rejects_changed_schedule_recipe_before_mutation(self):
        from tests.test_revised_resume import RevisedResumeTests
        fixture = RevisedResumeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        recipe = self.resolve('imagenet', 'B5X_LORA_LP_50EP_v2')
        with patch('downstream.revised_recipe.resolve_recipe', return_value=recipe):
            original = fixture.fresh()
        fixture.advance(original, 1)
        for changed_recipe in (replace(recipe, epochs=100),
                               replace(recipe, source_protocol_id='B5X_LORA_LP_v2')):
            with patch('downstream.revised_recipe.resolve_recipe', return_value=changed_recipe):
                changed = fixture.fresh()
            with self.assertRaisesRegex(ValueError, 'identity'):
                changed.load(str(original.path))
            self.assertEqual(changed.start_epoch, 0)

    def test_execution_records_selected_horizon_with_synthetic_model(self):
        from tests.test_revised_task_cli import RevisedCLI
        fixture = RevisedCLI()
        fixture.setUp()
        cfg = fixture.config() | {'schedule': 'B5X_LORA_LP_50EP_v2', 'backbone': self.backbone()}
        with tempfile.TemporaryDirectory() as td, \
             patch.object(fixture.api, 'build_frozen_backbone', return_value=fixture.backbone()), \
             patch.object(fixture.api, 'build_datasets', return_value=fixture.data('imagenet')), \
             patch.object(fixture.api, 'checkpoint_identity', return_value='a' * 64):
            result = fixture.api.run(cfg, Path(td) / 'out')
        self.assertEqual(result['recipe']['epochs'], 50)
        self.assertEqual(result['recipe']['source_protocol_id'], cfg['schedule'])
        self.assertFalse(result['record_value'])
