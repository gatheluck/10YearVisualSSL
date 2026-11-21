"""Audit full portable Figure 3 outputs at the reporting CLI."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from downstream import contract
from downstream.revised_recipe import resolve_recipe
from downstream.revised_reporting import reporting_fields


class PortableFigure(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def record(self, task, track, seed, *, alter=None, schedule=None):
        population, key, count, name = {
            'imagenet': ([1281167, 50000], 'top1', 'n', 'imagenet_top1'),
            'ssv2': ([168913, 24777], 'top1', 'n', 'ssv2_top1'),
            'coco': ([118287, 5000], 'bbox_ap', 'n_images', 'coco_map'),
            'ade20k': ([20210, 2000], 'miou', 'n', 'ade20k_miou'),
            'nyuv2': ([795, 654], 'rmse_unaligned', 'n_images', 'nyuv2_rmse'),
        }[task]
        out = self.root / f'{task}-{track}-{seed}'
        out.mkdir()
        backbone = {'kind': 'fixture'}
        if schedule is not None:
            backbone.update(kind='dinov3_hf', arch='released', encoder='weights')
        cfg = {'profile': 'b5x_lora_dpt_v2', 'task': task, 'track': track, 'seed': seed,
               'device': 'cuda', 'backbone': backbone, 'data_root': 'private'}
        if schedule is not None:
            cfg['schedule'] = schedule
        config = out.with_suffix('.json')
        config.write_text(json.dumps(cfg))
        recipe = json.loads(json.dumps(asdict(resolve_recipe(task, track, schedule=schedule, backbone=backbone))))
        value = .4 + seed / 100 if task == 'nyuv2' else 50 + seed
        metrics = {key: value, count: population[1]}
        if task == 'nyuv2':
            metrics.update(rmse=.3, scale_alignment='per-image median', n_input_images=654)
        if task == 'ade20k':
            metrics.update(n_pixels=92_007_036, n_classes_present=150, pixel_acc=78.5)
        result = {'profile': cfg['profile'], 'task': task, 'track': track, 'status': 'component_complete',
                      'recipe': recipe, 'source_protocol_id': recipe['source_protocol_id'],
                      'epochs_completed': recipe['epochs'], 'training_complete': True,
                      'source_populations': population[:], 'intended_populations': population[:],
                      'used_populations': population[:], 'missing_populations': [0, 0],
                      'metrics': metrics, 'world_size': 1, 'accumulation_steps': 1, 'precision': 'cuda_bf16', 'checkpoint_sha256': 'a'*64}
        steps = population[0] // recipe['effective_batch']
        result.update(updates=steps * recipe['epochs'], loss_log=[
            {'epoch': i + 1, 'updates': steps, 'loss': .25} for i in range(recipe['epochs'])])
        (out/'trainable.pt').write_bytes(b'fixture checkpoint bytes')
        result.update(reporting_fields(result))
        published = {name: value/100 if task == 'coco' else .3 if task == 'nyuv2' else value}
        if task in ('ssv2', 'imagenet'):
            metrics['top5'] = published[task + '_top5'] = 75.
        if task == 'coco':
            metrics['n_detections'] = 5235
            for suffix, contract_suffix in [('50', '_50'), ('75', '_75'), ('s', '_small'),
                                            ('m', '_medium'), ('l', '_large')]:
                metrics['bbox_ap' + suffix] = value
                published['coco_map' + contract_suffix] = value / 100
        if task == 'ade20k':
            published['ade20k_pixel_accuracy'] = metrics['pixel_acc']
        if alter:
            alter(result, published)
        (out/'result.json').write_text(json.dumps(result))
        contract.write_metrics(out, published, {key: key for key in published})
        contract.write_manifest(out, task=task, method_ref='revised-component', status='ok',
            config_sha256=contract.sha256_bytes(config.read_bytes()), started_at='start', finished_at='end',
            seed=seed, backbone=backbone)
        return {'seed': seed, 'config': str(config), 'out': str(out), 'exit_status': 0}

    def report(self, records):
        config = self.root / 'plan.json'
        out = self.root / 'report.json'
        out.unlink(missing_ok=True)
        config.write_text(json.dumps({'schema': 'portable_figure3_v1', 'runs': records,
                                          'expected_seeds': [0, 1, 2]}))
        subprocess.run([sys.executable, '-m', 'downstream.revised_reporting', '--config', str(config),
                        '--out', str(out)], check=True, capture_output=True)
        self.assertNotIn(str(self.root), out.read_text())
        result = json.loads(out.read_text())
        self.assertEqual(result['schema'], 'portable_figure3_v1')
        return result

    def test_all_seven_routes_use_paper_units_and_sample_sd(self):
        for task in ('imagenet', 'ssv2', 'coco', 'ade20k', 'nyuv2'):
            for track in (('lora', 'dpt') if task in ('ade20k', 'nyuv2') else ('lora',)):
                with self.subTest(task=task, track=track):
                    report = self.report([self.record(task, track, i) for i in (2, 0, 1)])
                    self.assertTrue(report['complete'])
                    self.assertAlmostEqual(report['paper_metric']['mean'], .41 if task=='nyuv2' else 51)
                    self.assertAlmostEqual(report['paper_metric']['std'], .01 if task=='nyuv2' else 1)
                    self.assertFalse(report['historical_run_verified'])

    def test_count_recipe_and_contract_drift_are_rejected(self):
        def count(result, published):
            result['used_populations'][1] -= 1
        def recipe(result, published):
            result['recipe']['head_lr'] *= 2
        def units(result, published):
            published['coco_map'] *= 100
        report = self.report([self.record('coco', 'lora', i, alter=fn)
                              for i, fn in enumerate((count, recipe, units))])
        self.assertEqual(report['completed_seeds'], [])
        self.assertEqual(len(report['invalid_runs']), 3)
        self.assertFalse(report['complete'])

    def test_full_classification_requires_consistent_top5(self):
        for task in ('imagenet', 'ssv2'):
            def swapped_units(_, contract_metrics, task=task):
                contract_metrics[task + '_top5'] /= 100

            def invalid_order(result, contract_metrics, task=task):
                result['metrics']['top5'] = contract_metrics[task + '_top5'] = 30

            def above_ceiling(result, contract_metrics, task=task):
                result['metrics']['top5'] = contract_metrics[task + '_top5'] = 101

            records = [self.record(task, 'lora', seed, alter=change)
                       for seed, change in enumerate((swapped_units, invalid_order, above_ceiling))]
            self.assertEqual(self.report(records)['completed_seeds'], [], task)

    def test_full_coco_requires_all_bbox_metrics_and_detection_count(self):
        def missing_area(_, published):
            del published['coco_map_small']

        def wrong_threshold(_, published):
            published['coco_map_75'] *= 100

        def impossible_count(result, _):
            result['metrics']['n_detections'] = -1

        records = [self.record('coco', 'lora', seed, alter=change)
                   for seed, change in enumerate((missing_area, wrong_threshold, impossible_count))]
        self.assertEqual(self.report(records)['completed_seeds'], [])

    def test_coco_rejects_consistent_but_invalid_auxiliary_values(self):
        def above_ceiling(result, published):
            result['metrics']['bbox_apl'] = 105.
            published['coco_map_large'] = 1.05

        def invalid_count(result, _):
            result['metrics']['n_detections'] = True

        def nonnumeric_score(result, published):
            result['metrics']['bbox_ap50'] = True
            published['coco_map_50'] = .01

        records = [self.record('coco', 'lora', seed, alter=change)
                   for seed, change in enumerate((above_ceiling, invalid_count, nonnumeric_score))]
        self.assertEqual(self.report(records)['completed_seeds'], [])

    def test_empty_coco_predictions_can_only_publish_zero_ap(self):
        def empty(result, published):
            result['metrics']['n_detections'] = 0
            for key in result['metrics']:
                if key.startswith('bbox_'):
                    result['metrics'][key] = 0.
            for key in published:
                published[key] = 0.

        def inconsistent(result, published):
            empty(result, published)
            result['metrics']['bbox_apm'] = 1.
            published['coco_map_medium'] = .01

        records = [self.record('coco', 'lora', 0, alter=empty),
                   self.record('coco', 'lora', 1, alter=inconsistent)]
        self.assertEqual(self.report(records)['completed_seeds'], [0])

    def test_absent_seeds_and_wrong_evaluated_count(self):
        def count(result, published):
            result['metrics']['n'] -= 1
        report = self.report([self.record('ade20k', 'dpt', 0),
                              self.record('ade20k', 'dpt', 1, alter=count)])
        self.assertEqual(report['completed_seeds'], [0])
        self.assertEqual(report['missing_seeds'], [2])
        self.assertFalse(report['complete'])


    def test_full_ade_validation_mask_and_pixel_accuracy_are_verified(self):
        def lost_pixel(result, _):
            result['metrics']['n_pixels'] -= 1

        def lost_class(result, _):
            result['metrics']['n_classes_present'] -= 1

        def wrong_accuracy(_, published):
            published['ade20k_pixel_accuracy'] = 0.785

        records = [self.record('ade20k', 'lora', i, alter=alter)
                   for i, alter in enumerate((lost_pixel, lost_class, wrong_accuracy))]
        report = self.report(records)
        self.assertEqual(report['completed_seeds'], [])
        self.assertEqual([r['seed'] for r in report['invalid_runs']], [0, 1, 2])

    def test_consistent_but_impossible_ade_pixel_accuracy_is_rejected(self):
        def impossible(result, published):
            result['metrics']['pixel_acc'] = published['ade20k_pixel_accuracy'] = 101
        report = self.report([self.record('ade20k', 'dpt', 0, alter=impossible)])
        self.assertEqual(report['completed_seeds'], [])

    def test_different_execution_topologies_do_not_complete_one_cell(self):
        report = self.report([self.record('imagenet', 'lora', seed, alter=
                              (lambda r, _: r.update(world_size=2)) if seed == 2 else None)
                              for seed in range(3)])
        self.assertFalse(report['complete'])
        self.assertEqual(report['completed_seeds'], [0, 1])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'provider discovery requires torch')
    def test_named_short_schedules_are_selected_from_configuration(self):
        for task, name in [('imagenet', 'B5X_LORA_LP_50EP_v2'), ('ssv2', 'B5X_LORA_LP_24EP_v2')]:
            report = self.report([self.record(task, 'lora', seed, schedule=name) for seed in range(3)])
            self.assertTrue(report['complete'])
            self.assertEqual(report['protocol_id'], name)


    def test_claimed_completion_requires_all_updates_and_trainable_artifact(self):
        def wrong_total(r, _):
            r['updates'] -= 1
        def wrong_epoch(r, _):
            r['loss_log'][4]['updates'] = 0
        records = [self.record('coco', 'lora', i, alter=fn)
                   for i, fn in enumerate((wrong_total, wrong_epoch, None))]
        run = records[-1]
        out = Path(run['out'])
        (out/'trainable.pt').unlink()
        cfg = Path(run['config'])
        contract.write_manifest(out, task='coco', method_ref='revised-component', status='ok',
            config_sha256=contract.sha256_bytes(cfg.read_bytes()), started_at='start', finished_at='end',
            seed=2, backbone={'kind': 'fixture'})
        report = self.report(records)
        self.assertEqual([r['seed'] for r in report['invalid_runs']], [0, 1, 2])
        self.assertFalse(report['complete'])
