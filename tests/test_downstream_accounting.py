"""Only technically valid final-epoch runs enter downstream seed summaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from downstream import contract

ROOT = Path(__file__).resolve().parents[1]


class TestAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        spec=importlib.util.find_spec('downstream.accounting')
        self.assertIsNotNone(spec,'downstream final-run accounting is missing')
        from downstream import accounting
        self.accounting=accounting

    def run_record(self,seed,value,**overrides):
        out=self.root/str(seed); out.mkdir()
        cfg=dict(task='imagenet_classification',seed=seed,adaptation='frozen',
                 backbone={'kind':'fixture','encoder':'fixed-checkpoint'},probe={'epochs':3})
        cfg.update(overrides)
        config=self.root/f'config-{seed}.json'; config.write_text(json.dumps(cfg))
        raw={'top1':value,'epochs':3}
        contract.write_metrics(out,raw,{'top1':'imagenet_top1','epochs':'epochs_completed'})
        (out/'results.json').write_text(json.dumps(dict(task=cfg['task'],backbone=cfg['backbone'],
             adaptation=cfg['adaptation'],final=raw,canonical_eligible=False,record_value=False)))
        contract.write_manifest(out,task=cfg['task'],method_ref='fixture',status='ok',
             config_sha256=contract.sha256_bytes(config.read_bytes()),started_at='start',finished_at='end',
             seed=seed,backbone=cfg['backbone'])
        return dict(seed=seed,out=str(out),config=str(config),exit_status=0)

    def test_low_and_zero_scores_partial_seeds_and_sample_std(self):
        records=[self.run_record(0,0),self.run_record(1,4)]
        report=self.accounting.aggregate(records,expected_seeds=[0,1,2])
        self.assertEqual(report['completed_seeds'],[0,1])
        self.assertEqual(report['missing_seeds'],[2])
        self.assertFalse(report['complete'])
        self.assertEqual(report['metrics']['imagenet_top1']['values'],[0,4])
        self.assertEqual(report['metrics']['imagenet_top1']['n'],2)
        self.assertEqual(report['metrics']['imagenet_top1']['mean'],2)
        self.assertAlmostEqual(report['metrics']['imagenet_top1']['std'],2**1.5)
        self.assertFalse(report['canonical_eligible'])

    def test_singleton_std_unknown_and_failed_run_is_not_zero(self):
        good=self.run_record(0,5); bad=self.run_record(1,90); bad['exit_status']=1
        report=self.accounting.aggregate([good,bad],expected_seeds=[0,1])
        self.assertIsNone(report['metrics']['imagenet_top1']['std'])
        self.assertEqual(report['metrics']['imagenet_top1']['mean'],5)
        self.assertEqual(report['invalid_runs'][0]['seed'],1)
        self.assertTrue(report['invalid_runs'][0]['reasons'])

    def test_duplicate_seed_or_changed_recipe_refused(self):
        a=self.run_record(0,2); b=self.run_record(1,3,adaptation='attentive')
        for records in ([a,a],[a,b]):
            with self.assertRaises(ValueError):self.accounting.aggregate(records,expected_seeds=[0,1])

    def test_changed_artifact_wrong_seed_and_partial_epoch_are_not_valid(self):
        a=self.run_record(0,2)
        path=Path(a['out'])/'metrics.json'
        obj=json.loads(path.read_text()); obj['metrics']['imagenet_top1']=99; path.write_text(json.dumps(obj))
        self.assertEqual(self.accounting.aggregate([a],expected_seeds=[0])['completed_seeds'],[])
        for mutate in ('seed','epochs','nonfinite'):
            b=self.run_record(len(list(self.root.glob('config-*.json'))),3)
            out=Path(b['out'])
            if mutate=='seed':
                manifest=out/'run_manifest.json'; data=json.loads(manifest.read_text()); data['seed']=999
                manifest.write_text(json.dumps(data))
            else:
                metrics=out/'metrics.json'; data=json.loads(metrics.read_text())
                data['metrics']['epochs_completed' if mutate=='epochs' else 'imagenet_top1']=2 if mutate=='epochs' else float('nan')
                metrics.write_text(json.dumps(data))
                manifest=out/'run_manifest.json'; data=json.loads(manifest.read_text()); data['artifacts']=contract.collect_artifacts(out)
                manifest.write_text(json.dumps(data))
            report=self.accounting.aggregate([b],expected_seeds=[b['seed']])
            self.assertEqual(report['completed_seeds'],[],mutate)

    def test_cli_delivers_parseable_report_without_overwriting(self):
        record=self.run_record(0,3)
        plan=self.root/'plan.json'; plan.write_text(json.dumps({'expected_seeds':[0,1],'runs':[record]}))
        output=self.root/'report.json'
        cmd=[sys.executable,'-m','downstream.accounting','--config',str(plan),'--out',str(output)]
        first=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertEqual(json.loads(output.read_text())['missing_seeds'],[1])
        second=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(second.returncode,0)





class TestV1Reporting(unittest.TestCase):
    setUp = TestAccounting.setUp
    run_record = TestAccounting.run_record

    @staticmethod
    def finalize(record):
        out = Path(record['out'])
        manifest = json.loads((out / contract.MANIFEST).read_text())
        keys = ('task', 'method_ref', 'status', 'config_sha256', 'started_at', 'finished_at', 'seed', 'backbone')
        contract.write_manifest(out, **{key: manifest[key] for key in keys})

    def make_cell(self, task, fields, epochs, adaptation, cap=0):
        records = []
        for seed in (0, 1, 2):
            settings = dict(epochs=epochs, max_train_samples=cap, max_val_samples=0,
                            max_steps_per_epoch=0)
            key = 'detector' if task == 'coco_detection' else 'probe'
            record = self.run_record(seed, seed, task=task, adaptation=adaptation,
                                     **{key: settings})
            out = Path(record['out'])
            raw = {key: float(seed + index) for index, key in enumerate(fields)}
            raw['epochs'] = epochs
            contract.write_metrics(out, raw, {**fields, 'epochs': 'epochs_completed'})
            path = out / 'results.json'
            result = json.loads(path.read_text())
            result['final'] = raw
            path.write_text(json.dumps(result))
            self.finalize(record)
            records.append(record)
        return records

    def report(self, records, seeds=(0, 1, 2)):
        return self.accounting.aggregate(records, expected_seeds=list(seeds), basic5_v1=True)

    def test_full_three_seed_output_and_no_scientific_promotion(self):
        fields = {'top1': 'imagenet_top1', 'top5': 'imagenet_top5'}
        rows = self.make_cell('imagenet_classification', fields, 100, 'finetune')
        report = self.report(rows)
        self.assertTrue(report['complete'])
        self.assertEqual(report['reporting_protocol'], 'BASIC5_FINETUNE_v1')
        self.assertEqual(report['metrics']['imagenet_top1']['std'], 1)
        for row in rows:
            path = Path(row['out']) / 'results.json'
            result = json.loads(path.read_text())
            result.update(canonical_eligible=True, record_value=True)
            path.write_text(json.dumps(result))
            self.finalize(row)
        report = self.report(rows)
        self.assertFalse(report['record_value'])
        self.assertFalse(report['canonical_eligible'])
        with self.assertRaises(ValueError):
            self.report(rows[:1], [0])

    def test_depth_and_video_cli_report_all_metrics(self):
        cases = [
            ('nyuv2_depth', 30, {'rmse': 'nyuv2_rmse', 'abs_rel': 'nyuv2_absrel',
                                 **{f'delta{i}': f'nyuv2_delta{i}' for i in (1, 2, 3)}}),
            ('ssv2_video', 50, {'top1': 'ssv2_top1', 'top5': 'ssv2_top5'}),
        ]
        for task, epochs, fields in cases:
            with self.subTest(task=task):
                root = Path(self.tmp.name) / task
                root.mkdir()
                self.root = root
                rows = self.make_cell(task, fields, epochs, 'frozen')
                config = root / 'plan.json'
                config.write_text(json.dumps({'expected_seeds': [0, 1, 2], 'runs': rows}))
                output = root / 'summary.json'
                subprocess.run([sys.executable, '-m', 'downstream.accounting', '--basic5-v1',
                                '--config', str(config), '--out', str(output)], check=True)
                report = json.loads(output.read_text())
                self.assertTrue(report['complete'])
                self.assertEqual(report['reporting_protocol'], 'BASIC5_FAIR_v1')
                self.assertEqual(set(report['metrics']), set(fields.values()))

    def test_short_capped_missing_or_remapped_scores_excluded(self):
        fields = {'miou': 'ade20k_miou', 'pacc': 'ade20k_pixel_accuracy'}
        rows = self.make_cell('ade20k_segmentation', fields, 20, 'attentive')
        for row, mode in zip(rows, ('missing', 'altered', 'cross_task')):
            path = Path(row['out']) / contract.METRICS
            data = json.loads(path.read_text())
            if mode == 'missing':
                del data['metrics']['ade20k_miou']
            elif mode == 'altered':
                data['metrics']['ade20k_pixel_accuracy'] += 1
            else:
                data['metrics']['imagenet_top1'] = 1
            path.write_text(json.dumps(data))
            self.finalize(row)
        report = self.report(rows)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['invalid_runs']), 3)
        self.assertEqual(report['metrics'], {})

    def test_coco_requires_twelve_full_epochs_and_uncapped_data(self):
        fields = {f'bbox_mAP{s}': f'coco_map{s}' for s in ('', '_50', '_75', '_small', '_medium', '_large')}
        rows = self.make_cell('coco_detection', fields, 2, 'frozen', cap=1)
        report = self.report(rows)
        self.assertEqual(report['completed_seeds'], [])
        self.assertIn('v1 epoch horizon differs', report['invalid_runs'][0]['reasons'])
        self.assertIn('v1 requires uncapped data and updates', report['invalid_runs'][0]['reasons'])


if __name__ == '__main__':
    unittest.main()
