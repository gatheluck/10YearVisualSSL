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


if __name__=='__main__':unittest.main()
