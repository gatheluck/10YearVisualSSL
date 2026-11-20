"""Historical Figure 3 reporting through the same CLI as portable reports."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NativeReporting(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runs = []
        self.records = []
        for seed in range(3):
            record = dict(dataset='nyuv2', method_id='example', track='DPT',
                          protocol_id='B5X_DPT_LP_v2', seed=seed, status='COMPLETED_VALID',
                          code_sha256='a' * 64, checkpoint_identity={'file': '/private/weight'},
                          recipe={'epochs': 30}, loss_log=[{'epoch': 30}],
                          metrics=dict(n_images=654, scale_alignment='per-image median',
                                       primary=0.3, rmse=0.3, rmse_unaligned=0.4 + seed / 100))
            self.records.append(record)
            self.runs.append(dict(seed=seed, result=self.pin(seed, 'result', record),
                                  completion=self.pin(seed, 'completion', record)))
        self.cell = dict(dataset='nyuv2', method_id='example', track='DPT',
                         protocol_id='B5X_DPT_LP_v2', runs=self.runs,
                         paper_mean='0.410', paper_sd='0.010')

    def pin(self, seed, name, record):
        path = self.root / f'{seed}-{name}.json'
        raw = json.dumps(record).encode()
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def report(self):
        config = self.root / 'plan.json'
        config.write_text(json.dumps(dict(schema='native_figure3_v1', cells=[self.cell])))
        out = self.root / 'report.json'
        proc = subprocess.run([sys.executable, '-m', 'downstream.revised_reporting',
                               '--config', str(config), '--out', str(out)], capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        raw = out.read_text()
        self.assertNotIn('/private/weight', raw)
        self.assertNotIn(str(self.root), raw)
        return json.loads(raw)['cells'][0]

    def test_unaligned_mean_and_sample_sd_match_paper(self):
        result = self.report()
        self.assertEqual(result['status'], 'match')
        self.assertEqual(result['observed'], {'mean': '0.410', 'sd': '0.010'})

    def test_wrong_sd_alone_is_a_mismatch(self):
        self.cell['paper_sd'] = '0.000'
        self.assertEqual(self.report()['status'], 'mismatch')

    def test_partial_seeds_never_report_a_match(self):
        self.runs.pop()
        self.assertEqual(self.report()['status'], 'incomplete')

    def test_different_completion_artifact_is_invalid(self):
        self.runs[0]['completion'] = self.pin(0, 'completion', self.records[2])
        self.assertEqual(self.report()['status'], 'invalid')

    def test_wrong_final_epoch_is_invalid(self):
        self.records[0]['loss_log'][-1]['epoch'] = 29
        for key in ('result', 'completion'):
            self.runs[0][key] = self.pin(0, key, self.records[0])
        self.assertEqual(self.report()['status'], 'invalid')

    def test_different_code_across_seeds_is_diagnosed(self):
        self.records[1]['code_sha256'] = 'b' * 64
        for key in ('result', 'completion'):
            self.runs[1][key] = self.pin(1, key, self.records[1])
        row = self.report()
        self.assertEqual(row['status'], 'invalid')
        self.assertEqual(row['reason'], 'code differs across seeds')
        self.assertNotIn('observed', row)
        self.assertEqual(row['arithmetic_only'], {'mean': '0.410', 'sd': '0.010'})

    def test_identical_code_with_different_recipe_is_invalid(self):
        self.records[2]['recipe']['lr'] = 0.1
        for key in ('result', 'completion'):
            self.runs[2][key] = self.pin(2, key, self.records[2])
        self.assertEqual(self.report()['status'], 'invalid')
