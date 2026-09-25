"""Dry plans expand every row without exporting an unused checkout per row."""
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
import test_run_ci_locally as fixture


class DryRunExports(unittest.TestCase):
    def exercise(self, dry):
        runner=fixture.runner
        doc={'jobs':{
            'discover':{'outputs':{'rows':'unused'},'steps':[{'run':
                'printf \'rows=["alpha","beta"]\\n\' >> "$GITHUB_OUTPUT"'}]},
            'probe':{'needs':'discover','strategy':{'matrix':{'name':'${{ fromJson(needs.discover.outputs.rows) }}'}},
                     'steps':[{'name':'isolated work','run':'test ! -e row-marker && touch row-marker'}]}}}
        exports=[]; ran=[]; log=io.StringIO()
        def export(root,dest):
            exports.append(dest)
            (dest/'tracked-file').write_text('committed data')
        def run(command,tree,image,platform,place):
            ran.append(tree)
            self.assertTrue((tree/'tracked-file').is_file())
            return subprocess.run(['bash','-c',command],cwd=tree).returncode
        with mock.patch.object(runner,'dirty_files',return_value=[]), mock.patch.object(runner,'export_head',side_effect=export), mock.patch.object(runner,'run_step',side_effect=run), mock.patch.object(runner,'image_for',return_value='test-image'), redirect_stdout(log):
            report=runner.execute(doc,'pull_request',None,'linux/amd64',dry,fixture.ROOT)
        self.assertIn('name=alpha',log.getvalue()); self.assertIn('name=beta',log.getvalue())
        self.assertTrue(all(not p.exists() for p in exports))
        return exports,ran,report

    def test_dry_run_exports_once_but_runs_real_discovery_and_expands_all_rows(self):
        exports,ran,_=self.exercise(True)
        self.assertEqual(len(exports),1)
        self.assertEqual(ran,[])

    def test_real_rows_still_have_independent_trees_and_execute(self):
        exports,ran,_=self.exercise(False)
        self.assertEqual(len(exports),3)
        self.assertEqual(len(set(ran)),2)
