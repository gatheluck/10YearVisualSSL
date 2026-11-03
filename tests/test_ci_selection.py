"""Execute the workflow's selector against real commits, including deletions."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._checkout import needs_git
from tests.test_ci import discovery_script, needs_yaml, parsed

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / '.github/workflows/tests.yml').is_file(),
                     'workflow is not included in the runtime image')
@needs_git
@needs_yaml
class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.git('config', 'user.name', 'Fixture')
        self.write('.gitignore', '__pycache__/\n')
        for name in ('alpha', 'alphabet', 'beta'):
            self.write(f'methods/{name}/requirements.lock.txt', 'example==1\n')
            self.write(f'methods/{name}/model.py', 'VALUE = 1\n')
            self.write(f'methods/{name}/README.md', 'Example\n')
        self.write('README.md', 'Example\n')
        helper = ROOT / 'ci_selection.py'
        if helper.exists():
            shutil.copy2(helper, self.root / helper.name)
        self.commit()
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, text=True)

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def select(self, base=None, event='pull_request'):
        self.commit()
        # Use exactly the discovery code executed in Actions.
        helper = ROOT / 'ci_selection.py'
        if helper.exists():
            shutil.copy2(helper, self.root / helper.name)
        payload = self.root / '.git/event.json'
        payload.write_text(json.dumps({'pull_request': {'base': {'sha': base or self.base}}}))
        proc = subprocess.run(
            [sys.executable, '-c', discovery_script(parsed()['tests.yml'])],
            cwd=self.root, capture_output=True, text=True, check=False,
            env={**os.environ, 'GITHUB_EVENT_NAME': event,
                 'GITHUB_EVENT_PATH': str(payload)},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return dict(line.split('=', 1) for line in proc.stdout.splitlines())

    def test_document_only_keeps_core_and_selects_no_heavy_jobs(self):
        self.write('README.md', 'Updated prose\n')
        result = self.select()
        self.assertEqual(json.loads(result['methods']), [])
        self.assertEqual(result['heavy'], 'false')
        self.assertEqual(result['downstream'], 'false')

    def test_local_workflow_uses_same_commit_as_export(self):
        from tests.test_run_ci_locally import runner
        self.write('.github/workflows/tests.yml', 'jobs: {committed: {}}\n')
        self.commit()
        workflow = self.root / '.github/workflows/tests.yml'
        workflow.write_text('jobs: {uncommitted: {}}\n')
        self.assertEqual(runner.load_workflow(workflow, root=self.root),
                         {'jobs': {'committed': {}}})
        self.assertEqual(runner.load_workflow(workflow),
                         {'jobs': {'uncommitted': {}}})
        with self.assertRaises(runner.CannotReproduce):
            runner.load_workflow(self.root / 'absent.yml', root=self.root)

    def test_docs_and_method_readme_are_light_but_config_is_not(self):
        self.write('docs/guide.md', 'Prose\n')
        self.write('methods/beta/README.md', 'Updated\n')
        self.assertEqual(self.select()['heavy'], 'false')
        self.base = self.git('rev-parse', 'HEAD').strip()
        self.write('methods/beta/configs/train.yaml', 'epochs: 2\n')
        self.assertEqual(json.loads(self.select()['methods']), ['beta'])

    def test_deleted_readme_is_still_checked_by_core(self):
        self.git('rm', 'README.md')
        self.assertEqual(self.select()['heavy'], 'false')

    def test_new_method_is_discovered_and_selected(self):
        self.write('methods/gamma/requirements.lock.txt', 'example==1\n')
        self.assertEqual(json.loads(self.select()['methods']), ['gamma'])

    def test_invalid_base_cannot_be_interpreted_as_git_option(self):
        self.write('README.md', 'Updated\n')
        self.assertEqual(self.select(base='--help')['heavy'], 'true')

    def test_workflow_gates_empty_matrix_and_keeps_base_checks(self):
        from tests.test_run_ci_locally import runner
        jobs = parsed()['tests.yml']['jobs']
        self.assertNotIn('if', jobs['core'])
        self.assertIn('tests/run-tests.sh', str(jobs['core']['steps']))
        self.assertEqual(jobs['discover']['steps'][0]['with']['fetch-depth'], 0)
        for doc_only in (True, False):
            outputs = {'discover': {'heavy': str(not doc_only).lower(),
                                    'downstream': str(not doc_only).lower()}}
            for name in ('locked', 'container', 'downstream'):
                self.assertEqual(runner.applicable(jobs[name], 'pull_request', outputs)[0],
                                 not doc_only, name)
        self.assertFalse(runner.applicable(jobs['locked'], 'push', {})[0])
        self.assertTrue(runner.applicable(jobs['downstream'], 'push',
                        {'discover': {'downstream': 'true'}})[0])

    def test_method_changes_match_complete_directory_not_prefix(self):
        self.write('methods/alpha/model.py', 'VALUE = 2\n')
        result = self.select()
        self.assertEqual(json.loads(result['methods']), ['alpha'])
        self.assertEqual(result['downstream'], 'true')

    def test_rename_checks_old_and_new_method(self):
        self.git('mv', 'methods/alpha/model.py', 'methods/beta/moved.py')
        self.assertEqual(json.loads(self.select()['methods']), ['alpha', 'beta'])

    def test_shared_and_unknown_paths_run_every_environment(self):
        for path in ('downstream/shared.py', 'tests/test_method_alpha.py',
                     '.github/workflows/new.yml', 'docs/example.json',
                     'unknown.md', 'methods/new/model.py', 'third_party/new'):
            with self.subTest(path=path):
                self.write(path, 'changed\n')
                result = self.select()
                self.assertEqual(json.loads(result['methods']), ['alpha', 'alphabet', 'beta'])
                self.assertEqual(result['downstream'], 'true')
                self.base = self.git('rev-parse', 'HEAD').strip()

    def test_deleted_method_lock_falls_back_to_all_remaining_methods(self):
        self.git('rm', 'methods/alpha/requirements.lock.txt')
        self.assertEqual(json.loads(self.select()['methods']), ['alphabet', 'beta'])

    def test_unavailable_base_and_push_use_full_selection(self):
        self.write('README.md', 'New text\n')
        result = self.select(base='a' * 40)
        self.assertEqual(json.loads(result['methods']), ['alpha', 'alphabet', 'beta'])
        self.write('README.md', 'Another text\n')
        self.assertEqual(self.select(event='push')['downstream'], 'true')

    def test_document_named_source_and_symlink_are_not_prose(self):
        self.write('docs/README.md.py', 'print(1)\n')
        self.assertEqual(self.select()['heavy'], 'true')
        self.base = self.git('rev-parse', 'HEAD').strip()
        (self.root / 'README.md').unlink()
        (self.root / 'README.md').symlink_to('methods/alpha/model.py')
        self.assertEqual(self.select()['heavy'], 'true')


class LocalConditions(unittest.TestCase):
    def test_output_gate_evaluates_both_controls_and_rejects_missing_output(self):
        from tests.test_run_ci_locally import runner
        spec = {'if': "github.event_name == 'pull_request' && needs.discover.outputs.heavy == 'true'"}
        for value, expected in (('true', True), ('false', False)):
            self.assertEqual(runner.applicable(spec, 'pull_request',
                             {'discover': {'heavy': value}})[0], expected)
        with self.assertRaises(runner.CannotReproduce):
            runner.applicable(spec, 'pull_request', {})
        with self.assertRaises(runner.CannotReproduce):
            runner.applicable(spec, 'pull_request', {'discover': {'heavy': 'TRUE'}})

    def test_execution_waits_for_discovery_before_gating(self):
        import contextlib
        import io
        from unittest.mock import patch

        from tests.test_run_ci_locally import runner
        for value in ('true', 'false'):
            doc = {'jobs': {
                'discover': {'outputs': {'heavy': 'unused'}, 'steps': []},
                'check': {'needs': 'discover',
                          'if': "needs.discover.outputs.heavy == 'true'",
                          'steps': [{'run': 'echo selected'}]}}}
            with patch.object(runner, 'dirty_files', return_value=[]), \
                 patch.object(runner, 'export_head'), \
                 patch.object(runner, '_collect_outputs', return_value=({'heavy': value}, [])), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                runner.execute(doc, 'pull_request', None, 'linux/amd64', True, ROOT)
            self.assertEqual('echo selected' in log.getvalue(), value == 'true')
