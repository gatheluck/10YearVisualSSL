"""Equivalent lock inputs share a full suite, never a method's image smoke."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import ci_selection
from tests.test_ci import discovery_script, needs_yaml, parsed

ROOT = Path(__file__).resolve().parents[1]


def lock(version='1.0', digest='a'):
    return f'example=={version} \\\n    --hash=sha256:{digest * 64}\n'


class PipReference(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pip'), 'pip parser unavailable')
    def test_real_groups_preserve_pip_requirements_and_hash_options(self):
        from unittest.mock import patch

        from pip._internal.network.session import PipSession
        from pip._internal.req.req_file import parse_requirements

        methods = sorted(p.parent.name for p in
                         (ROOT / 'methods').glob('*/requirements.lock.txt'))
        self.assertTrue(methods, 'no lock inputs to check')
        groups = ci_selection.environment_groups(methods, root=ROOT)
        self.assertEqual(sorted(name for group in groups for name in group), methods)
        with PipSession() as session, patch.object(
                session, 'get', side_effect=AssertionError('network forbidden in lock audit')):
            def parsed_input(name):
                path = ROOT / 'methods' / name / 'requirements.lock.txt'
                return [(r.requirement, r.options, r.is_editable, r.constraint)
                        for r in parse_requirements(str(path), session=session)]

            for group in groups:
                if len(group) < 2:
                    continue
                expected = parsed_input(group[0])
                for method in group[1:]:
                    with self.subTest(method=method):
                        self.assertEqual(parsed_input(method), expected)


class EnvironmentGroups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'requirements-tools.lock.txt').write_text(lock())

    def put(self, name, text):
        path = self.root / 'methods' / name / 'requirements.lock.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def groups(self, names):
        return ci_selection.environment_groups(names, root=self.root)

    def test_comments_share_one_environment_but_all_members_remain_visible(self):
        self.put('alpha', '# First method\n' + lock())
        self.put('alphabet', '# Different explanation \\\n# Continued comment\n\n' + lock().replace('\n', '\r\n'))
        self.put('beta', lock(version='2.0'))
        self.assertEqual(self.groups(['beta', 'alphabet', 'alpha']),
                         [['alpha', 'alphabet'], ['beta']])
        self.assertEqual(self.groups(['alphabet']), [['alphabet']])
        self.assertEqual(self.groups([]), [])

    def test_version_hash_and_requirement_order_are_not_equated(self):
        first = lock()
        other = lock().replace('example', 'another')
        variants = [first, lock(version='2.0'), lock(digest='b'),
                    first + other, other + first]
        names = [f'method_{i}' for i in range(len(variants))]
        for name, value in zip(names, variants):
            self.put(name, value)
        self.assertEqual(self.groups(names), [[name] for name in names])

    def test_unsupported_or_incomplete_inputs_keep_separate_jobs(self):
        variants = ['-r other.txt\n', '--extra-index-url https://example.invalid\n',
                    'example==1.0\n', 'example @ ./local.whl\n',
                    lock().replace('==1.0', '==${VERSION}'),
                    lock().replace('==1.0', '==1.0; python_version > "3"'),
                    lock().replace('example==', 'example[extra]=='),
                    lock().replace('    --hash', '\n    --hash'),
                    lock().replace('    --hash', '# interrupted\n    --hash'),
                    lock().rstrip() + ' \\\n', '', '# coding: latin-1\n' + lock(),
                    '# non-ASCII: \u00e9\n' + lock()]
        for value in variants:
            with self.subTest(value=value):
                self.put('alpha', value)
                self.put('beta', value)
                self.assertEqual(self.groups(['alpha', 'beta']), [['alpha'], ['beta']])

    def test_missing_and_symlinked_locks_are_not_deduplicated(self):
        target = self.put('alpha', lock())
        link = self.put('beta', lock())
        link.unlink()
        link.symlink_to(target)
        self.assertEqual(self.groups(['alpha', 'beta', 'missing']),
                         [['alpha'], ['beta'], ['missing']])

    def test_unknown_tools_input_disables_grouping(self):
        self.put('alpha', lock())
        self.put('beta', lock())
        tools = self.root / 'requirements-tools.lock.txt'
        tools.write_text('-r extra.txt\n')
        self.assertEqual(self.groups(['alpha', 'beta']), [['alpha'], ['beta']])
        tools.unlink()
        self.assertEqual(self.groups(['alpha', 'beta']), [['alpha'], ['beta']])

    @needs_yaml
    @unittest.skipUnless((ROOT / '.github/workflows/tests.yml').is_file(), 'no workflow')
    def test_actual_discovery_keeps_every_container_and_groups_only_locked_jobs(self):
        self.put('alpha', lock())
        self.put('beta', '# Different method\n' + lock())
        self.put('gamma', lock(version='2.0'))
        shutil.copy2(ROOT / 'ci_selection.py', self.root / 'ci_selection.py')
        result = subprocess.run(
            [sys.executable, '-c', discovery_script(parsed()['tests.yml'])],
            cwd=self.root, env={**os.environ, 'GITHUB_EVENT_NAME': 'push'},
            text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertEqual(json.loads(outputs['methods']), ['alpha', 'beta', 'gamma'])
        self.assertEqual(json.loads(outputs['locked_methods']), ['alpha', 'gamma'])
        self.assertIn('beta', result.stderr, 'group membership must be visible in the log')


@needs_yaml
@unittest.skipUnless((ROOT / '.github/workflows/tests.yml').is_file(), 'no workflow')
class WorkflowContract(unittest.TestCase):
    def test_only_locked_matrix_is_grouped_and_test_commands_are_method_independent(self):
        jobs = parsed()['tests.yml']['jobs']
        locked = jobs['locked']
        self.assertEqual(locked['strategy']['matrix'],
                         {'method': '${{ fromJSON(needs.discover.outputs.locked_methods) }}'})
        self.assertEqual(jobs['container']['strategy']['matrix'],
                         {'method': '${{ fromJSON(needs.discover.outputs.methods) }}'})
        commands = [step['run'] for step in locked['steps'] if 'run' in step]
        self.assertIn('.venv/bin/python -m unittest discover -s tests -v', commands)
        text = json.dumps(locked).replace('methods/${{ matrix.method }}/requirements.lock.txt', '')
        self.assertNotIn('matrix.', text, 'grouped jobs may differ only by lock path')
        self.assertIn('tests.test_method_${{ matrix.method }}',
                      json.dumps(jobs['container']['steps']))
