"""Behavioral specification for an anonymous, committed-source ZIP export."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tests._checkout import needs_git

TOOL = Path(__file__).resolve().parents[1] / 'bin' / 'submission-archive.py'


@needs_git
class TestSubmissionArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Fixture Author')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.write('LICENSE', 'Copyright Example Upstream\nPermission notice must be retained.\n')
        self.write('main.py', 'print("ready")\n')
        self.commit()
        self.policy = {'version': 1, 'include': ['main.py'], 'exclude': [],
                       'forbidden': ['Sensitive Person', 'private-owner'],
                       'replacements': {}, 'approvals': {}, 'max_bytes': 1000000}
        self.report = self.base / 'report.json'
        self.out = self.base / 'submission.zip'

    def git(self, *args, cwd=None):
        return subprocess.check_output(['git', '-C', str(cwd or self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def write(self, name, text):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-qm', 'fixture')

    def module(self):
        self.assertTrue(TOOL.is_file(), 'submission archive implementation is missing')
        spec = importlib.util.spec_from_file_location('submission_archive', TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def run_build(self, out=True):
        return self.module().build(self.repo, self.policy, self.report,
                                   self.out if out else None)

    def findings(self):
        return {f['rule'] for f in json.loads(self.report.read_text())['findings']}

    def assert_blocked(self, rule):
        self.assertFalse(self.run_build())
        self.assertFalse(self.out.exists())
        self.assertIn(rule, self.findings())

    def test_committed_export_is_deterministic_and_runs_without_checkout(self):
        self.write('main.py', 'raise RuntimeError("uncommitted must not ship")\n')
        self.write('secret.txt', 'Sensitive Person')
        self.assertTrue(self.run_build())
        a = self.out.read_bytes()
        with zipfile.ZipFile(self.out) as z:
            self.assertEqual(set(z.namelist()), {'code/LICENSE', 'code/main.py'})
            self.assertEqual(z.read('code/LICENSE'), (self.repo/'LICENSE').read_bytes())
            z.extractall(self.base/'unpacked')
        result = subprocess.check_output([sys.executable, str(self.base/'unpacked/code/main.py')])
        self.assertEqual(result, b'ready\n')
        self.out.unlink()
        self.assertTrue(self.run_build())
        self.assertEqual(a, self.out.read_bytes())
        self.assertTrue(json.loads(self.report.read_text())['working_tree_ignored'])

    def test_identifiers_in_content_and_paths_are_blocked(self):
        for name, body in [('main.py', '# SENSITIVE PERSON'), ('private-owner.py', 'pass')]:
            with self.subTest(name=name):
                self.write(name, body); self.commit(); self.policy['include'] = [name]
                self.assert_blocked('identifier')

    def test_encoded_and_unicode_identifiers_are_blocked(self):
        for body in ('private%2Downer', 'private&#45;owner', '\uff50\uff52\uff49\uff56\uff41\uff54\uff45-owner', 'private\u200b-owner'):
            with self.subTest(body=body):
                self.write('main.py', '# '+body);self.commit();self.assert_blocked('identifier')

    def test_builtin_sensitive_content_and_external_links(self):
        cases = [('email', 'contact person@example.invalid'), ('local-path', '/Users/anyone/project'),
                 ('local-path', '/groups/team/project'), ('secret', '-----BEGIN PRIVATE KEY-----'),
                 ('external-link', 'https://example.invalid/project'), ('external-link', 'git@example.invalid:project/repo')]
        for rule, body in cases:
            with self.subTest(rule=rule):
                self.write('main.py', '# '+body);self.commit();self.assert_blocked(rule)

    def test_replacement_is_hash_bound_and_original_unchanged(self):
        self.write('main.py', 'print("Sensitive Person")\n');self.commit()
        original=(self.repo/'main.py').read_bytes()
        self.policy['replacements']['main.py']={'sha256':hashlib.sha256(original).hexdigest(), 'text':'print("anonymous")\n', 'reason':'Remove local attribution'}
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:self.assertEqual(z.read('code/main.py'),b'print("anonymous")\n')
        self.assertEqual((self.repo/'main.py').read_bytes(),original)
        self.out.unlink();self.write('main.py','print("changed")\n');self.commit()
        self.assert_blocked('stale-replacement')

    def test_replacement_output_is_scanned(self):
        self.policy['replacements']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'text':'Sensitive Person','reason':'fixture'}
        self.assert_blocked('identifier')

    def test_license_cannot_be_silently_excluded_or_replaced(self):
        self.policy['exclude']=['LICENSE'];self.assert_blocked('license-exclusion')
        self.policy['exclude']=[]
        self.policy['replacements']['LICENSE']={'sha256':hashlib.sha256((self.repo/'LICENSE').read_bytes()).hexdigest(),'text':'Anonymous','reason':'fixture'}
        self.assert_blocked('license-replacement')

    def test_approved_link_does_not_approve_identifiers_or_changed_files(self):
        self.write('main.py','# https://example.invalid/upstream\n');self.commit()
        self.policy['approvals']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'rules':['external-link'],'reason':'Reviewed upstream URL'}
        self.assertTrue(self.run_build());self.out.unlink()
        self.write('main.py','# https://example.invalid/private-owner\n');self.commit()
        self.assert_blocked('identifier');self.assertIn('stale-approval',self.findings())

    def test_binary_requires_hash_bound_manual_review(self):
        (self.repo/'main.py').write_bytes(b'\xff\x00binary');self.commit();self.assert_blocked('binary')
        self.policy['approvals']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'rules':['binary'],'reason':'Inspected metadata and visible content'}
        self.assertTrue(self.run_build())

    def test_binary_visible_identifier_cannot_be_approved_away(self):
        raw=b'\xff\x00private-owner';(self.repo/'main.py').write_bytes(raw);self.commit()
        self.policy['approvals']['main.py']={'sha256':hashlib.sha256(raw).hexdigest(),'rules':['binary'],'reason':'fixture'}
        self.assert_blocked('identifier')

    def test_symlink_lfs_and_git_metadata_are_not_shipped(self):
        (self.repo/'link').symlink_to('/etc/passwd');self.commit();self.policy['include']=['link'];self.assert_blocked('symlink')
        self.write('pointer','version https://git-lfs.github.com/spec/v1\noid sha256:abc\n');self.commit();self.policy['include']=['pointer'];self.assert_blocked('lfs-pointer')
        self.write('.gitmodules','private-owner');self.commit();self.policy['include']=['main.py','.gitmodules'];self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:self.assertNotIn('code/.gitmodules',z.namelist())

    def test_missing_selection_and_unused_rules_fail_closed(self):
        self.policy['include']=['missing'];self.assert_blocked('missing-selection')
        self.policy['include']=['main.py'];self.policy['approvals']['missing']={'sha256':'0'*64,'rules':['binary'],'reason':'fixture'};self.assert_blocked('unused-policy')

    def test_case_colliding_archive_paths_are_rejected(self):
        # Construct tree objects directly: macOS checkout cannot hold both names.
        oid=self.git('rev-parse','HEAD:main.py')
        tree=subprocess.check_output(['git','-C',str(self.repo),'mktree'],input=f'100644 blob {oid}\tFoo.py\n100644 blob {oid}\tfoo.py\n'.encode()).decode().strip()
        commit=subprocess.check_output(['git','-C',str(self.repo),'commit-tree',tree,'-p','HEAD'],input=b'collision').decode().strip()
        self.git('update-ref','HEAD',commit);self.policy['include']=['Foo.py','foo.py'];self.assert_blocked('path-collision')

    def test_size_limit_empty_policy_and_path_traversal(self):
        self.policy['max_bytes']=1;self.assert_blocked('size-limit')
        self.policy['max_bytes']=1000000;self.policy['forbidden']=[]
        with self.assertRaises(ValueError):self.run_build()
        self.policy['forbidden']=['private-owner'];self.policy['include']=['../outside']
        with self.assertRaises(ValueError):self.run_build()

    def test_existing_archive_and_report_cannot_overwrite_input(self):
        self.out.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):self.run_build()
        self.assertEqual(self.out.read_bytes(),b'keep')
        self.out.unlink();self.report=self.repo/'main.py'
        with self.assertRaises(ValueError):self.run_build()
        self.assertEqual((self.repo/'main.py').read_text(),'print("ready")\n')

    def test_submodule_materializes_pin_not_dirty_checkout(self):
        upstream=self.base/'upstream';upstream.mkdir()
        self.git('init','-q',cwd=upstream);self.git('config','user.name','Upstream',cwd=upstream);self.git('config','user.email','upstream@example.invalid',cwd=upstream)
        (upstream/'dep.py').write_text('VALUE = 7\n');(upstream/'LICENSE').write_text('Upstream license\n')
        self.git('add','.',cwd=upstream);self.git('commit','-qm','first',cwd=upstream)
        self.git('-c','protocol.file.allow=always','submodule','add',str(upstream),'vendor/dep');self.commit()
        (self.repo/'vendor/dep/dep.py').write_text('private-owner')
        self.policy['include'].append('vendor')
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:
            self.assertEqual(z.read('code/vendor/dep/dep.py'),b'VALUE = 7\n')
            self.assertEqual(z.read('code/vendor/dep/LICENSE'),b'Upstream license\n')
            self.assertFalse(any('.git' in Path(n).parts for n in z.namelist()))
        self.out.unlink();shutil.rmtree(self.repo/'vendor/dep')
        self.assert_blocked('missing-submodule')

    def test_cli_reports_failure_without_printing_matched_content(self):
        self.assertTrue(TOOL.is_file())
        self.write('main.py','# private-owner');self.commit()
        policy=self.base/'policy.json';policy.write_text(json.dumps(self.policy))
        r=subprocess.run([sys.executable,str(TOOL),'--repo',str(self.repo),'--policy',str(policy),'--report',str(self.report),'--out',str(self.out)],capture_output=True,text=True)
        self.assertEqual(r.returncode,1);self.assertNotIn('private-owner',r.stdout+r.stderr);self.assertFalse(self.out.exists())

    def test_archive_filename_cannot_reveal_identity(self):
        self.out=self.base/'private-owner-submission.zip'
        self.assert_blocked('identifier')

    def test_copyright_headers_cannot_be_removed_by_replacement(self):
        self.write('main.py','# Copyright Example Upstream\nprint("ready")\n');self.commit()
        self.policy['replacements']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'text':'print("ready")\n','reason':'fixture'}
        self.assert_blocked('license-replacement')

    def test_windows_reserved_paths_fail_closed(self):
        self.write('CON.txt','data');self.commit();self.policy['include']=['CON.txt']
        self.assert_blocked('unsafe-archive-path')

    def test_notebook_and_embedded_image_require_manual_review(self):
        self.write('plot.ipynb','{"cells": []}');self.commit();self.policy['include']=['plot.ipynb'];self.assert_blocked('opaque')
        self.write('image.svg','<svg href="data:image/png;base64,AA=="/>');self.commit();self.policy['include']=['image.svg'];self.assert_blocked('opaque')

    def test_replacement_size_and_report_privacy(self):
        self.policy['replacements']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'text':'x'*1000001,'reason':'fixture'}
        self.assert_blocked('size-limit')
        self.assertEqual(self.report.stat().st_mode & 0o777,0o600)

    def test_original_size_limit_applies_before_small_replacement(self):
        self.write('main.py','# '+'x'*2000);self.commit()
        self.policy['max_bytes']=1000
        self.policy['replacements']['main.py']={'sha256':hashlib.sha256((self.repo/'main.py').read_bytes()).hexdigest(),'text':'pass\n','reason':'fixture'}
        self.assert_blocked('size-limit')

    def test_long_text_scan_has_bounded_runtime(self):
        self.write('main.py','# '+'x'*200000);self.commit()
        policy=self.base/'policy.json';policy.write_text(json.dumps(self.policy))
        try:
            r=subprocess.run([sys.executable,str(TOOL),'--repo',str(self.repo),'--policy',str(policy),'--report',str(self.report)],capture_output=True,timeout=5)
        except subprocess.TimeoutExpired:
            self.fail('plain text inspection exceeded five seconds')
        self.assertEqual(r.returncode,0,r.stderr)

    def test_documented_policy_and_cli_example_produce_working_zip(self):
        root=TOOL.parents[1]
        example=root/'configs/submission-policy.example.json'
        self.assertTrue(example.is_file(),'documented policy example is missing')
        policy=json.loads(example.read_text())
        self.module().policy_check(policy)
        doc=(root/'docs/SUBMISSION_ARCHIVE.md')
        self.assertTrue(doc.is_file(),'submission documentation is missing')
        self.assertIn('python3 bin/submission-archive.py',doc.read_text())
        self.assertIn('configs/submission-policy.example.json',doc.read_text())
        self.write('README.md','Run main.py for the fixture.\n');self.commit()
        # Use the example without private identifiers or edits.
        self.policy=policy
        private_policy=self.base/'policy.json';private_policy.write_text(json.dumps(policy))
        result=subprocess.run([sys.executable,str(TOOL),'--repo',str(self.repo),'--ref','HEAD','--policy',str(private_policy),'--report',str(self.report),'--out',str(self.out)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(self.out.is_file(),'documented CLI did not create the archive')
        with zipfile.ZipFile(self.out) as z:
            self.assertIn('code/README.md',z.namelist())
            self.assertIn('code/LICENSE',z.namelist())
        # Relative documentation links point to actual repository files.
        self.assertIn('SUBMISSION_ARCHIVE.md',(root/'docs/HANDOFF.md').read_text())

    def test_private_policy_inside_checkout_is_rejected(self):
        policy=self.repo/'private-policy.json';policy.write_text(json.dumps(self.policy))
        r=subprocess.run([sys.executable,str(TOOL),'--repo',str(self.repo),'--policy',str(policy),'--report',str(self.report),'--out',str(self.out)],capture_output=True)
        self.assertEqual(r.returncode,2)
        self.assertFalse(self.out.exists())

    def test_boolean_policy_version_is_not_an_integer_version(self):
        self.policy['version']=True
        with self.assertRaises(ValueError):
            self.run_build()

    def test_malformed_approval_is_a_clean_cli_validation_error(self):
        self.policy['approvals']['main.py']={'sha256':'0'*64,'rules':[{}],'reason':'fixture'}
        policy=self.base/'policy.json';policy.write_text(json.dumps(self.policy))
        r=subprocess.run([sys.executable,str(TOOL),'--repo',str(self.repo),'--policy',str(policy),'--report',str(self.report),'--out',str(self.out)],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        self.assertNotIn('Traceback',r.stderr)
        self.assertFalse(self.out.exists())

    def test_subdirectory_cannot_bypass_private_output_boundary(self):
        nested=self.repo/'nested';nested.mkdir()
        private_report=self.repo/'private-report.json'
        with self.assertRaises(ValueError):
            self.module().build(nested,self.policy,private_report,self.out)
        self.assertFalse(private_report.exists())
        self.assertFalse(self.out.exists())
