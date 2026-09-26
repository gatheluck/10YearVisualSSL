"""Behavioral specification for an anonymous, committed-source ZIP export."""
import hashlib
import importlib.util
import json
import posixpath
import re
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

    def test_supplied_protocol_links_survive_committed_export(self):
        source = TOOL.parent.parent / 'docs' / 'submission_protocols'
        expected = {f'{family}_{track}.md' for family in ('BASIC5', 'EXTEND')
                    for track in ('LINEAR', 'ATTENTIVE', 'FINETUNE')}
        expected.add('BASIC5_FRONTIER_SYSTEM_PROMPTS.md')
        expected.add('00_scope_and_replication.md')
        expected.update(f'EXTEND_{track}_v1.json'
                        for track in ('LINEAR', 'ATTENTIVE', 'FINETUNE'))
        self.assertTrue(source.is_dir(), 'supplied protocol directory missing')
        for path in source.iterdir():
            if path.is_file():
                self.write('protocols/' + path.name, path.read_text())
        self.write('SUBMISSION_SCOPE.md', (source.parent / 'SUBMISSION_SCOPE.md').read_text())
        self.commit()
        self.policy['include'] = ['protocols', 'SUBMISSION_SCOPE.md']
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as archive:
            index = archive.read('code/protocols/README.md').decode()
            links = re.findall(r'\]\(([^)]+)\)', index)
            self.assertTrue(expected.issubset(set(links)))
            for target in links:
                self.assertEqual(archive.read(posixpath.normpath('code/protocols/' + target)),
                                 (source / target).read_bytes())

    def test_initial_protocols_export_with_separate_evidence_and_proposals(self):
        source = TOOL.parent.parent / 'docs' / 'initial_protocols'
        expected = {'INITIAL_STEP1_v1.md', 'INITIAL_STEP2_v1.md',
                    'INITIAL_STEP3_3DFM_v1.md', 'INITIAL_STEP3_4DFM_v1.md',
                    'INITIAL_STEP3_VIDEO_SSL_v1.md',
                    'INITIAL_STEP3_VIDEO_WORLD_MODELS_v1.md',
                    'INITIAL_STEP3_VISION_GENERATIVE_v1.md',
                    'INITIAL_STEP3_VLM_v1.md'}
        self.assertTrue(source.is_dir(), 'initial protocol companions missing')
        for path in source.glob('*.md'):
            self.write('initial/' + path.name, path.read_text())
        self.commit()
        self.policy['include'] = ['initial']
        self.assertTrue(self.run_build(), self.report.read_text())
        with zipfile.ZipFile(self.out) as archive:
            index = archive.read('code/initial/README.md').decode()
            links = re.findall(r'\]\(([^)]+)\)', index)
            self.assertEqual(set(links), expected)
            for name in expected:
                body = archive.read('code/initial/' + name).decode()
                self.assertEqual(body, (source / name).read_text())
                observed, proposal = body.split('## Proposed changes and unresolved choices', 1)
                self.assertIn('## Historical evidence and its limits', observed)
                self.assertIn('not evidence of completed runs', proposal)
                self.assertNotRegex(body, r'(?i)example job id|projdesc/|canonical v1')
                self.assertNotRegex(body, r'/(?:groups|home|Users)/')
            generative = archive.read('code/initial/INITIAL_STEP3_VISION_GENERATIVE_v1.md').decode()
            observed, proposal = generative.split('## Proposed changes and unresolved choices', 1)
            self.assertIn('K=7', observed)
            self.assertIn('final-layer CLS', proposal)
            video = archive.read('code/initial/INITIAL_STEP3_VIDEO_SSL_v1.md').decode()
            observed, proposal = video.split('## Proposed changes and unresolved choices', 1)
            self.assertIn('20 epochs', observed)
            self.assertIn('50 epochs', proposal)
            self.assertIn('seed remains unresolved', proposal)

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

    def authorize_root_notice(self):
        original = 'MIT License\n\nCopyright (c) 2026 private-owner\n\nPermission and warranty terms remain intact.\n'
        self.write('LICENSE', original)
        self.commit()
        self.policy['first_party_license'] = {
            'sha256': hashlib.sha256(original.encode()).hexdigest(),
            'copyright_line': 'Copyright (c) 2026 private-owner',
            'authorized': True,
            'reason': 'Rights holder requests an anonymous review copy.',
        }
        return original

    def test_authorized_root_notice_preserves_terms_and_source(self):
        original = self.authorize_root_notice()
        self.write('vendor/LICENSE', 'Copyright Third Party\nKeep these terms.\n')
        self.commit()
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:
            self.assertEqual(z.read('code/LICENSE').decode(), original.replace(
                'Copyright (c) 2026 private-owner', 'Copyright (c) 2026 Anonymous authors'))
            self.assertEqual(z.read('code/vendor/LICENSE'), b'Copyright Third Party\nKeep these terms.\n')
        self.assertEqual((self.repo/'LICENSE').read_text(), original)
        report = json.loads(self.report.read_text())
        record = next(x for x in report['files'] if x['path'] == 'LICENSE')
        self.assertTrue(record['replaced'])
        self.assertNotEqual(record['source_sha256'], record['archive_sha256'])

    def test_authorized_root_notice_still_scans_remaining_contents(self):
        self.authorize_root_notice()
        original = (self.repo/'LICENSE').read_text() + 'Sensitive Person\n'
        self.write('LICENSE', original); self.commit()
        self.policy['first_party_license']['sha256'] = hashlib.sha256(original.encode()).hexdigest()
        self.assert_blocked('identifier')

    def test_root_notice_authorization_is_hash_and_line_bound(self):
        self.authorize_root_notice()
        self.policy['first_party_license']['sha256'] = '0'*64
        self.assert_blocked('stale-license-authorization')
        self.policy['first_party_license']['sha256'] = hashlib.sha256((self.repo/'LICENSE').read_bytes()).hexdigest()
        self.policy['first_party_license']['copyright_line'] = 'Copyright (c) 2026 wrong-holder'
        self.assert_blocked('license-authorization-line')

    def test_root_notice_authorization_cannot_target_other_files(self):
        self.authorize_root_notice()
        self.policy['first_party_license']['path'] = 'vendor/LICENSE'
        with self.assertRaises(ValueError): self.run_build()
        del self.policy['first_party_license']['path']
        self.write('vendor/LICENSE', 'Copyright (c) 2026 private-owner\n'); self.commit()
        self.assert_blocked('identifier')

    def test_root_notice_requires_explicit_valid_authorization(self):
        self.authorize_root_notice()
        for value in (False, 1, 'yes', None):
            self.policy['first_party_license']['authorized'] = value
            with self.assertRaises(ValueError): self.run_build()
        self.policy['first_party_license']['authorized'] = True
        for line in ('', 'Permission notice', 'Copyright (c) 2026 owner\nextra'):
            self.policy['first_party_license']['copyright_line'] = line
            with self.assertRaises(ValueError): self.run_build()

    def test_root_notice_authorization_is_not_a_general_replacement(self):
        self.authorize_root_notice()
        self.policy['replacements']['LICENSE'] = {
            'sha256': hashlib.sha256((self.repo/'LICENSE').read_bytes()).hexdigest(),
            'text': 'Terms removed', 'reason': 'Invalid blanket change'}
        self.assert_blocked('license-replacement')

    def test_absent_or_repeated_root_notice_is_rejected(self):
        self.authorize_root_notice()
        original = (self.repo/'LICENSE').read_text()
        self.write('LICENSE', original + 'Copyright (c) 2026 private-owner\n'); self.commit()
        self.policy['first_party_license']['sha256'] = hashlib.sha256((self.repo/'LICENSE').read_bytes()).hexdigest()
        self.assert_blocked('license-authorization-line')
        (self.repo/'LICENSE').unlink(); self.commit()
        self.assert_blocked('unused-license-authorization')

    def test_readme_config_examples_create_resolvable_files(self):
        readme = (TOOL.parents[1]/'README.md').read_text()
        commands = [line for line in readme.splitlines()
                    if line.startswith('mkdir -p configs && printf') or line.startswith('printf ')]
        self.assertEqual(len(commands), 2)
        for command in commands:
            result = subprocess.run(command, shell=True, cwd=self.base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.base/'configs/base.json').is_file())
        self.assertTrue((self.base/'configs/example.json').is_file())
        resolved = self.base/'resolved.json'
        result = subprocess.run([sys.executable, str(TOOL.with_name('resolve-config.py')),
                                 '--config', str(self.base/'configs/example.json'),
                                 '--set', 'DATA_ROOT=/data', '--out', str(resolved)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(resolved.read_text())
        self.assertEqual(data['seed'], 0)
        self.assertEqual(data['optimizer']['lr'], 0.03)
        self.assertEqual(data['data_root'], '/data')

    def test_scheduler_job_templates_are_not_account_paths(self):
        for value in ('/scratch/slurm_tmpdir/{job_id}/',
                      "/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"):
            self.write('main.py', '# '+value+'\n'); self.commit()
            self.assertTrue(self.run_build())
            self.out.unlink()
        for value in ('/scratch/slurm_tmpdir/alice/',
                      '/scratch/slurm_tmpdir/{job_id}/alice',
                      '/scratch/slurm_tmpdir/{unknown}/',
                      '/scratch/alice/{job_id}/',
                      '/scratch/slurm_tmpdir/{job_id}/ /home/alice/project',
                      '/scratch/slurm_tmpdir/{job_id}/ private-owner'):
            self.write('main.py', '# '+value+'\n'); self.commit()
            self.assertFalse(self.run_build())
            self.assertFalse(self.out.exists())

    def test_root_readme_can_reference_license_without_duplicating_notice(self):
        source = (TOOL.parents[1]/'README.md').read_text()
        self.write('README.md', source); self.commit()
        self.policy['include'].append('README.md')
        self.policy['replacements']['README.md'] = {
            'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'text': '# Anonymous source\n\nSee [LICENSE](LICENSE).\n',
            'reason': 'Replace first-party introduction while preserving LICENSE.'}
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:
            self.assertEqual(z.read('code/LICENSE'), (self.repo/'LICENSE').read_bytes())
            self.assertIn('code/LICENSE', z.namelist())

    def test_repository_only_workflow_assertions_require_a_checkout(self):
        root = TOOL.parents[1]
        stage = self.base/'export'; (stage/'tests').mkdir(parents=True)
        (stage/'tests/__init__.py').write_text('')
        modules = ('test_basic5_attentive_tasks', 'test_basic5_finetune_tasks', 'test_basic5_optimization')
        for name in (*modules, '_checkout'):
            shutil.copyfile(root/'tests'/f'{name}.py', stage/'tests'/f'{name}.py')
        # Isolate the workflow reader from optional PyYAML: an absent workflow is
        # always an error inside a checkout, never a dependency-based skip.
        (stage/'tests/test_ci.py').write_text('HAVE_YAML = True\ndef parsed(): return {}\n')
        names = ('TestAttentiveCI.test_downstream_lock_runs_attentive_task_contracts',
                 'TestFinetuneCI.test_downstream_job_executes_finetune_tests',
                 'TestOptimizationCI.test_downstream_job_runs_optimization_with_real_dependencies')
        args = [sys.executable, '-m', 'unittest', '-v',
                *(f'tests.{m}.{n}' for m,n in zip(modules,names))]
        result = subprocess.run(args, cwd=stage, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('Ran 3 tests', result.stderr)
        self.assertIn('skipped=3', result.stderr)
        self.git('init', '-q', cwd=stage)
        result = subprocess.run(args, cwd=stage, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr.count("KeyError: 'tests.yml'"), 3, result.stderr)
        self.assertIn('Ran 3 tests', result.stderr)

    def add_bundled_vendor(self):
        upstream = self.base/'upstream'
        upstream.mkdir()
        self.git('init', '-q', cwd=upstream)
        self.git('config', 'user.name', 'Upstream', cwd=upstream)
        self.git('config', 'user.email', 'upstream@example.invalid', cwd=upstream)
        (upstream/'nativepkg').mkdir()
        (upstream/'nativepkg/__init__.py').write_text('VALUE = 7\n')
        self.git('add', '.', cwd=upstream); self.git('commit', '-qm', 'fixture', cwd=upstream)
        self.git('-c', 'protocol.file.allow=always', 'submodule', 'add', str(upstream), 'vendor/dep')
        self.commit(); self.policy['include'].append('vendor')

    def test_bundle_index_keeps_dependencies_local_without_git_metadata(self):
        from tests import _repo_files, test_method_requirements as requirements
        from unittest import mock
        self.add_bundled_vendor()
        self.write('methods/probe/main.py', 'import nativepkg\nimport undeclared_external\n')
        self.commit(); self.policy['include'].append('methods')
        self.assertTrue(self.run_build())
        stage = self.base/'unpacked'
        with zipfile.ZipFile(self.out) as z:
            self.assertIn('code/upstream_sources.json', z.namelist())
            self.assertEqual(json.loads(z.read('code/upstream_sources.json')),
                             {'version': 1, 'paths': ['vendor/dep']})
            self.assertNotIn('code/.gitmodules', z.namelist())
            z.extractall(stage)
        stage = stage/'code'
        self.assertEqual(_repo_files.submodule_paths(stage), {'vendor/dep'})
        with mock.patch.object(requirements, 'ROOT', stage):
            requirements.local_modules.cache_clear()
            try:
                self.assertEqual(requirements.imported_modules(stage/'methods/probe'),
                                 {'undeclared_external'})
            finally:
                requirements.local_modules.cache_clear()
        record = next(x for x in json.loads(self.report.read_text())['files']
                      if x['path'] == 'upstream_sources.json')
        self.assertIsNone(record['source_sha256'])
        self.assertTrue(record['generated'])

    def test_bundle_index_cannot_be_shadowed_or_replaced(self):
        self.add_bundled_vendor()
        self.write('upstream_sources.json', '{}'); self.commit()
        self.policy['include'].append('upstream_sources.json')
        self.assert_blocked('reserved-archive-path')
        self.policy['include'].remove('upstream_sources.json')
        self.assertTrue(self.run_build())
        with zipfile.ZipFile(self.out) as z:
            generated = z.read('code/upstream_sources.json')
        self.out.unlink()
        self.policy['replacements']['upstream_sources.json'] = {
            'sha256': hashlib.sha256(generated).hexdigest(), 'text': '{}',
            'reason': 'Invalid metadata override'}
        self.assert_blocked('generated-policy')

    def test_bundle_index_reader_rejects_invalid_roots_and_schema(self):
        from tests import _repo_files
        root = self.base/'export'; root.mkdir()
        (root/'vendor/dep').mkdir(parents=True)
        index = root/'upstream_sources.json'
        valid = {'version': 1, 'paths': ['vendor/dep']}
        index.write_text(json.dumps(valid))
        self.assertEqual(_repo_files.submodule_paths(root), {'vendor/dep'})
        for data in ({'version': True, 'paths': ['vendor/dep']},
                     {'version': 1, 'paths': ['../upstream']},
                     {'version': 1, 'paths': ['/outside']},
                     {'version': 1, 'paths': ['missing']},
                     {'version': 1, 'paths': ['vendor/dep', 'vendor/dep']},
                     {'version': 1, 'paths': 'vendor/dep'},
                     {**valid, 'remote': 'unused'}):
            index.write_text(json.dumps(data))
            with self.assertRaises(ValueError): _repo_files.submodule_paths(root)
        (root/'link').symlink_to(root/'vendor/dep', target_is_directory=True)
        index.write_text(json.dumps({'version': 1, 'paths': ['link']}))
        with self.assertRaises(ValueError): _repo_files.submodule_paths(root)
        # A checkout's existing declaration remains authoritative.
        (root/'.gitmodules').write_text('[submodule "declared"]\npath = vendor/declared\n')
        self.assertEqual(_repo_files.submodule_paths(root), {'vendor/declared'})
