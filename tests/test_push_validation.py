"""Offline validation receipts must gate real pushes without rerunning tests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests._checkout import needs_git

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'bin' / 'validate-push.py'


@needs_git
class TestPushValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='push-receipt-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'work'; self.root.mkdir()
        self.remote = Path(self.tmp.name) / 'remote.git'
        self.env = {k:v for k,v in os.environ.items() if not k.startswith(('GIT_', 'SSH_'))
                    and k not in ('TORCH_GATE_PYTHON','SKIP_TORCH_GATE','PYTHONPATH','PYTHONPYCACHEPREFIX')}
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'
        self.git('init','-q','-b','feature')
        self.git('config','user.name','Fixture'); self.git('config','user.email','fixture@example.invalid')
        subprocess.run(['git','init','-q','--bare',str(self.remote)],check=True,capture_output=True,env=self.env)
        self.git('remote','add','origin',str(self.remote))
        (self.root/'.githooks').mkdir(); (self.root/'bin').mkdir(); (self.root/'tests').mkdir()
        shutil.copy(ROOT/'.githooks/pre-push',self.root/'.githooks/pre-push')
        if TOOL.exists(): shutil.copy(TOOL,self.root/'bin/validate-push.py')
        shutil.copy(ROOT/'tests/_repo_files.py',self.root/'tests/_repo_files.py')
        (self.root/'.gitignore').write_text('.venvs/\n__pycache__/\n*.pyc\n')
        (self.root/'torch.py').write_text('__version__ = "fixture-only"\n')
        (self.root/'source.py').write_text('VALUE = 1\n')
        (self.root/'tests/test_a.py').write_text(
            'import os, pathlib, unittest\nclass A(unittest.TestCase):\n'
            ' def test_first(self):\n'
            '  p=pathlib.Path(".git/suite-count"); p.write_text(p.read_text()+"x" if p.exists() else "x")\n'
            '  os.environ["FIXTURE_SAME_PROCESS"]="yes"\n'
            '  self.assertFalse(pathlib.Path(".git/fail-suite").exists(), "fixture failure")\n')
        (self.root/'tests/test_b.py').write_text(
            'import os, unittest\nclass B(unittest.TestCase):\n'
            ' def test_second(self): self.assertEqual(os.environ.get("FIXTURE_SAME_PROCESS"),"yes")\n')
        self.py = self.root/'.venvs/fixture/bin/python'; self.py.parent.mkdir(parents=True)
        # A real interpreter, with only standard-library and fixture modules. Not a fake exit code.
        import shlex
        self.py.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' -S "$@"\n'); self.py.chmod(0o755)
        self.env['TORCH_GATE_PYTHON'] = str(self.py)
        self.commit()
        self.git('config','core.hooksPath','.githooks')

    def git(self,*args,check=True):
        return subprocess.run(['git',*args],cwd=self.root,env=self.env,capture_output=True,text=True,check=check)

    def commit(self):
        self.git('add','-A'); self.git('commit','-qm','Fixture state')

    def command(self,action,*args):
        self.assertTrue((self.root/'bin/validate-push.py').is_file(),'offline validation command is missing')
        return subprocess.run([sys.executable,str(self.root/'bin/validate-push.py'),action,*args],
                              cwd=self.root,env=self.env,capture_output=True,text=True)

    def prepare(self,force=False):
        r=self.command('prepare',*(['--force'] if force else []))
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        return r

    def count(self):
        p=self.root/'.git/suite-count'; return len(p.read_text()) if p.exists() else 0

    def receipt(self): return self.root/'.git/push-validation/receipt.json'

    def test_unprepared_push_fails_without_running_tests_or_updating_remote(self):
        r=self.git('push','origin','HEAD:refs/heads/feature',check=False)
        self.assertNotEqual(r.returncode,0,'unvalidated push was accepted')
        self.assertEqual(self.count(),0,'pre-push must never start the expensive suite')
        self.assertNotEqual(subprocess.run(['git','--git-dir',str(self.remote),'rev-parse','--verify',
                                           'refs/heads/feature'],capture_output=True).returncode,0)

    def test_prepare_is_offline_and_push_reuses_success_in_one_process(self):
        self.git('remote','set-url','origin',str(self.remote/'missing'))
        self.prepare(); self.assertEqual(self.count(),1)
        self.prepare(); self.assertEqual(self.count(),1)
        self.git('remote','set-url','origin',str(self.remote))
        r=self.git('push','origin','HEAD:refs/heads/feature',check=False)
        self.assertEqual(r.returncode,0,r.stderr); self.assertEqual(self.count(),1)
        record=json.loads(self.receipt().read_text()); self.assertEqual(record['exit_code'],0)
        self.assertEqual(record['head'],self.git('rev-parse','HEAD').stdout.strip())
        self.assertEqual(self.receipt().stat().st_mode & 0o777,0o600)

    def test_transport_failure_then_wrapper_retry_does_not_rerun(self):
        self.prepare(); self.git('remote','set-url','origin',str(self.remote/'missing'))
        self.assertNotEqual(self.command('push','--remote','origin').returncode,0)
        self.assertEqual(self.count(),1)
        self.git('remote','set-url','origin',str(self.remote))
        r=self.command('push','--remote','origin'); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(self.count(),1)

    def test_root_changes_new_commit_and_untracked_files_invalidate(self):
        self.prepare()
        p=self.root/'source.py'; original=p.read_bytes(); old=p.stat()
        p.write_bytes(original.replace(b'1',b'2')); os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
        self.assertNotEqual(self.command('check').returncode,0)
        p.write_bytes(original); (self.root/'unexpected.py').write_text('x=1\n')
        self.assertNotEqual(self.command('check').returncode,0)
        (self.root/'unexpected.py').unlink(); self.git('commit','--allow-empty','-qm','New commit')
        self.assertNotEqual(self.command('check').returncode,0)
        self.assertEqual(self.count(),1)

    def test_failure_revokes_previous_success_and_keeps_diagnostics(self):
        self.prepare(); (self.root/'.git/fail-suite').touch()
        r=self.command('prepare','--force'); self.assertNotEqual(r.returncode,0)
        self.assertFalse(self.receipt().exists()); self.assertIn('fixture failure',r.stdout+r.stderr)
        self.assertNotEqual(self.command('check').returncode,0)
        self.assertEqual(self.count(),2)

    def test_source_change_during_suite_cannot_issue_receipt(self):
        p=self.root/'tests/test_b.py'
        p.write_text(p.read_text()+'\nclass Change(unittest.TestCase):\n def test_change(self):\n  from pathlib import Path\n  Path("source.py").write_text("VALUE = 9\\n")\n')
        self.commit(); r=self.command('prepare')
        self.assertNotEqual(r.returncode,0); self.assertFalse(self.receipt().exists())
        self.assertEqual(self.count(),1)

    def test_corrupt_expired_or_missing_log_records_fail_closed(self):
        self.prepare(); raw=self.receipt().read_text(); record=json.loads(raw)
        for change in [{'version':999},{'exit_code':1},{'created_at':0},{'created_at':10**20},
                       {'fingerprint':'wrong'},{'log_sha256':'wrong'},{'log':'../../outside'}]:
            with self.subTest(change=change):
                self.receipt().write_text(json.dumps({**record,**change}))
                self.assertNotEqual(self.command('check').returncode,0)
        self.receipt().write_text('{'); self.assertNotEqual(self.command('check').returncode,0)
        self.receipt().write_text(raw)
        (self.receipt().parent/record['log']).unlink()
        self.assertNotEqual(self.command('check').returncode,0)

    def test_changed_interpreter_and_test_environment_invalidate(self):
        self.prepare(); self.py.write_text(self.py.read_text()+'# changed\n')
        self.assertNotEqual(self.command('check').returncode,0)
        self.prepare(); self.env['OMP_NUM_THREADS']='7'
        self.assertNotEqual(self.command('check').returncode,0)

    def test_receipt_is_bound_to_checkout_and_actual_pushed_commit(self):
        self.prepare()
        self.git('branch','other'); self.git('commit','--allow-empty','-qm','Head advances'); self.prepare()
        r=self.git('push','origin','other:refs/heads/other',check=False)
        self.assertNotEqual(r.returncode,0,'receipt for HEAD authorized another commit')
        self.assertEqual(self.count(),2)

    def test_skip_flag_and_missing_torch_cannot_authorize_push(self):
        self.env['SKIP_TORCH_GATE']='1'
        self.assertNotEqual(self.git('push','origin','HEAD:refs/heads/feature',check=False).returncode,0)
        self.env.pop('SKIP_TORCH_GATE'); (self.root/'torch.py').unlink(); self.commit()
        self.assertNotEqual(self.command('prepare').returncode,0)
        self.assertFalse(self.receipt().exists()); self.assertEqual(self.count(),0)

    def test_dirty_submodule_content_is_bound_without_modifying_it(self):
        upstream=Path(self.tmp.name)/'upstream'; upstream.mkdir()
        subprocess.run(['git','init','-q',str(upstream)],check=True,capture_output=True,env=self.env)
        for key,value in [('user.name','Fixture'),('user.email','fixture@example.invalid')]:
            subprocess.run(['git','-C',str(upstream),'config',key,value],check=True,env=self.env)
        (upstream/'code.py').write_text('x=1\n')
        for args in [('add','.'),('commit','-qm','Upstream')]:
            subprocess.run(['git','-C',str(upstream),*args],check=True,env=self.env)
        self.git('-c','protocol.file.allow=always','submodule','add',str(upstream),'vendor'); self.commit()
        p=self.root/'vendor/code.py'; p.write_text('x=2\n'); self.prepare()
        self.assertEqual(p.read_text(),'x=2\n'); p.write_text('x=3\n')
        self.assertNotEqual(self.command('check').returncode,0)
        self.assertEqual(self.count(),1)

    def test_prepare_rejects_dirty_root_and_disabled_hook(self):
        (self.root/'source.py').write_text('VALUE = 2\n')
        self.assertNotEqual(self.command('prepare').returncode,0)
        self.assertEqual(self.count(),0)
        self.git('restore','source.py'); self.git('config','core.hooksPath','missing')
        self.assertNotEqual(self.command('push').returncode,0)
        self.assertEqual(self.count(),0)

    def test_prepare_lock_prevents_concurrent_validation(self):
        import fcntl
        state=self.receipt().parent; state.mkdir(parents=True)
        with (state/'lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertNotEqual(self.command('prepare').returncode,0)
        self.assertEqual(self.count(),0)

    def test_hook_clears_git_environment(self):
        self.prepare()
        head=self.git('rev-parse','HEAD').stdout.strip()
        env={**self.env,'GIT_DIR':'/missing','GIT_WORK_TREE':'/missing','GIT_INDEX_FILE':'/missing'}
        r=subprocess.run(['bash','.githooks/pre-push'],cwd=self.root,env=env,text=True,
                         input=f'HEAD {head} refs/heads/feature {"0"*40}\n',capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr); self.assertEqual(self.count(),1)

    def test_dependency_byte_change_invalidates_even_with_same_timestamp(self):
        deps=self.root/'.venvs/deps'; deps.mkdir()
        dep=deps/'library.py'; dep.write_text('VALUE = 1\n')
        # Add an actual import search path without enabling the host site packages.
        original=(self.root/'torch.py').read_text()
        (self.root/'torch.py').write_text('import sys\nsys.path.append('+repr(str(deps))+')\n'+original)
        self.commit(); self.prepare(); before=dep.stat()
        dep.write_text('VALUE = 2\n'); os.utime(dep,ns=(before.st_atime_ns,before.st_mtime_ns))
        self.assertNotEqual(self.command('check').returncode,0)


    def test_remote_option_cannot_bypass_hook(self):
        self.assertNotEqual(self.command('push','--remote=--no-verify').returncode,0)
        self.assertEqual(self.count(),0)

    def test_receipt_copied_to_another_checkout_is_rejected(self):
        self.prepare()
        copy=Path(self.tmp.name)/'copy'
        shutil.copytree(self.root,copy,symlinks=True)
        receipt=copy/'.git/push-validation/receipt.json'
        record=json.loads(receipt.read_text()); record['python']=str(copy/'.venvs/fixture/bin/python')
        receipt.write_text(json.dumps(record))
        env={**self.env,'TORCH_GATE_PYTHON':record['python']}
        r=subprocess.run([sys.executable,str(copy/'bin/validate-push.py'),'check'],cwd=copy,
                         env=env,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)



    def test_explicit_python_override_is_used_for_actual_push(self):
        self.env['TORCH_GATE_PYTHON']='/missing/interpreter'
        r=self.command('push','--python',str(self.py))
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(self.count(),1)



if __name__=='__main__': unittest.main()
