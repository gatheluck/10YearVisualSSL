#!/usr/bin/env python3
"""Run the whole Torch suite offline and bind its success to the pushed content."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from _repo_files import LS_FILES  # noqa: E402

COMMAND = ['-m', 'unittest', 'discover', '-s', 'tests']
MAX_AGE = 24 * 60 * 60
VOLATILE = {'PWD', 'OLDPWD', 'SHLVL', '_', 'TERM', 'COLUMNS', 'LINES',
            'TORCH_GATE_PYTHON', 'SKIP_TORCH_GATE', 'PYTHONPYCACHEPREFIX'}


class Invalid(RuntimeError):
    """Validation cannot authorize this push."""


@lru_cache(maxsize=1)
def test_environment():
    # Git's launcher can add SDK variables (macOS) and prepend its exec path.
    # Normalize through the same local launcher before testing and checking.
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(('GIT_', 'SSH_')) and k not in VOLATILE}
    result = subprocess.check_output(
        ['git', '-c', 'alias.push-validation-env=!env -0', 'push-validation-env'],
        cwd=ROOT, env=clean, stderr=subprocess.PIPE)
    launched = dict(os.fsdecode(item).split('=', 1) for item in result.split(b'\0') if item)
    env = {k: v for k, v in launched.items()
           if not k.startswith(('GIT_', 'SSH_')) and k not in VOLATILE}
    env['PATH'] = os.pathsep.join(dict.fromkeys(env['PATH'].split(os.pathsep)))
    env.update(PYTHONPATH='.', PYTHONDONTWRITEBYTECODE='1')
    return env


def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root,
                                   env=test_environment(), stderr=subprocess.PIPE)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    link = os.readlink(path) if path.is_symlink() else None
    if path.is_file():
        return [link, path.stat().st_mode & 0o777, sha(path)]
    # External directory symlinks and missing targets are recorded, not traversed.
    return [link, 'directory' if path.is_dir() else 'absent']


def source_identity(root):
    entries = git('ls-files', '--stage', '-z', root=root)
    links = {os.fsdecode(row.split(b'\t', 1)[1]) for row in entries.split(b'\0')
             if row.startswith(b'160000 ')}
    records = [git('rev-parse', 'HEAD', root=root).decode().strip(), entries.hex()]
    for name in sorted(set(git(*LS_FILES[1:], root=root).split(b'\0')) - {b''}):
        rel = os.fsdecode(name)
        path = root / rel
        if rel in links and (path / '.git').exists():
            value = source_identity(path)
        else:
            value = file_identity(path)
        records.append([rel, value])
    return records


def runtime_identity(python, env):
    # Fresh bytecode location also prevents stale ignored caches from masking edits.
    probe = ('import torch, sys, json, platform; '
             'print(json.dumps(dict(paths=sys.path, executable=sys.executable, '
             'prefix=sys.prefix, version=sys.version, platform=platform.platform())))')
    result = subprocess.run([python, '-c', probe], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    if result.returncode:
        raise Invalid('Selected Python cannot import torch; no validation was issued.')
    data = json.loads(result.stdout)
    records = [[python, file_identity(Path(python))],
               [data['executable'], file_identity(Path(data['executable']))]]
    cfg = Path(data['prefix']) / 'pyvenv.cfg'
    records.append([str(cfg), file_identity(cfg)])
    for value in sorted(set(data['paths'])):
        path = Path(value or ROOT).resolve()
        if path == ROOT:
            continue
        if not path.is_dir():
            records.append([str(path), file_identity(path)])
            continue
        for folder, dirs, files in os.walk(path, followlinks=False):
            # Site directories are hashed separately only when on sys.path.
            dirs[:] = sorted(d for d in dirs if d not in
                             ('__pycache__', 'site-packages', 'dist-packages'))
            base = Path(folder)
            for name in sorted(files + [d for d in dirs if (base / d).is_symlink()]):
                if name.endswith(('.pyc', '.pyo')):
                    continue
                target = base / name
                records.append([str(target), file_identity(target)])
    return [data, records]


def fingerprint(python, env):
    if git('status', '--porcelain', '--untracked-files=all', '--ignore-submodules=all').strip():
        raise Invalid('Commit root changes before validation; dirty submodules are fingerprinted separately.')
    data = [str(ROOT.resolve()), source_identity(ROOT), runtime_identity(python, env),
            test_environment(), COMMAND]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def state_dir():
    path = Path(os.fsdecode(git('rev-parse', '--git-path', 'push-validation')).strip())
    path = (ROOT / path).resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


@contextmanager
def locked(state):
    with (state / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Invalid('Another validation/check is running; retry after it finishes.') from exc
        yield


def select_python(explicit=None):
    chosen = explicit or os.environ.get('TORCH_GATE_PYTHON')
    candidates = [Path(chosen).absolute()] if chosen else sorted(ROOT.glob('.venvs/*/bin/python'))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            if chosen or subprocess.run([str(candidate), '-c', 'import torch'], cwd=ROOT,
                                        env=test_environment(), capture_output=True).returncode == 0:
                return str(candidate)
    raise Invalid('No Torch interpreter; set TORCH_GATE_PYTHON or --python.')


def verify(state, python, env):
    record = json.loads((state / 'receipt.json').read_text())
    if (record['version'] != 1 or record['exit_code'] != 0 or
            not 0 <= time.time() - record['created_at'] < MAX_AGE):
        raise Invalid('Invalid or expired validation receipt.')
    log = record['log']
    if not isinstance(log, str) or Path(log).name != log or not log.endswith('.log'):
        raise Invalid('Invalid validation log path.')
    if sha(state / log) != record['log_sha256']:
        raise Invalid('Validation log changed.')
    if record['python'] != python or record['head'] != git('rev-parse', 'HEAD').decode().strip():
        raise Invalid('Interpreter or commit changed.')
    if fingerprint(python, env) != record['fingerprint']:
        raise Invalid('Source, dependencies or test environment changed.')
    return record


def prepare(state, python, env, force):
    receipt = state / 'receipt.json'
    if not force:
        try:
            record = verify(state, python, env)
            print('Reusing successful offline validation.', flush=True)
            return record
        except (Invalid, OSError, ValueError, KeyError, TypeError):
            pass
    receipt.unlink(missing_ok=True)
    before = fingerprint(python, env)
    name = uuid.uuid4().hex + '.log'
    log = state / name
    print('Offline whole-suite validation; log: ' + str(log), flush=True)
    with log.open('x') as stream:
        log.chmod(0o600)
        result = subprocess.run([python, *COMMAND], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-30:]), file=sys.stderr)
        raise Invalid('Whole-suite validation failed; previous success revoked.')
    if fingerprint(python, env) != before:
        raise Invalid('Inputs changed during validation; no receipt issued.')
    record = dict(version=1, exit_code=0, created_at=time.time(), python=python,
                  head=git('rev-parse', 'HEAD').decode().strip(), fingerprint=before,
                  log=name, log_sha256=sha(log))
    tmp = state / (uuid.uuid4().hex + '.json')
    with tmp.open('x') as stream:
        tmp.chmod(0o600)
        json.dump(record, stream, indent=2)
    os.replace(tmp, receipt)
    print('Offline validation passed; receipt saved.', flush=True)
    return record


def check_refs(record, lines):
    for line in lines:
        fields = line.split()
        if (len(fields) != 4 or fields[1] != record['head'] or
                not fields[2].startswith('refs/heads/')):
            raise Invalid('Receipt authorizes only the current HEAD commit to branch refs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'push'])
    parser.add_argument('--python')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--remote', default='origin')
    parser.add_argument('--hook', action='store_true')
    args = parser.parse_args()
    try:
        if os.environ.get('SKIP_TORCH_GATE'):
            raise Invalid('SKIP_TORCH_GATE is not supported; validation is required.')
        lines = sys.stdin.read().splitlines() if args.hook else []
        if args.hook and not lines:
            return 0
        if args.action == 'push':
            if not args.remote or args.remote.startswith('-'):
                raise Invalid('Remote must be a name or URL, never a Git option.')
            hooks = Path(os.fsdecode(git('config', '--get', 'core.hooksPath')).strip())
            if (ROOT / hooks).resolve() != ROOT / '.githooks' or not os.access(ROOT / '.githooks/pre-push', os.X_OK):
                raise Invalid('Enable repository hooks with git config core.hooksPath .githooks.')
            branch = git('symbolic-ref', '--short', 'HEAD').decode().strip()
        state = state_dir()
        with locked(state), tempfile.TemporaryDirectory(prefix='push-bytecode-') as cache:
            env = {**test_environment(), 'PYTHONPYCACHEPREFIX': cache}
            if args.action == 'check' and not args.python and not os.environ.get('TORCH_GATE_PYTHON'):
                python = json.loads((state / 'receipt.json').read_text())['python']
            else:
                python = select_python(args.python)
            if args.action == 'check':
                record = verify(state, python, env)
                if args.hook:
                    check_refs(record, lines)
                print('Validation receipt matches; no tests rerun.', flush=True)
            else:
                prepare(state, python, env, args.force)
        if args.action == 'push':
            # No network operation occurs until successful local preparation.
            return subprocess.run(['git', 'push', '-u', args.remote,
                                   'HEAD:refs/heads/' + branch], cwd=ROOT,
                                  env={**os.environ, 'TORCH_GATE_PYTHON': python}).returncode
        return 0
    except (Invalid, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print('Push validation refused: ' + str(exc) + '\nRun python3 bin/validate-push.py prepare before pushing.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
