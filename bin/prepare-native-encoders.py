#!/usr/bin/env python3
"""Prepare a batch of native encoders using pinned per-method provenance.

The JSON sources file maps method names to user-supplied checkpoint paths.
Run with a Python environment compatible with all selected methods. Every
export runs in a fresh process. Output must be new; partial failures are
reported and never silently reused. Logs can contain private paths: keep the
entire output local. Public weight acquisition uses fetch-weights.py first.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def plan(sources, methods: Path, out: Path, python: str):
    if not isinstance(sources, dict) or not sources:
        raise ValueError('sources must be a nonempty method-to-path mapping')
    if out.exists():
        raise ValueError('output already exists; choose a new directory')
    jobs = []
    for name, source in sorted(sources.items()):
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_]+', name):
            raise ValueError('invalid method name')
        method = methods / name
        if method.resolve().parent != methods.resolve():
            raise ValueError('method escapes methods directory')
        if not isinstance(source, str) or not Path(source).is_file():
            raise ValueError(f'{name}: source is not a file')
        artifact = json.loads((method / 'provenance.json').read_text())['step1_native_artifact']
        sha = artifact.get('sha256', '')
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError(f'{name}: invalid pinned sha256')
        for key in ('state_key', 'strip_prefix'):
            if not isinstance(artifact.get(key), str):
                raise ValueError(f'{name}: explicit {key} required')
        config = method / 'configs/linear_eval.yaml'
        if not config.is_file():
            raise ValueError(f'{name}: missing linear_eval config')
        command = [python, str(ROOT / 'bin/export-native-encoder.py'),
                   '--source', source, '--sha256', sha,
                   '--method-dir', str(method), '--config', str(config),
                   '--out', str(out / name), '--state-key', artifact['state_key'],
                   '--strip-prefix', artifact['strip_prefix']]
        jobs.append((name, command))
        if 'module_map' in artifact:
            command.extend(['--module-map', json.dumps(artifact['module_map'])])
    return jobs


def execute(sources, methods: Path, out: Path, python: str):
    jobs = plan(sources, methods, out, python)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    (out / 'batch.json').write_text(json.dumps({'status': 'incomplete', 'methods': []}) + '\n')
    for name, command in jobs:
        with (out / f'{name}.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                code = result.returncode
            except OSError as exc:
                log.write(str(exc))
                code = 127
        records.append({'method': name, 'returncode': code})
        report = {'status': 'failed' if any(r['returncode'] for r in records) else 'ok',
                  'methods': records}
        if len(records) != len(jobs):
            report['status'] = 'incomplete'
        (out / 'batch.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods-dir', type=Path, default=ROOT / 'methods')
    args = parser.parse_args(argv)
    report = execute(json.loads(args.sources.read_text()), args.methods_dir, args.out, sys.executable)
    print(json.dumps(report))
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
