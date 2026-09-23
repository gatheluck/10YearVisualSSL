#!/usr/bin/env python3
"""Audit committed source and create a deterministic anonymous submission ZIP.

The JSON policy and audit report are private, outside the source checkout.
This is a conservative disclosure gate, not proof of anonymity or licensing.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import unicodedata
from urllib.parse import unquote
import zipfile

METADATA = {'.git', '.gitmodules', '.github', '.githooks', '.gitignore', '.gitattributes'}
APPROVABLE = {'email', 'external-link', 'binary', 'opaque'}
RULES = {
    'email': r'(?<![\w.+-])[\w.+-]{1,254}@[\w.-]{1,254}\.[a-z]{2,}',
    'local-path': r'/(?:users|home|groups|scratch)/[^\s"\']+',
    'secret': r'-----begin (?:[a-z0-9]+ )*private key-----|\b(?:ghp|github_pat)_[a-z0-9_]{20,}',
    'external-link': r'(?:https?|ssh|git|ftp)://[^\s<>"\']+|git@[^\s]+',
    'opaque': r'data:[^;\s]+;base64,',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(text):
    for _ in range(3):
        text = html.unescape(unquote(text))
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if unicodedata.category(c) != 'Cf')


def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('policy paths must be nonempty POSIX relative paths')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('unsafe policy path')
    return value


def policy_check(policy):
    required = {'version', 'include', 'exclude', 'forbidden', 'replacements', 'approvals', 'max_bytes'}
    if (not isinstance(policy, dict) or set(policy) != required or
            type(policy['version']) is not int or policy['version'] != 1):
        raise ValueError('unknown or missing policy fields/version')
    for key in ('include', 'exclude'):
        if not isinstance(policy[key], list) or (key == 'include' and not policy[key]):
            raise ValueError('include/exclude must be lists; include cannot be empty')
        for p in policy[key]:
            safe_path(p)
        if len(set(policy[key])) != len(policy[key]):
            raise ValueError('duplicate policy paths')
    if not isinstance(policy['forbidden'], list) or not policy['forbidden'] or any(
            not isinstance(x, str) or not canonical(x).strip() for x in policy['forbidden']):
        raise ValueError('explicit private identifying terms are required')
    if type(policy['max_bytes']) is not int or policy['max_bytes'] < 1:
        raise ValueError('max_bytes must be a positive integer')
    for section in ('replacements', 'approvals'):
        if not isinstance(policy[section], dict):
            raise ValueError('policy edits must be mappings')
        for path, entry in policy[section].items():
            safe_path(path)
            fields = {'sha256', 'reason', 'text' if section == 'replacements' else 'rules'}
            if not isinstance(entry, dict) or set(entry) != fields:
                raise ValueError('unknown or missing edit fields')
            if not isinstance(entry['reason'], str) or not entry['reason'].strip():
                raise ValueError('review reason is required')
            if not isinstance(entry['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
                raise ValueError('edit must pin a SHA256')
            if section == 'replacements' and not isinstance(entry['text'], str):
                raise ValueError('replacement must be text')
            if section == 'approvals' and (not isinstance(entry['rules'], list) or not entry['rules'] or
                    any(not isinstance(x, str) or x not in APPROVABLE for x in entry['rules'])):
                raise ValueError('invalid approval rule; identifiers cannot be waived')


def selected(path, prefixes):
    return any(path == p or path.startswith(p + '/') for p in prefixes)


def license_file(path):
    name = PurePosixPath(path).name.casefold()
    return any(name == x or name.startswith(x + '.') or name.startswith(x + '-')
               for x in ('license', 'licence', 'copying', 'copyright', 'notice'))


def git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.check_output(['git', '-C', str(root), *args], env=env, stderr=subprocess.PIPE)


def collect(root, revision, policy, finding, excluded, pins):
    """Read committed blobs, including initialized gitlinks at their recorded pin.

    This enumerates a requested immutable export, not the working-tree ownership
    scan in tests._repo_files. Neither untracked nor dirty bytes are inputs.
    """
    files = {}
    total = 0
    matched = set()

    def walk(repo, commit, prefix=''):
        nonlocal total
        entries = git(repo, 'ls-tree', '-rz', '--full-tree', commit).split(b'\0')
        regular = []
        for entry in entries:
            if not entry:
                continue
            header, rawname = entry.split(b'\t', 1)
            mode, kind, oid = header.decode('ascii').split()
            name = rawname.decode('utf-8')
            path = prefix + name
            safe_path(path)
            for item in policy['include']:
                if path == item or path.startswith(item + '/'):
                    matched.add(item)
            wanted = selected(path, policy['include'])
            if kind == 'commit':
                wanted = wanted or any(item.startswith(path + '/') for item in policy['include'])
                if not wanted or selected(path, policy['exclude']):
                    excluded.append({'path': path, 'reason': 'selection'})
                    continue
                checkout = repo / name
                try:
                    if checkout.is_symlink() or not checkout.resolve().is_relative_to(root):
                        raise ValueError('unsafe submodule checkout')
                    actual = Path(git(checkout, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
                    if actual != checkout.resolve():
                        raise ValueError('uninitialized submodule')
                    git(checkout, 'cat-file', '-e', oid + '^{commit}')
                    pins.append({'path': path, 'commit': oid})
                    walk(checkout, oid, path + '/')
                except (OSError, subprocess.CalledProcessError, ValueError):
                    finding(path, 'missing-submodule')
                continue
            # Retain license notices in every visited repository, even if the
            # content allowlist omits them. Explicit exclusion is a blocker.
            notice = license_file(path)
            if notice and selected(path, policy['exclude']):
                finding(path, 'license-exclusion')
            if any(part.casefold() in METADATA for part in PurePosixPath(path).parts):
                excluded.append({'path': path, 'reason': 'git-metadata'})
                continue
            if (not wanted and not notice) or (selected(path, policy['exclude']) and not notice):
                excluded.append({'path': path, 'reason': 'selection'})
                continue
            if mode == '120000':
                finding(path, 'symlink')
                continue
            if mode not in ('100644', '100755') or kind != 'blob':
                finding(path, 'unsupported-entry')
                continue
            regular.append((path, mode, oid))
        # Batch object reads avoid a new process per source file. Drain each
        # response before writing the next request, so pipes cannot deadlock.
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        with subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, env=env) as proc:
            try:
                for path, mode, oid in regular:
                    proc.stdin.write((oid + '\n').encode()); proc.stdin.flush()
                    header = proc.stdout.readline().split()
                    if len(header) != 3 or header[1] != b'blob':
                        raise ValueError('unavailable Git blob')
                    size = int(header[2])
                    total += size
                    if total > policy['max_bytes']:
                        finding(path, 'size-limit')
                        proc.kill()
                        return
                    data = proc.stdout.read(size)
                    if len(data) != size or proc.stdout.read(1) != b'\n':
                        raise ValueError('truncated Git blob')
                    files[path] = (mode, data)
            finally:
                proc.stdin.close()
                if proc.poll() is None:
                    proc.wait()
            if proc.returncode:
                raise ValueError('Git blob reader failed')

    walk(root, revision)
    for path in sorted(set(policy['include']) - matched):
        finding(path, 'missing-selection')
    return files


def atomic_report(path, report):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write((json.dumps(report, indent=2, ensure_ascii=True) + '\n').encode())
            f.flush()
            os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def build(repo, policy, report_path, output=None, ref='HEAD'):
    policy_check(policy)
    root = Path(repo).resolve()
    actual_root = Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    if actual_root != root:
        raise ValueError('repo must name the checkout root, not a subdirectory')
    report_path = Path(report_path).resolve()
    output = Path(output).absolute() if output is not None else None
    for p in (report_path, output):
        if p is not None and (p.resolve().is_relative_to(root) or not p.parent.is_dir()):
            raise ValueError('outputs must use existing directories outside the source checkout')
    if output is not None and (output.resolve() == report_path or os.path.lexists(output)):
        raise FileExistsError('output already exists or aliases report')
    revision = git(root, 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}').decode().strip()
    findings, excluded, pins = [], [], []

    def finding(path, rule):
        entry = {'path': path, 'rule': rule}
        if entry not in findings:
            findings.append(entry)

    if output is not None:
        if any(canonical(term) in canonical(output.name) for term in policy['forbidden']):
            finding('(archive filename)', 'identifier')
    files = collect(root, revision, policy, finding, excluded, pins)
    for section in ('replacements', 'approvals'):
        for path in policy[section]:
            if path not in files:
                finding(path, 'unused-policy')
    packed, names, records = {}, {}, []
    for path, (mode, original) in sorted(files.items()):
        normalized = canonical(path)
        if normalized in names:
            finding(path, 'path-collision')
        names[normalized] = path
        bad_component = any(part.endswith(('.', ' ')) or re.fullmatch(
            r'(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', part.casefold())
            for part in PurePosixPath(path).parts)
        if any(ord(c) < 32 for c in path) or ':' in path or bad_component:
            finding(path, 'unsafe-archive-path')
        data = original
        replacement = policy['replacements'].get(path)
        if replacement:
            if license_file(path) or re.search(
                    rb'(?im)^\s*(?:#|//|\*)?\s*(?:copyright|spdx-license-identifier)\b', original):
                finding(path, 'license-replacement')
            elif digest(original) != replacement['sha256']:
                finding(path, 'stale-replacement')
            else:
                data = replacement['text'].encode('utf-8')
        approved = set()
        approval = policy['approvals'].get(path)
        if approval:
            if digest(data) != approval['sha256']:
                finding(path, 'stale-approval')
            else:
                approved = set(approval['rules'])
        # Even reviewed binaries receive a best-effort visible-byte scan.
        text = data.decode('utf-8', errors='replace')
        scan = canonical(path + '\n' + text)
        if any(canonical(term) in scan for term in policy['forbidden']):
            finding(path, 'identifier')
        for rule, pattern in RULES.items():
            if re.search(pattern, scan) and rule not in approved:
                finding(path, rule)
        if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
            finding(path, 'lfs-pointer')
        try:
            data.decode('utf-8')
            binary = b'\x00' in data
        except UnicodeDecodeError:
            binary = True
        if binary and 'binary' not in approved:
            finding(path, 'binary')
        if path.casefold().endswith('.ipynb') and 'opaque' not in approved:
            finding(path, 'opaque')
        packed[path] = (mode, data)
        records.append({'path': path, 'source_sha256': digest(original), 'archive_sha256': digest(data),
                        'replaced': bool(replacement), 'approved_rules': sorted(approved)})
    if not packed:
        finding('', 'empty-archive')
    if sum(len(data) for _, data in packed.values()) > policy['max_bytes']:
        finding('', 'size-limit')
    report = {'status': 'blocked' if findings else 'audit-passed', 'commit': revision,
              'policy_sha256': digest(json.dumps(policy, sort_keys=True).encode()),
              'working_tree_ignored': True, 'submodules': pins, 'files': records,
              'excluded': excluded, 'findings': findings,
              'limits': 'Not proof of anonymity, legal permission, or scientific reproducibility. Review visible binary content, encoded data, and links manually.'}
    atomic_report(report_path, report)
    if findings:
        return False
    if output is not None:
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
                temp = Path(f.name)
            with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
                for path, (mode, data) in sorted(packed.items()):
                    info = zipfile.ZipInfo('code/' + path, (1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = (0o100755 if mode == '100755' else 0o100644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, data)
            with zipfile.ZipFile(temp) as z:
                if z.testzip() is not None:
                    raise ValueError('ZIP integrity check failed')
            # Link is exclusive: a concurrent writer cannot be overwritten.
            os.link(temp, output)
            report.update(status='created', archive_sha256=digest(temp.read_bytes()))
            atomic_report(report_path, report)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.policy.resolve().is_relative_to(args.repo.resolve()):
            raise ValueError('policy must remain outside the source checkout')
        if args.policy.resolve() == args.report.resolve() or (args.out and args.policy.resolve() == args.out.resolve()):
            raise ValueError('policy must be separate from outputs')
        ok = build(args.repo, json.loads(args.policy.read_text()), args.report, args.out, args.ref)
    except (ValueError, OSError, subprocess.CalledProcessError):
        # Never echo input text, credentials, or Git's private stderr.
        print('Archive not created: invalid policy, source, or output. Check private inputs.')
        return 2
    print('Archive check passed.' if ok else 'Archive blocked; inspect the private report.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
