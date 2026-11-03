"""Conservative PR selection; unknown inputs retain the full CI matrix."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


def _lock_input(path):
    """Exact non-comment input for our narrow hashed-lock grammar, or None.

    This is deliberately not a general pip parser. Includes, markers, options,
    alternate encodings and interrupted continuations retain separate jobs.
    Preserve requirement order, spelling, whitespace and every wheel hash.
    """
    if path.is_symlink():
        return None
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return None
    if not text.isascii() or re.search(r'coding\s*[:=]', text):
        return None
    if any(ord(c) < 32 and c not in '\t\n\r' for c in text):
        return None
    significant = []
    pending = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            if pending:
                return None
            continue
        pattern = (r'[ \t]+--hash=sha256:[0-9a-f]{64}(?: \\)?' if pending else
                   r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]* \\')
        if not re.fullmatch(pattern, line):
            return None
        significant.append(line)
        pending = line.endswith('\\')
    if pending or not significant:
        return None
    return tuple(significant)


def environment_groups(methods, root=Path('.')):
    """Group selected methods only when both installation inputs are identical.

    The grouped job's runner, interpreter and test commands are shared. Each
    method still gets its own container install, environment check and smoke.
    Unknown inputs are isolated rather than guessed equivalent.
    """
    tools = _lock_input(root / 'requirements-tools.lock.txt')
    groups = {}
    for method in sorted(methods):
        lock = _lock_input(root / 'methods' / method / 'requirements.lock.txt')
        key = (tools, lock) if tools is not None and lock is not None else method
        groups.setdefault(key, []).append(method)
        if tools is None or lock is None:
            print(f'CI environment {method}: unrecognized lock input; keeping its own job',
                  file=sys.stderr)
    result = list(groups.values())
    print('CI environment groups: ' + json.dumps(result), file=sys.stderr)
    return result


def select(methods):
    """Return selected locked environments and whether downstream must run.

    Compare the event base to the actual tested checkout, including merge-base
    updates. Disable rename detection so both sides of a move are considered.
    Missing history, metadata and unexpected paths always request full coverage.
    """
    full = (methods, True)
    if os.environ.get('GITHUB_EVENT_NAME') != 'pull_request':
        return full
    try:
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        base = event['pull_request']['base']['sha']
        if not isinstance(base, str) or not re.fullmatch(r'[0-9a-f]{40}', base):
            raise ValueError('invalid base revision')
        raw = subprocess.check_output(
            ['git', 'diff', '--no-ext-diff', '--no-renames', '--name-only',
             '-z', base, 'HEAD', '--'], stderr=subprocess.PIPE)
        paths = raw.decode('utf-8').split('\0')[:-1]
        if not paths:
            return full
        selected = set()
        for path in paths:
            parts = PurePosixPath(path).parts
            if not parts or any(x in ('', '.', '..') for x in path.split('/')):
                return full
            # A link named README.md can change executable code indirectly.
            for revision in (base, 'HEAD'):
                entry = subprocess.check_output(['git', 'ls-tree', revision, '--', path])
                if entry.startswith((b'120000 ', b'160000 ')):
                    return full
            prose = (path == 'README.md' or
                     (parts[0] == 'docs' and path.endswith('.md')) or
                     (len(parts) == 3 and parts[0] == 'methods' and
                      parts[1] in methods and parts[2] == 'README.md'))
            if prose:
                continue
            if len(parts) >= 3 and parts[0] == 'methods' and parts[1] in methods:
                selected.add(parts[1])
            else:
                return full
        return sorted(selected), bool(selected)
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError) as exc:
        print(f'CI selection unavailable; running all environments: {type(exc).__name__}',
              file=sys.stderr)
        return full
