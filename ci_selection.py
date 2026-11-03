"""Conservative PR selection; unknown inputs retain the full CI matrix."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


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
