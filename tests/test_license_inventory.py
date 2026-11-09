"""Public licensing evidence stays attached to the checked-in sources.

These checks verify membership, pins, notice bytes and navigable evidence;
they do not decide ownership, derivative-work status or legal permission.
"""

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path

from tests._checkout import needs_checkout
from tests._repo_files import submodule_paths

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'docs/THIRD_PARTY_LICENSES.json'


class LicenseInventory(unittest.TestCase):
    def inventory(self):
        self.assertTrue(INDEX.is_file(), 'public license inventory is missing')
        return json.loads(INDEX.read_text())

    def test_every_submodule_has_one_record(self):
        rows = self.inventory()['modules']
        paths = [row['path'] for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        if not (ROOT / '.gitmodules').is_file():
            self.skipTest('export has no Git submodule declaration')
        pending = ['']
        expected = set()
        while pending:
            parent = pending.pop()
            for child in submodule_paths(ROOT / parent):
                path = (Path(parent) / child).as_posix()
                expected.add(path)
                pending.append(path)
        self.assertEqual(set(paths), expected)

    def test_notice_bytes_and_links_match_checked_out_sources(self):
        for row in self.inventory()['modules']:
            self.assertRegex(row['revision'], r'^[0-9a-f]{40}$')
            self.assertTrue(row['notices'])
            for notice in row['notices']:
                with self.subTest(path=row['path'], notice=notice['path']):
                    path = ROOT / row['path'] / notice['path']
                    self.assertTrue(path.is_file(), 'initialize recursive submodules')
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), notice['sha256'])
                    self.assertEqual(notice['url'], row['repository'] + '/blob/' +
                                     row['revision'] + '/' + notice['path'])

    @needs_checkout
    def test_pins_and_notice_membership_match_git_objects(self):
        rows = self.inventory()['modules']
        for row in rows:
            with self.subTest(path=row['path']):
                parent = ROOT / row['parent']
                rel = Path(row['path']).relative_to(row['parent'] or '.')
                entry = subprocess.check_output(
                    ['git', 'ls-files', '--stage', '--', str(rel)], cwd=parent, text=True)
                self.assertEqual(entry.split()[:2], ['160000', row['revision']])
                module = ROOT / row['path']
                self.assertTrue((module / '.git').exists(), 'initialize recursive submodules')
                names = subprocess.check_output(
                    ['git', 'ls-tree', '-r', '--name-only', row['revision']],
                    cwd=module, text=True).splitlines()
                expected = {p for p in names if re.match(
                    r'^(?:.*[-_.])?(?:licen[sc]e|copying|notice|acknowledgements)(?:[._-].*)?$',
                    Path(p).name, re.IGNORECASE)}
                self.assertEqual({n['path'] for n in row['notices']}, expected)
                self.assertTrue(expected, 'no notice evidence recorded')
                for notice in row['notices']:
                    raw = subprocess.check_output(
                        ['git', 'show', row['revision'] + ':' + notice['path']], cwd=module)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), notice['sha256'])
                    self.assertEqual(notice['url'], row['repository'] + '/blob/' +
                                     row['revision'] + '/' + notice['path'])

    def test_historical_findings_have_current_evidence_not_a_clearance(self):
        rows = self.inventory()['historical_findings']
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['id'] for row in rows}), len(rows))
        for row in rows:
            self.assertEqual(row['permission_status'], 'unresolved')
            self.assertRegex(row['upstream_revision'], r'^[0-9a-f]{40}$')
            self.assertTrue(row['evidence'])
            for item in row['evidence']:
                path = ROOT / item['path']
                self.assertTrue(path.is_file(), str(path))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])

    def test_readme_links_to_public_inventory_and_local_links_resolve(self):
        target = 'docs/THIRD_PARTY_LICENSES.md'
        readme = (ROOT / 'README.md').read_text()
        section = readme.split('## License\n', 1)[1].split('\n## ', 1)[0]
        self.assertIn(target, re.findall(r'\]\(([^)]+)\)', section))
        doc = ROOT / target
        self.assertTrue(doc.is_file(), 'public license explanation is missing')
        links = re.findall(r'\]\(([^)]+)\)', doc.read_text())
        self.assertTrue(links)
        for link in links:
            if '://' not in link and not link.startswith('#'):
                self.assertTrue((doc.parent / link.split('#')[0]).exists(), link)


if __name__ == '__main__':
    unittest.main()
