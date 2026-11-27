"""Check what a reader can see in the parent project's source distribution."""
import json
import re
import unittest
from pathlib import Path

from tests._repo_files import repository_files

ROOT = Path(__file__).resolve().parents[1]


class PublicSourceDisclosures(unittest.TestCase):
    def test_capture_provenance_does_not_publish_the_private_repository_address(self):
        address = re.compile(rb'[\w.-]+/[\w.-]+CapturePrivate\b', re.I)
        affected = [file.relative_to(ROOT).as_posix() for file in repository_files(ROOT)
                    if address.search(file.read_bytes())]
        self.assertEqual(affected, [], f"Private capture addresses in {len(affected)} files")

    def test_mutation_usage_notes_work_outside_the_original_checkout(self):
        affected = []
        for spec in (ROOT / 'mutations').glob('*.json'):
            note = json.loads(spec.read_text()).get('_note', '')
            if re.search(r'/(?:home|Users)/[^/\s]+/', note):
                affected.append(spec.name)
        self.assertEqual(affected, [], f"Machine-specific interpreters in {len(affected)} notes")
