"""Portable registries preserve the supplied protocol, with relocatable data roots."""
import hashlib
import json
from pathlib import Path
from string import Template
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'submission_protocols'
# Digests of reviewed reference JSON objects with only dataset paths removed.
# Private original bytes and source identities stay outside the repository.
EXPECTED = {'LINEAR': 'efa41faa31fa04cf21de6702faa75a275f1e88cc586cb214eab6e2a530f5fff9', 'ATTENTIVE': 'fa1817b6e1c9a109970ae8245b693c9aa3f8842b3678d48c4a130f06b851976e', 'FINETUNE': '92b24f067f8dbd1576da47a36939345ad024b3a7a7d99ab927aff9b968fe9a9c'}


class TestExtendedRegistries(unittest.TestCase):
    def load(self, track):
        path = ROOT / f'EXTEND_{track}_v1.json'
        self.assertTrue(path.is_file(), 'required protocol registry missing')
        return json.loads(path.read_text())

    def test_non_path_content_matches_reviewed_reference(self):
        for track, digest in EXPECTED.items():
            with self.subTest(track=track):
                data = self.load(track)
                for dataset in data.get('datasets', {}).values():
                    dataset.pop('path')
                actual = hashlib.sha256(json.dumps(
                    data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(actual, digest)

    def test_all_data_paths_are_relocatable_without_original_locations(self):
        data = self.load('LINEAR')
        for key, dataset in data['datasets'].items():
            with self.subTest(dataset=key):
                self.assertEqual(dataset['path'], '${DATA_ROOT}/' + key)
                for root in ('/datasets/location-a', '/datasets/location b'):
                    resolved = Path(Template(dataset['path']).substitute(DATA_ROOT=root))
                    self.assertEqual(resolved.parent, Path(root))
                    self.assertEqual(resolved.name, key)
        for track in EXPECTED:
            text = json.dumps(self.load(track))
            self.assertNotRegex(text, r'/(?:groups|home|Users)/|[A-Za-z]:\\Users\\')

    def test_inheritance_and_recipe_references_resolve_in_same_bundle(self):
        base = self.load('LINEAR')
        self.assertEqual(len(base['datasets']), 81)
        self.assertEqual(base['dataset_policy']['included_dataset_configs'], 81)
        self.assertEqual(len(base['models']), 8)
        for dataset in base['datasets'].values():
            self.assertIn(dataset['recipe'], base['recipes'])
            self.assertTrue(dataset['split'])
            self.assertTrue(dataset['primary_metric'])
        for track in ('ATTENTIVE', 'FINETUNE'):
            child = self.load(track)
            self.assertEqual(child['inherits']['protocol'], base['protocol_id'])
            self.assertEqual(child['models'], base['models'])
            self.assertEqual(child['execution']['seeds'], [0])
            self.assertEqual(set(child['recipe_overrides']), set(base['recipes']))


if __name__ == '__main__':
    unittest.main()
