"""Reference population admission uses the native released V-JEPA name."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HAVE = importlib.util.find_spec('torch') is not None
if HAVE:
    from downstream import revised_tasks as api
    from downstream.revised_recipe import resolve_recipe


class Population:
    def __init__(self, size):
        self.size = size
        self.class_to_idx = {str(i): i for i in range(1000)}

    def __len__(self):
        return self.size


@unittest.skipUnless(HAVE, 'torch required')
class ReleasedGuardTests(unittest.TestCase):
    def spec(self):
        return {'kind': 'vjepa2_1', 'arch': 'vit_giant_xformers',
                'encoder': '/read-only/checkpoint.pt', 'img_size': 384, 'patch_size': 16}

    def prepare(self, spec, populations=None):
        cfg = {'task': 'imagenet', 'backbone': spec}
        pair = tuple(Population(n) for n in (populations or api.POPULATIONS['imagenet']))
        with patch.object(api, 'build_datasets', return_value=pair):
            return api.prepare_datasets(cfg, resolve_recipe('imagenet', 'lora'))

    def test_provider_owns_native_admission(self):
        from downstream import spatial_backbones as spatial
        spec = {"kind": "synthetic", "arch": "native"}
        provider = SimpleNamespace(is_released_spec=lambda value: value == spec)
        with patch.dict(spatial._PROVIDERS, {"synthetic": Path("/synthetic")}), \
             patch.object(spatial, "_load_provider", return_value=provider):
            self.assertEqual(self.prepare(spec)[2], api.POPULATIONS["imagenet"])

    def test_native_released_giant_is_admitted(self):
        self.assertEqual(self.prepare(self.spec())[2], api.POPULATIONS['imagenet'])

    def test_reduced_or_nonreleased_specs_are_rejected(self):
        changes = ({'embed_dim': 16}, {'depth': 2}, {'num_heads': 2},
                   {'arch': 'fixture'}, {'arch': 'vit_large'}, {'kind': 'other'},
                   {'encoder': ''}, {'img_size': 32}, {'patch_size': 8})
        for update in changes:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'reference'):
                self.prepare(self.spec() | update)

    def test_incomplete_population_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, 'population'):
            self.prepare(self.spec(), (1000, 1000))

    def test_existing_released_name_remains_admitted(self):
        self.assertEqual(self.prepare({'kind': 'siglip2_g', 'arch': 'released'})[2],
                         api.POPULATIONS['imagenet'])


if __name__ == '__main__':
    unittest.main()
