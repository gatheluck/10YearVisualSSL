"""Native Cityscapes train-ID membership, separate from score attribution."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from downstream import extended_segmentation_membership as membership


class CityscapesMembership(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split, city in [('train', 'alpha'), ('val', 'beta')]:
            self.put(f'leftImg8bit/{split}/{city}/{city}_000000_000001_leftImg8bit.png', split.encode())
            self.put(f'gtFine/{split}/{city}/{city}_000000_000001_gtFine_labelTrainIds.png', b'mask')

    def put(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def build(self):
        return membership.build('cityscapes', self.root, fixture_counts=(1, 1))

    def test_native_join_and_identity_label_map(self):
        result = self.build()
        self.assertEqual(result['classes'], [str(i) for i in range(19)])
        self.assertEqual(result['label_map'], {**{str(i): i for i in range(19)}, '255': 255})
        self.assertEqual(result['train'][0]['image'], 'leftImg8bit/train/alpha/alpha_000000_000001_leftImg8bit.png')
        self.assertEqual(result['validation'][0]['mask'], 'gtFine/val/beta/beta_000000_000001_gtFine_labelTrainIds.png')
        self.assertNotIn(str(self.root), json.dumps(result))
        evidence = json.loads(result['split_evidence'])
        self.assertTrue(evidence['fixture'])
        self.assertFalse(evidence['authenticity_verified'])
        self.assertFalse(evidence['mask_values_verified'])

    def test_release_counts_and_no_fixture_cli(self):
        self.assertEqual(membership.COUNTS['cityscapes'], (2975, 500, 19))
        with self.assertRaisesRegex(ValueError, 'split count'):
            membership.build('cityscapes', self.root)
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps({'dataset': 'cityscapes', 'data_root': str(self.root)}))
        out = self.root / 'samples.json'
        with self.assertRaisesRegex(ValueError, 'split count'):
            membership.main(['--config', str(cfg), '--out', str(out)])
        self.assertFalse(out.exists())

    def test_raw_ids_are_not_a_fallback(self):
        path = next((self.root / 'gtFine/train').rglob('*labelTrainIds.png'))
        path.rename(path.with_name(path.name.replace('labelTrainIds', 'labelIds')))
        with self.assertRaisesRegex(ValueError, 'labelTrainIds'):
            self.build()

    def test_extra_unpaired_trainid_mask_is_rejected(self):
        self.put('gtFine/train/alpha/extra_gtFine_labelTrainIds.png', b'extra')
        with self.assertRaisesRegex(ValueError, 'pair'):
            self.build()

    def test_nested_city_identity_is_not_flattened(self):
        self.put('leftImg8bit/train/gamma/alpha_000000_000001_leftImg8bit.png', b'train2')
        self.put('gtFine/train/gamma/alpha_000000_000001_gtFine_labelTrainIds.png', b'mask2')
        result = membership.build('cityscapes', self.root, fixture_counts=(2, 1))
        self.assertEqual(len(result['train']), 2)
        self.assertIn('/gamma/', result['train'][1]['image'])

    def test_escaping_image_symlink_is_rejected(self):
        path = next((self.root / 'leftImg8bit/train').rglob('*.png'))
        with tempfile.TemporaryDirectory() as folder:
            outside = Path(folder) / 'external.png'
            outside.write_bytes(b'external')
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, 'escaping'):
                self.build()

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'),
                         'semantic reader requires the downstream environment')
    def test_manifest_loads_real_pngs_and_rejects_unknown_mask_ids(self):
        import torch
        from PIL import Image

        from downstream.extended_segmentation import load_data

        for path in (self.root / 'leftImg8bit').rglob('*.png'):
            Image.new('RGB', (19, 13), (13, 57, 91)).save(path)
        for path in (self.root / 'gtFine').rglob('*.png'):
            mask = Image.new('L', (19, 13), 18)
            mask.paste(255, (0, 0, 9, 13))
            mask.save(path)
        path = self.root / 'samples.json'
        path.write_text(json.dumps(self.build()))
        _, validation, meta = load_data(path, self.root, 'captured_pair_crop224_v1')
        image, mask = validation[0]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(set(torch.unique(mask).tolist()), {18, 255})
        self.assertEqual(meta['validation_count'], 1)
        bad = next((self.root / 'gtFine/val').rglob('*.png'))
        Image.new('L', (19, 13), 19).save(bad)
        with self.assertRaisesRegex(ValueError, 'unknown native mask ID'):
            validation[0]


if __name__ == '__main__':
    unittest.main()
