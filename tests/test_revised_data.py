"""Observable image/label geometry of the explicit revised experiment readers."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE = all(importlib.util.find_spec(name) is not None
           for name in ('torch', 'torchvision', 'PIL', 'numpy'))
if HAVE:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import functional as F


@unittest.skipUnless(HAVE, 'torch, torchvision, Pillow and numpy required')
class RevisedReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        y, x = np.indices((281, 397))
        self.rgb = np.stack(((7*x+3*y) % 256, (x+11*y) % 256,
                             (13*x+17*y) % 256), axis=-1).astype('uint8')
        self.labels = np.choose((x//17+y//23) % 6,
                                [0, 1, 2, 150, 151, 255]).astype('uint8')

    def normalize(self, image):
        return F.normalize(F.to_tensor(image), [.485, .456, .406],
                           [.229, .224, .225])

    def image_folder(self):
        for split in ('train', 'val'):
            for name in ('zebra', 'ant'):
                path = self.root / split / name / 'one.png'
                path.parent.mkdir(parents=True)
                Image.fromarray(self.rgb).save(path)

    def ade_folder(self):
        for split in ('training', 'validation'):
            image = self.root / 'images' / split / 'one.jpg'
            mask = self.root / 'annotations' / split / 'one.png'
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            Image.fromarray(self.rgb).save(image)
            Image.fromarray(self.labels).save(mask)

    def test_imagenet_validation_center_crop_and_single_normalization(self):
        from downstream import revised_data
        self.image_folder()
        data = revised_data.build_imagenet(self.root, 'val')
        image, label = data[0]
        expected = self.normalize(F.center_crop(F.resize(
            Image.fromarray(self.rgb), 256,
            interpolation=T.InterpolationMode.BICUBIC), 224))
        self.assertEqual(data.class_to_idx, {'ant': 0, 'zebra': 1})
        self.assertEqual(label, 0)
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_imagenet_train_crop_flip_matches_seeded_recipe(self):
        from downstream import revised_data
        self.image_folder()
        data = revised_data.build_imagenet(self.root, 'train')
        reference = T.Compose([
            T.RandomResizedCrop(224, scale=(.08, 1.), ratio=(.75, 4/3),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(.5), T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        for seed in (1, 7, 14):
            torch.manual_seed(seed)
            expected = reference(Image.fromarray(self.rgb))
            torch.manual_seed(seed)
            actual, _ = data[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ade_validation_square_geometry_and_all_label_boundaries(self):
        from downstream import revised_data
        self.ade_folder()
        data = revised_data.RevisedADE20K(self.root, 'val')
        actual, labels = data[0]
        with Image.open(self.root / 'images/validation/one.jpg') as img:
            expected = self.normalize(img.convert('RGB').resize(
                (224, 224), Image.Resampling.BICUBIC))
        with Image.open(self.root / 'annotations/validation/one.png') as mask:
            raw = np.array(mask.resize((224, 224), Image.Resampling.NEAREST))
        expected_labels = np.full(raw.shape, 255, dtype=np.int64)
        valid = (raw >= 1) & (raw <= 150)
        expected_labels[valid] = raw[valid].astype(np.int64)-1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(labels, torch.from_numpy(expected_labels)))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(set(labels.unique().tolist()), {0, 1, 149, 255})

    def test_ade_training_shared_scale_crop_flip_uses_python_rng(self):
        from downstream import revised_data
        self.ade_folder()
        data = revised_data.RevisedADE20K(self.root, 'train')
        for flip in (False, True):
            with patch('random.uniform', return_value=1.25), \
                 patch('random.randint', side_effect=[19, 31]), \
                 patch('random.random', return_value=.1 if flip else .9):
                actual, labels = data[0]
            with Image.open(self.root / 'images/training/one.jpg') as img:
                transformed = img.convert('RGB').resize((496, 351),
                    Image.Resampling.BICUBIC).crop((31, 19, 255, 243))
            with Image.open(self.root / 'annotations/training/one.png') as mask:
                mask = mask.resize((496, 351), Image.Resampling.NEAREST)
                raw = np.array(mask.crop((31, 19, 255, 243)))
            if flip:
                transformed = F.hflip(transformed)
                raw = np.fliplr(raw)
            mapped = np.full(raw.shape, 255, dtype=np.int64)
            valid = (raw >= 1) & (raw <= 150)
            mapped[valid] = raw[valid].astype(np.int64)-1
            torch.testing.assert_close(actual, self.normalize(transformed),
                                       rtol=0, atol=0)
            self.assertTrue(torch.equal(labels, torch.from_numpy(mapped)))

    def test_ade_small_scaled_image_is_at_least_crop_size(self):
        from downstream import revised_data
        self.ade_folder()
        data = revised_data.RevisedADE20K(self.root, 'train')
        with patch('random.uniform', return_value=.5), \
             patch('random.randint', return_value=0), \
             patch('random.random', return_value=.9):
            actual, labels = data[0]
        with Image.open(self.root / 'images/training/one.jpg') as img:
            expected = self.normalize(img.convert('RGB').resize(
                (224, 224), Image.Resampling.BICUBIC))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(labels.shape), (224, 224))

    def test_invalid_split_and_missing_input_fail_before_training(self):
        from downstream import revised_data
        for factory in (revised_data.build_imagenet, revised_data.RevisedADE20K):
            with self.assertRaisesRegex(ValueError, 'split'):
                factory(self.root, 'test')
            with self.assertRaises(FileNotFoundError):
                factory(self.root, 'train')
        self.ade_folder()
        (self.root / 'annotations/training/one.png').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'mask'):
            revised_data.RevisedADE20K(self.root, 'train')


if __name__ == '__main__':
    unittest.main()
