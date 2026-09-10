#!/usr/bin/env python3
"""Tests for `probe_transforms`, the one implementation of BASIC5 rule `aug`.

BASIC5_FAIR_v1 rule `aug` (docs/BASIC5_PROTOCOL.md): the linear probe's
train-time augmentation is **RandomResizedCrop + HorizontalFlip only**. The
feature-cache family of methods historically read the train split with the
*deterministic* eval transform (no augmentation at all); `probe_transforms`
supplies the train transform they must use instead, in exactly one place so the
rule is not implemented once per method.

These tests pin what that transform is -- and, as a negative control, what it is
*not* (no colour jitter, blur, or grayscale): rule `aug` is those two
augmentations and nothing else, so a helper that quietly added a third would be
caught here.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import probe_transforms                                   # noqa: E402

try:
    import torch                                          # noqa: F401
    import torchvision                                    # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "probe_transforms needs torchvision")


def _types(compose):
    return [type(t).__name__ for t in compose.transforms]


class TestBasic5TrainTransform(unittest.TestCase):
    @needs_deps
    def test_it_is_random_resized_crop_then_horizontal_flip_then_totensor(self):
        from torchvision import transforms
        t = probe_transforms.basic5_train_transform(224)
        self.assertIsInstance(t, transforms.Compose)
        self.assertEqual(
            _types(t),
            ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor"],
            "rule aug is RandomResizedCrop + HorizontalFlip only, then ToTensor")

    @needs_deps
    def test_the_crop_output_size_is_the_requested_size(self):
        t = probe_transforms.basic5_train_transform(160)
        crop = t.transforms[0]
        self.assertEqual(tuple(crop.size), (160, 160),
                         "the crop must output the requested size")

    @needs_deps
    def test_a_normalize_tail_is_appended_when_given(self):
        from torchvision import transforms
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        t = probe_transforms.basic5_train_transform(224, normalize=norm)
        self.assertEqual(
            _types(t),
            ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor",
             "Normalize"],
            "a per-method Normalize (rule e) is appended after ToTensor")
        self.assertIs(t.transforms[-1], norm,
                      "the caller's Normalize is used verbatim")

    @needs_deps
    def test_no_normalize_when_none(self):
        t = probe_transforms.basic5_train_transform(224, normalize=None)
        self.assertNotIn("Normalize", _types(t),
                         "a [0,1]-trained backbone gets no normalisation")

    @needs_deps
    def test_the_interpolation_reaches_the_crop_when_given(self):
        from torchvision.transforms import InterpolationMode
        t = probe_transforms.basic5_train_transform(
            224, interpolation=InterpolationMode.BICUBIC)
        self.assertEqual(t.transforms[0].interpolation,
                         InterpolationMode.BICUBIC)

    @needs_deps
    def test_no_other_augmentation_is_present(self):
        # Negative control: rule aug is RandomResizedCrop + HorizontalFlip ONLY.
        # A helper that added colour jitter, blur, grayscale, rotation, or a
        # vertical flip would still "have RRC + HFlip", so membership alone would
        # not catch it -- the exhaustive type list above would, and this names
        # the specific decoys a looser check would miss.
        from torchvision import transforms
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        present = set(_types(
            probe_transforms.basic5_train_transform(224, normalize=norm)))
        for forbidden in ("ColorJitter", "RandomGrayscale", "GaussianBlur",
                          "RandomVerticalFlip", "RandomRotation",
                          "RandomErasing"):
            self.assertNotIn(forbidden, present,
                             f"{forbidden} is not part of BASIC5 rule aug")


if __name__ == "__main__":
    unittest.main()
