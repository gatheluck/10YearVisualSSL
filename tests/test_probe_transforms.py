#!/usr/bin/env python3
"""Tests for `probe_transforms`, the one place BASIC5 rules `aug` and `b` live.

BASIC5_FAIR_v1 rule `aug` (docs/BASIC5_PROTOCOL.md): the linear probe's
train-time augmentation is **RandomResizedCrop + HorizontalFlip only**. The
feature-cache family of methods historically read the train split with the
*deterministic* eval transform (no augmentation at all); `probe_transforms`
supplies the train transform they must use instead, in exactly one place so the
rule is not implemented once per method.

BASIC5_FAIR_v1 rule `b`: the *eval* preprocessing is **Resize (shorter side)
256 + CenterCrop 224** -- one deterministic 224 centre crop that preserves
aspect ratio, not a square resize that distorts it and skips the crop.
`probe_transforms.basic5_eval_transform` is the single implementation of that
rule, the deterministic sibling of the train transform.

These tests pin what those transforms are -- and, as negative controls, what
they are *not* (for `aug`: no colour jitter, blur, or grayscale; for `b`: no
square resize, and the centre crop is never dropped), so a helper that quietly
changed one would be caught here.
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


class TestBasic5EvalTransform(unittest.TestCase):
    @needs_deps
    def test_it_is_resize_then_centercrop_then_totensor(self):
        from torchvision import transforms
        t = probe_transforms.basic5_eval_transform(224)
        self.assertIsInstance(t, transforms.Compose)
        self.assertEqual(
            _types(t),
            ["Resize", "CenterCrop", "ToTensor"],
            "rule b is Resize (shorter side) + CenterCrop, then ToTensor")

    @needs_deps
    def test_the_resize_is_shorter_side_not_a_square(self):
        # The heart of rule b: an int Resize scales the shorter side and keeps
        # aspect ratio; a (h, w) tuple squashes the image. The wrong-crop methods
        # this rule reconciles used a square tuple, so this is the check that
        # actually separates conformant from not.
        t = probe_transforms.basic5_eval_transform(224)
        resize = t.transforms[0]
        self.assertIsInstance(resize.size, int,
                              "Resize must take a single int (shorter side), "
                              "not a (h, w) square that distorts aspect ratio")

    @needs_deps
    def test_the_default_resize_is_256_for_a_224_crop(self):
        # Rule b's canonical pairing: shorter side 256, centre crop 224. The
        # default generalises it as round(image_size * 256 / 224), which is 256
        # at 224 and scales proportionally for a native-resolution backbone.
        t = probe_transforms.basic5_eval_transform(224)
        self.assertEqual(t.transforms[0].size, 256)
        self.assertEqual(tuple(t.transforms[1].size), (224, 224))

    @needs_deps
    def test_the_resize_scales_with_the_crop_size(self):
        t = probe_transforms.basic5_eval_transform(112)
        self.assertEqual(t.transforms[0].size, round(112 * 256 / 224))
        self.assertEqual(tuple(t.transforms[1].size), (112, 112))

    @needs_deps
    def test_an_explicit_resize_overrides_the_default(self):
        t = probe_transforms.basic5_eval_transform(224, resize=248)
        self.assertEqual(t.transforms[0].size, 248,
                         "an explicit shorter-side resize is honoured verbatim")

    @needs_deps
    def test_a_normalize_tail_is_appended_when_given(self):
        from torchvision import transforms
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        t = probe_transforms.basic5_eval_transform(224, normalize=norm)
        self.assertEqual(
            _types(t),
            ["Resize", "CenterCrop", "ToTensor", "Normalize"],
            "a per-method Normalize (rule e) is appended after ToTensor")
        self.assertIs(t.transforms[-1], norm,
                      "the caller's Normalize is used verbatim")

    @needs_deps
    def test_no_normalize_when_none(self):
        t = probe_transforms.basic5_eval_transform(224, normalize=None)
        self.assertNotIn("Normalize", _types(t),
                         "a [0,1]-trained backbone gets no normalisation")

    @needs_deps
    def test_the_interpolation_reaches_the_resize_when_given(self):
        from torchvision.transforms import InterpolationMode
        t = probe_transforms.basic5_eval_transform(
            224, interpolation=InterpolationMode.BICUBIC)
        self.assertEqual(t.transforms[0].interpolation,
                         InterpolationMode.BICUBIC)

    @needs_deps
    def test_no_random_augmentation_is_present(self):
        # Negative control: the eval transform is deterministic. None of rule
        # aug's random ops may leak into it -- a helper that reused the train
        # ops here would still "have Resize + CenterCrop", so membership alone
        # would not catch it; naming the decoys does.
        from torchvision import transforms
        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        present = set(_types(
            probe_transforms.basic5_eval_transform(224, normalize=norm)))
        for forbidden in ("RandomResizedCrop", "RandomHorizontalFlip",
                          "RandomVerticalFlip", "ColorJitter"):
            self.assertNotIn(forbidden, present,
                             f"{forbidden} is not part of the eval transform")


if __name__ == "__main__":
    unittest.main()
