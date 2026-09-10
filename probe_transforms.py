"""The one place BASIC5_FAIR_v1 rules `aug` and `b` are implemented.

**rule aug** (docs/BASIC5_PROTOCOL.md): the linear probe's train-time
augmentation is **RandomResizedCrop + HorizontalFlip only**.

**rule b** (docs/BASIC5_PROTOCOL.md): the eval preprocessing is **Resize
(shorter side) 256 + CenterCrop 224** -- one deterministic 224 centre crop that
preserves aspect ratio. `basic5_eval_transform` is the single implementation of
that rule, the deterministic sibling of `basic5_train_transform`. Both live here
so the rules are not implemented once per method (there are ~50, and scanners
that each reimplemented a rule are the common root of past defects here).

The feature-cache family of methods extracts the frozen backbone's features once
and trains the linear head on the cache, so the backbone never re-runs per epoch.
Historically that meant the train split was read with the *deterministic* eval
transform -- resize + centre crop -- i.e. no train augmentation at all. This
module supplies the train transform those methods must use instead.

It lives in one place so the rule is not implemented once per method (there are
~50, and scanners that each reimplemented a rule are the common root of past
defects here). A method passes only the parts that *other* rules govern -- the
crop output size (rule b), the per-image normalisation tail (rule e), the
interpolation -- and this module owns which augmentations are applied.

Per the port's decision recorded in docs/BASIC5_PROTOCOL.md (option B, matching
21_barlow_twins), the augmented pass is still extracted once and cached: one
RandomResizedCrop + HorizontalFlip draw per image. This preserves the feature
cache; it does not re-run the backbone every epoch.

torchvision is imported lazily so the base environment (standard library only)
can import this module without torch installed.
"""

from __future__ import annotations


def basic5_train_transform(image_size, normalize=None, interpolation=None):
    """The BASIC5 rule-`aug` train transform.

    ``RandomResizedCrop(image_size)`` then ``RandomHorizontalFlip`` then
    ``ToTensor`` then ``normalize`` if one is given -- and nothing else. No
    colour jitter, blur, grayscale, rotation, or vertical flip: rule `aug` is
    RandomResizedCrop + HorizontalFlip *only*.

    Args:
        image_size: the crop output size (an int or ``(h, w)``). Which size is
            rule b's business; it is passed in, not chosen here.
        normalize: an already-built ``torchvision.transforms.Normalize`` (rule
            e's business), or ``None`` for a backbone trained on ``[0, 1]``.
        interpolation: a ``torchvision.transforms.InterpolationMode``, or
            ``None`` to use torchvision's default.

    Returns:
        a ``torchvision.transforms.Compose``.
    """
    from torchvision import transforms

    crop_kwargs = {}
    if interpolation is not None:
        crop_kwargs["interpolation"] = interpolation
    steps = [
        transforms.RandomResizedCrop(image_size, **crop_kwargs),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        steps.append(normalize)
    return transforms.Compose(steps)


def basic5_eval_transform(image_size, resize=None, normalize=None,
                          interpolation=None):
    """The BASIC5 rule-`b` eval transform.

    ``Resize(shorter side)`` then ``CenterCrop(image_size)`` then ``ToTensor``
    then ``normalize`` if one is given -- and nothing else. The resize is a
    *single int*, so torchvision scales the shorter side and **preserves aspect
    ratio**; it is deliberately not a ``(h, w)`` square, which would distort the
    image and skip the centre crop (the exact shape rule b reconciles).

    Args:
        image_size: the centre-crop output size (an int). This is the model's
            eval input size; which size is the caller's business (rule b keeps a
            native-resolution backbone at its own size), so it is passed in.
        resize: the shorter-side resize target. ``None`` uses the canonical
            ``round(image_size * 256 / 224)`` -- 256 at 224, scaled
            proportionally otherwise -- so the 256/224 ratio holds at any size.
        normalize: an already-built ``torchvision.transforms.Normalize`` (rule
            e's business), or ``None`` for a backbone trained on ``[0, 1]``.
        interpolation: a ``torchvision.transforms.InterpolationMode``, or
            ``None`` to use torchvision's default.

    Returns:
        a ``torchvision.transforms.Compose``.
    """
    from torchvision import transforms

    if resize is None:
        resize = round(image_size * 256 / 224)
    resize_kwargs = {}
    if interpolation is not None:
        resize_kwargs["interpolation"] = interpolation
    steps = [
        transforms.Resize(resize, **resize_kwargs),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        steps.append(normalize)
    return transforms.Compose(steps)
