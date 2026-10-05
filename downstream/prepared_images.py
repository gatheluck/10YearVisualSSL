"""Explicit prepared train/val memberships; never generate a random partition."""

import re
from collections import Counter

COUNTS = {
    "action40": (4000, 5532, 40),
    "cifar10": (50000, 10000, 10),
    "cifar100": (50000, 10000, 100),
    "kmnist": (60000, 10000, 10),
    "imagenet_1percent": (12811, 50000, 1000),
    "imagenet_10percent": (128116, 50000, 1000),
    "omniglot15": (24345, 8115, 1623),
}
QUOTAS = {
    "action40": (100, None),
    "cifar10": (5000, 1000),
    "cifar100": (500, 100),
    "kmnist": (6000, 1000),
    "imagenet_1percent": (None, 50),
    "imagenet_10percent": (None, 50),
    "omniglot15": (15, 5),
}
SUBSETS = {"imagenet_1percent", "imagenet_10percent"}
LOCAL_IDS = {"cifar10", "cifar100", "kmnist"}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}


def membership(dataset, root, counts, *, fixture, eval_root=None):
    # Imported here to share the same path validation and staging implementation.
    from downstream.native_images import file_at

    inventories = {}
    for split, base in (("train", root), ("val", eval_root or root)):
        directory = base / split
        if directory.is_symlink():
            raise ValueError("source symlinks are not accepted")
        if not directory.is_dir():
            raise ValueError("missing split directory")
        found = {}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("source symlinks are not accepted")
            if path.is_dir() or path.name == ".DS_Store":
                continue
            name = path.relative_to(base).as_posix()
            if len(name.split("/")) != 3 or path.suffix.lower() not in SUFFIXES:
                raise ValueError("expected split/class/image layout")
            file_at(base, name)
            found[name] = path.parent.name
        directories = {p.name for p in directory.iterdir() if p.is_dir()}
        if not found or directories != set(found.values()):
            raise ValueError("missing or empty class directory")
        inventories[split] = found
    classes = sorted(set(inventories["train"].values()))
    if len(classes) != counts[2]:
        raise ValueError("prepared class count differs")
    pattern = (
        r"n[0-9]{8}"
        if dataset in SUBSETS
        else r"[0-9]"
        if dataset == "kmnist"
        else None
    )
    if pattern and not all(re.fullmatch(pattern, c) for c in classes):
        raise ValueError("invalid prepared class name")
    labels = {c: i for i, c in enumerate(classes)}
    rows, identities = [], {}
    for index, split in enumerate(("train", "val")):
        found = inventories[split]
        if set(found.values()) != set(classes):
            raise ValueError("prepared split class coverage differs")
        quota = QUOTAS[dataset][index]
        if (
            not fixture
            and quota is not None
            and set(Counter(found.values()).values()) != {quota}
        ):
            raise ValueError("prepared per-class count differs")
        identities[split] = {
            name if dataset in LOCAL_IDS else name.split("/", 1)[1] for name in found
        }
        for name, cls in found.items():
            rows.append(
                {
                    "split": "train" if split == "train" else "test",
                    "path": name,
                    "source": name,
                    "target": labels[cls],
                    "source_root": root if split == "train" else eval_root or root,
                    "source_role": "training_root"
                    if split == "train"
                    else "evaluation_root",
                }
            )
    if identities["train"] & identities["val"]:
        raise ValueError("cross-split source identity overlap")
    return (
        classes,
        rows,
        {
            "upstream_membership_verified": False,
            "membership_policy": "prepared splits; no sampling or inferred membership",
            "item_identity": "split/class/filename"
            if dataset in LOCAL_IDS
            else "class/filename",
            "evaluation": "provided val directory",
            "separate_evaluation_root": eval_root is not None,
            "publisher_partition": "not the disjoint-alphabet split"
            if dataset == "omniglot15"
            else "unverified",
            "transform_reference": "captured_rgb_rrc_v1",
            "small_image_recipe_conflict": dataset in LOCAL_IDS
            or dataset == "omniglot15",
        },
        {},
    )
