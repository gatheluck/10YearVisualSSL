"""Explicit person/scene/question membership with independent image partitions."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate

from downstream import contract
from downstream.structured_metrics import (
    check_pose,
    check_seven_pose,
    pckh_counts,
    project_so3,
)

FAMILIES = {
    "mpii_pose": "pose",
    "crowdpose": "pose",
    "seven_scenes": "localization",
    "cambridge_landmarks": "localization",
    "three_d_srbench": "reasoning",
}


def asset_bytes(root, asset):
    if not isinstance(asset, dict) or set(asset) not in ({"path"}, {"path", "member"}):
        raise ValueError("image asset requires path and optional ZIP member")
    root = Path(root).resolve()
    path = Path(asset["path"])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("relative asset path required")
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaping image asset")
    if "member" not in asset:
        return path.read_bytes()
    member = asset["member"]
    if (
        not isinstance(member, str)
        or not member
        or Path(member).is_absolute()
        or ".." in Path(member).parts
    ):
        raise ValueError("relative archive member required")
    with zipfile.ZipFile(path) as z:
        if z.namelist().count(member) != 1:
            raise ValueError("missing or duplicate ZIP member")
        return z.read(member)


def byte_tokens(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty full question/option required")
    return torch.tensor([b + 1 for b in text.encode("utf-8")], dtype=torch.long)


def collate(rows):
    if len(rows[0]) == 2:
        return default_collate(rows)
    images, inputs, targets = zip(*rows)
    width = max(len(t) for q in inputs for t in q)
    count = max(map(len, inputs))
    q = torch.zeros(len(rows), count, width, dtype=torch.long)
    for i, questions in enumerate(inputs):
        for j, tokens in enumerate(questions):
            q[i, j, : len(tokens)] = tokens
    return torch.stack(images), q, default_collate(list(targets))


def _array(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError("invalid finite annotation shape")
    return a


def _person(person, dataset, validation):
    n = 16 if dataset == "mpii_pose" else 14
    _array(person["joints"], (n, 2))
    labeled = np.asarray(person["labeled"])
    if labeled.shape != (n,) or labeled.dtype != np.bool_:
        raise ValueError("joint labels must be boolean")
    crop = _array(person["crop"], (4,))
    if (crop[2:] <= 0).any():
        raise ValueError("positive person crop required")
    if dataset == "mpii_pose" and validation:
        if not {"pos_gt_src", "jnt_missing", "headboxes_src"} <= set(person):
            raise ValueError("MPII evaluation requires joined HRNet sidecar targets")
        pckh_counts(
            np.asarray(person["joints"])[None],
            np.asarray(person["pos_gt_src"])[None],
            np.asarray(person["jnt_missing"])[None],
            np.asarray(person["headboxes_src"])[None],
        )
    elif dataset == "crowdpose":
        keypoints = _array(person["keypoints"], (14, 3))
        box = _array(person["bbox"], (4,))
        if not np.isin(keypoints[:, 2], [0, 1, 2]).all() or (box[2:] <= 0).any():
            raise ValueError("invalid CrowdPose visibility or bbox")
        if not np.array_equal(keypoints[:, :2], person["joints"]) or not np.array_equal(
            keypoints[:, 2] > 0, labeled
        ):
            raise ValueError("CrowdPose joint/visibility representations disagree")
        if (
            type(person["num_keypoints"]) is not int
            or not 0 <= person["num_keypoints"] <= 14
            or person["iscrowd"] not in (0, 1)
        ):
            raise ValueError("invalid CrowdPose keypoint count or crowd flag")


class Samples(Dataset):
    def __init__(self, root, rows, dataset, scenes, train):
        self.root, self.rows, self.dataset, self.scenes, self.train = (
            root,
            rows,
            dataset,
            scenes,
            train,
        )
        self.family = FAMILIES[dataset]
        self.items = (
            [
                (r, p)
                for r in rows
                for p in r["people"]
                if any(p["labeled"]) and not p.get("iscrowd", 0)
            ]
            if self.family == "pose"
            else rows
        )
        if not self.items:
            raise ValueError("empty usable task population")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        row, person = item if self.family == "pose" else (item, None)
        with Image.open(io.BytesIO(asset_bytes(self.root, row["image"]))) as im:
            im = im.convert("RGB")
            if person is not None:
                crop = person["crop"]
                im = im.transform(
                    (224, 224),
                    Image.Transform.AFFINE,
                    (crop[2] / 224, 0, crop[0], 0, crop[3] / 224, crop[1]),
                    resample=Image.Resampling.BILINEAR,
                )
            else:
                im = im.resize((224, 224), Image.Resampling.BICUBIC)
            image = (
                torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255
            )
        if self.family == "reasoning":
            questions = [byte_tokens(s) for s in [row["question"], *row["options"]]]
            return image, questions, row["target"] if self.train else index
        if not self.train:
            return image, index
        if self.family == "localization":
            pose = np.asarray(row["pose"], dtype=np.float32)
            rotation = (
                project_so3(np.asarray(row["pose"])[:3, :3])
                if self.dataset == "seven_scenes"
                else pose[:3, :3]
            )
            target = np.r_[
                self.scenes.index(row["scene"]), pose[:3, 3], rotation.reshape(-1)
            ].astype(np.float32)
            return image, torch.from_numpy(target)
        person = self.items[index][1]
        crop = np.asarray(person["crop"])
        xy = (np.asarray(person["joints"]) - crop[:2]) / crop[2:] * 56
        y, x = np.mgrid[:56, :56]
        heat = np.exp(
            -(
                (x[None] - xy[:, 0, None, None]) ** 2
                + (y[None] - xy[:, 1, None, None]) ** 2
            )
            / 8
        )
        keep = (
            np.asarray(person["labeled"])
            & (xy[:, 0] >= -6)
            & (xy[:, 0] < 62)
            & (xy[:, 1] >= -6)
            & (xy[:, 1] < 62)
        )
        heat[~keep] = np.nan
        return image, torch.from_numpy(heat.astype(np.float32))


def load_data(manifest, root, dataset):
    if dataset not in FAMILIES:
        raise ValueError("unsupported structured dataset")
    raw = Path(manifest).read_bytes()
    data = json.loads(raw)
    family = FAMILIES[dataset]
    required = {"schema_version", "dataset", "split_evidence", "train", "validation"}
    if family == "localization":
        required.add("scenes")
    if (
        set(data) != required
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["dataset"] != dataset
    ):
        raise ValueError("invalid structured manifest identity/schema")
    if (
        not isinstance(data["split_evidence"], str)
        or not data["split_evidence"].strip()
    ):
        raise ValueError("explicit split evidence required")
    scenes = data.get("scenes", [])
    if family == "localization" and (
        not scenes
        or len(set(scenes)) != len(scenes)
        or any(not isinstance(s, str) or not s for s in scenes)
    ):
        raise ValueError("explicit ordered scene vocabulary required")
    ids, people, assets, cache, digest = set(), set(), {}, {}, hashlib.sha256()
    for split in ("train", "validation"):
        rows = data[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError("both splits must be nonempty")
        seen_scenes = set()
        for row in rows:
            ident = row.get("id")
            if not isinstance(ident, str) or not ident or ident in ids:
                raise ValueError("duplicate or invalid sample ID")
            ids.add(ident)
            key = json.dumps(row["image"], sort_keys=True)
            if key not in cache:
                cache[key] = hashlib.sha256(asset_bytes(root, row["image"])).hexdigest()
            sha = cache[key]
            if sha in assets and assets[sha] != split:
                raise ValueError("train/evaluation image leakage")
            assets[sha] = split
            digest.update(json.dumps([ident, sha], sort_keys=True).encode())
            if family == "localization":
                if row["scene"] not in scenes:
                    raise ValueError("unknown benchmark scene")
                seen_scenes.add(row["scene"])
                (check_seven_pose if dataset == "seven_scenes" else check_pose)(
                    row["pose"]
                )
            elif family == "reasoning":
                byte_tokens(row["question"])
                options = row["options"]
                if (
                    not isinstance(options, list)
                    or len(options) < 2
                    or len(set(options)) != len(options)
                ):
                    raise ValueError("distinct explicit answer options required")
                for option in options:
                    byte_tokens(option)
                if type(row["target"]) is not int or not 0 <= row["target"] < len(
                    options
                ):
                    raise ValueError("answer index outside input options")
            else:
                if dataset == "crowdpose" and not np.isfinite(
                    float(row["crowd_index"])
                ):
                    raise ValueError("nonfinite crowd index")
                for person in row["people"]:
                    pid = person.get("id")
                    if not isinstance(pid, str) or not pid or pid in people:
                        raise ValueError("duplicate or missing person ID")
                    people.add(pid)
                    _person(person, dataset, split == "validation")
            if family == "localization" and not seen_scenes <= set(scenes):
                raise ValueError("unknown scene")
        if family == "localization" and seen_scenes != set(scenes):
            raise ValueError("split lacks declared scenes")
    train = Samples(root, data["train"], dataset, scenes, True)
    validation = Samples(root, data["validation"], dataset, scenes, False)
    membership = {
        "manifest_sha256": contract.sha256_bytes(raw),
        "assets_sha256": digest.hexdigest(),
        "split_evidence": data["split_evidence"],
        "train_count": len(train),
        "validation_count": len(validation),
        "scenes": scenes,
        "outputs": (16 if dataset == "mpii_pose" else 14)
        if family == "pose"
        else len(scenes)
        if scenes
        else 1,
        "official_membership_verified": False,
    }
    return train, validation, membership
