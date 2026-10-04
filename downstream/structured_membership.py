"""Convert explicit native annotations; no inferred splits or benchmark certification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from downstream.structured_data import FAMILIES, asset_bytes
from downstream.structured_metrics import (
    check_pose,
    check_seven_pose,
    quaternion_matrix,
)

CAMBRIDGE_COUNTS = {
    "GreatCourt": (1532, 760),
    "KingsCollege": (1220, 343),
    "OldHospital": (895, 182),
    "ShopFacade": (231, 103),
    "StMarysChurch": (1487, 530),
}
SEVEN_COUNTS = {
    "chess": (4000, 2000),
    "fire": (2000, 2000),
    "heads": (1000, 1000),
    "office": (6000, 4000),
    "pumpkin": (4000, 2000),
    "redkitchen": (7000, 5000),
    "stairs": (2000, 1000),
}
SR_CSV_SHA256 = "bb6028c7c88e8d4c9bbdc9cdb4b989c53f32d230f60d40ad18c95622e12c3039"


def spatial_question(row):
    fields = ("qtype", "relation", "subject", "object1", "object2")
    values = {k: str(row[k]).strip() for k in fields}
    if not values["qtype"] or not values["relation"]:
        raise ValueError("spatial relation is missing")
    if (values["qtype"], values["relation"]) in {
        ("orientation", "viewpoint"),
        ("multi_object", "viewpoint towards object"),
    }:
        options = ["front", "left", "back", "right"]
    else:
        options = list(
            dict.fromkeys(
                v
                for k, v in values.items()
                if k in ("subject", "object1", "object2")
                and v.lower() not in ("", "nan")
            )
        )
        options += [v for v in ("yes", "no") if v not in options]
    return "Answer the spatial query described by these fields.\n" + "\n".join(
        k + ": " + v for k, v in values.items()
    ), options


def _relative(name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("unsafe native annotation path")
    return name


def _unique(values):
    if len(set(values)) != len(values):
        raise ValueError("duplicate native annotation identity")


def build(dataset, root, *, fixture_counts=None):
    if dataset not in FAMILIES:
        raise ValueError("unknown structured dataset")
    root = Path(root).resolve()
    hashes = {}
    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "split_evidence": "",
        "train": [],
        "validation": [],
    }

    def read(path, member=None):
        asset = {"path": _relative(path)}
        if member is not None:
            asset["member"] = _relative(member)
        raw = asset_bytes(root, asset)
        hashes[json.dumps(asset, sort_keys=True)] = hashlib.sha256(raw).hexdigest()
        return raw

    def names(path):
        p = (root / _relative(path)).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError("missing or escaping archive")
        with zipfile.ZipFile(p) as z:
            values = z.namelist()
            _unique(values)
            return values

    if dataset in ("cambridge_landmarks", "seven_scenes"):
        counts = (
            fixture_counts
            if fixture_counts is not None
            else CAMBRIDGE_COUNTS
            if dataset == "cambridge_landmarks"
            else SEVEN_COUNTS
        )
        result["scenes"] = sorted(counts)
        for scene in result["scenes"]:
            _relative(scene)
            sequences = {}
            if dataset == "seven_scenes":
                for split, file in [
                    ("train", "TrainSplit.txt"),
                    ("validation", "TestSplit.txt"),
                ]:
                    rows = read(scene + "/" + file).decode().splitlines()
                    seq = []
                    for line in rows:
                        if not line.strip():
                            continue
                        m = re.fullmatch(r"sequence(\d+)", line.strip())
                        if m is None:
                            raise ValueError("malformed official sequence split")
                        seq.append(f"seq-{int(m[1]):02d}")
                    _unique(seq)
                    if not seq:
                        raise ValueError("empty sequence split")
                    sequences[split] = seq
                if set(sequences["train"]) & set(sequences["validation"]):
                    raise ValueError("sequence split leakage")
            for index, split in enumerate(("train", "validation")):
                rows = []
                if dataset == "cambridge_landmarks":
                    archive = scene + ".zip"
                    sp = "train" if split == "train" else "test"
                    members = set(names(archive))
                    for line in (
                        read(archive, f"{scene}/dataset_{sp}.txt").decode().splitlines()
                    ):
                        parts = line.split()
                        if not parts:
                            continue
                        if not parts[0].lower().endswith((".png", ".jpg")):
                            if not line.startswith(
                                ("Visual Landmark Dataset", "ImageFile", "#")
                            ):
                                raise ValueError("malformed Cambridge header")
                            continue
                        if len(parts) != 8:
                            raise ValueError("malformed Cambridge pose")
                        member = scene + "/" + _relative(parts[0])
                        v = np.asarray(list(map(float, parts[1:])))
                        if member not in members:
                            raise ValueError("Cambridge image/pose join failed")
                        pose = np.eye(4)
                        pose[:3, 3] = v[:3]
                        pose[:3, :3] = quaternion_matrix(v[3:])
                        rows.append(
                            {
                                "id": member,
                                "image": {"path": archive, "member": member},
                                "scene": scene,
                                "pose": check_pose(pose).tolist(),
                            }
                        )
                else:
                    for seq in sequences[split]:
                        archive = f"{scene}/{seq}.zip"
                        members = names(archive)
                        colors = sorted(n for n in members if n.endswith(".color.png"))
                        if not colors or {
                            n.removesuffix(".color.png") for n in colors
                        } != {
                            n.removesuffix(".pose.txt")
                            for n in members
                            if n.endswith(".pose.txt")
                        }:
                            raise ValueError("7-Scenes image/pose join failed")
                        for member in colors:
                            if not member.startswith(seq + "/"):
                                raise ValueError("wrong sequence prefix")
                            pose = check_seven_pose(
                                np.loadtxt(
                                    io.BytesIO(
                                        read(
                                            archive,
                                            member.removesuffix(".color.png")
                                            + ".pose.txt",
                                        )
                                    )
                                )
                            )
                            rows.append(
                                {
                                    "id": scene + "/" + member,
                                    "image": {"path": archive, "member": member},
                                    "scene": scene,
                                    "pose": pose.tolist(),
                                }
                            )
                if len(rows) != counts[scene][index]:
                    raise ValueError("native scene split count mismatch")
                result[split].extend(rows)
    elif dataset == "three_d_srbench":
        raw = read("3dsrbench_v1.csv")
        if fixture_counts is None and hashlib.sha256(raw).hexdigest() != SR_CSV_SHA256:
            raise ValueError("3DSR CSV pin mismatch")
        pin = json.loads(read("sr_holdout.json"))
        for k in ("train_ids", "eval_ids"):
            _unique(pin[k])
        groups = {
            s: set(pin[k])
            for s, k in [("train", "train_ids"), ("validation", "eval_ids")]
        }
        if not all(groups.values()) or groups["train"] & groups["validation"]:
            raise ValueError("3DSR holdout leakage/empty")
        seen = set()
        for row in csv.DictReader(io.StringIO(raw.decode())):
            member = _relative(urlparse(row["image"]).path.lstrip("/"))
            if member not in groups["train"] | groups["validation"]:
                raise ValueError("unknown holdout image")
            seen.add(member)
            split = "train" if member in groups["train"] else "validation"
            question, options = spatial_question(row)
            if row["answer"] not in options:
                raise ValueError("answer not in input options")
            result[split].append(
                {
                    "id": row["qid"],
                    "image": {"path": "images/" + member},
                    "question": question,
                    "options": options,
                    "target": options.index(row["answer"]),
                }
            )
        if seen != groups["train"] | groups["validation"]:
            raise ValueError("incomplete holdout membership")
        expected = (pin["train_questions"], pin["eval_questions"])
        if tuple(map(len, (result["train"], result["validation"]))) != expected:
            raise ValueError("holdout question counts disagree")
    elif dataset == "crowdpose":
        archive = "archives/CrowdPose_images.zip"
        members = {}
        for name in names(archive):
            if name.lower().endswith((".jpg", ".png")):
                base = Path(name).name
                if base in members:
                    raise ValueError("duplicate image basename")
                members[base] = name
        annotation = read("archives/CrowdPose_annotations.tar.gz")
        with tarfile.open(fileobj=io.BytesIO(annotation), mode="r:gz") as t:
            for split, suffix in [
                ("train", "crowdpose_train.json"),
                ("validation", "crowdpose_val.json"),
            ]:
                candidates = [
                    m for m in t.getmembers() if m.name.endswith(suffix) and m.isfile()
                ]
                if len(candidates) != 1:
                    raise ValueError("ambiguous CrowdPose annotation member")
                stream = t.extractfile(candidates[0])
                if stream is None:
                    raise ValueError("missing annotation payload")
                with stream:
                    blob = json.load(stream)
                _unique([a["id"] for a in blob["images"]])
                _unique([a["id"] for a in blob["annotations"]])
                crowd_rows: dict[int, dict[str, Any]] = {}
                for im in blob["images"]:
                    basename = Path(im["file_name"]).name
                    if basename not in members:
                        raise ValueError("CrowdPose image join failed")
                    crowd_rows[im["id"]] = {
                        "id": str(im["id"]),
                        "image": {"path": archive, "member": members[basename]},
                        "crowd_index": im["crowdIndex"],
                        "people": [],
                    }
                for ann in blob["annotations"]:
                    if ann["image_id"] not in crowd_rows:
                        raise ValueError("unknown annotated image")
                    joints = np.asarray(ann["keypoints"], dtype=float).reshape(14, 3)
                    x, y, w, h = map(float, ann["bbox"])
                    side = max(w, h) * 1.25
                    if min(w, h) <= 0 or not np.isfinite([x, y, w, h]).all():
                        raise ValueError("invalid person bbox")
                    crowd_rows[ann["image_id"]]["people"].append(
                        {
                            "id": str(ann["id"]),
                            "joints": joints[:, :2].tolist(),
                            "labeled": (joints[:, 2] > 0).tolist(),
                            "crop": [
                                x + w / 2 - side / 2,
                                y + h / 2 - side / 2,
                                side,
                                side,
                            ],
                            "keypoints": joints.tolist(),
                            "bbox": ann["bbox"],
                            "num_keypoints": ann["num_keypoints"],
                            "iscrowd": ann.get("iscrowd", 0),
                        }
                    )
                result[split] = [crowd_rows[k] for k in sorted(crowd_rows)]
    else:
        pins = json.loads(read("mpii_pins.json"))
        raw = read("archives/mpii_release_v1.json")
        sha = hashlib.sha256(raw).hexdigest()
        if sha != pins["release_sha256"]:
            raise ValueError("MPII release pin mismatch")
        release = json.loads(raw)
        people = release["persons"]
        expected = 2958 if fixture_counts is None else fixture_counts[1]
        if (
            release["schema"] != "mpii-release-persons-v1"
            or release["validation_persons"] != expected
            or any(type(p["validation"]) is not bool for p in people)
            or release["training_rule"]
            != "non-validation RELEASE training persons on images without a validation person"
        ):
            raise ValueError("MPII release/split contract mismatch")
        _unique([p["id"] for p in people])
        excluded = release["excluded_nonvalidation_persons_on_validation_images"]
        _unique(excluded)
        if {p["id"] for p in people} & set(excluded):
            raise ValueError("excluded MPII person present")
        vpeople = [p for p in people if p["validation"]]
        if len(vpeople) != expected:
            raise ValueError("MPII validation population mismatch")
        vimages = {p["image"] for p in vpeople}
        if any(p["image"] in vimages for p in people if not p["validation"]):
            raise ValueError("MPII image split overlap")
        side = read("archives/" + _relative(pins["gt_val_sidecar_filename"]))
        if hashlib.sha256(side).hexdigest() != pins["gt_val_sidecar_sha256"]:
            raise ValueError("MPII sidecar pin mismatch")
        side = json.loads(side)
        if (
            side["schema"] != "mpii-hrnet-gt-val-v1"
            or side["source_sha256"]["release_json"] != sha
        ):
            raise ValueError("MPII sidecar release mismatch")
        _unique([p["person_id"] for p in side["columns"]])
        gt = {p["person_id"]: p for p in side["columns"]}
        if set(gt) != {p["id"] for p in vpeople}:
            raise ValueError("MPII sidecar person join failed")
        for split, flag in [("train", False), ("validation", True)]:
            rows = {}
            for p in people:
                if p["validation"] != flag:
                    continue
                image = _relative(p["image"])
                row = rows.setdefault(
                    image,
                    {"id": image, "image": {"path": "images/" + image}, "people": []},
                )
                person = {k: p[k] for k in ("id", "joints", "labeled", "crop")}
                if flag:
                    person.update(
                        {
                            k: gt[p["id"]][k]
                            for k in ("pos_gt_src", "jnt_missing", "headboxes_src")
                        }
                    )
                row["people"].append(person)
            result[split] = list(rows.values())
    for split in ("train", "validation"):
        if not result[split]:
            raise ValueError("empty native split")
        _unique([r["id"] for r in result[split]])
    if {r["id"] for r in result["train"]} & {r["id"] for r in result["validation"]}:
        raise ValueError("native split identity overlap")
    result["split_evidence"] = json.dumps(
        {
            "annotation_sha256": hashes,
            "fixture_counts": fixture_counts,
            "disclosure": "COCO-2100 local fitted holdout; not CircularEval"
            if dataset == "three_d_srbench"
            else "explicit native annotations; authenticity and historical score identity unverified",
        },
        sort_keys=True,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = build(args.dataset, args.data_root)
    with open(args.out, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
