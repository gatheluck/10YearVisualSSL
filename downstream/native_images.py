"""Stage inspected native image memberships without inferring splits.

Original images stay read-only. Publication uses a new directory, preserves image
bytes and records membership limitations instead of dropping inconvenient rows.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from downstream import prepared_images
from downstream.semantic_staging import write

COUNTS = {
    "clue_egg": (19376, 6300, 283),
    "clue_feather": (53243, 15268, 555),
    "clue_footprint": (12575, 3834, 117),
    "clue_skulls": (10337, 3787, 269),
    "clue_stools": (13399, 3639, 101),
    "sun397": (19850, 19850, 397),
}


def relative(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))
    ):
        raise ValueError("unsafe relative source path")
    return name


def file_at(root, name):
    relative(name)
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("source symlinks are not accepted")
    if not path.is_file():
        raise ValueError("missing source file")
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def checked_image(raw):
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.convert("RGB").load()
    except Exception as exc:
        raise ValueError("unreadable source image") from exc


def submission_groups(selected):
    groups = {}
    unknown = {}
    for sp, names in selected.items():
        groups[sp] = set()
        unknown[sp] = 0
        for name in names:
            match = re.fullmatch(
                r"(\d+)_\d+_bbox\d+\.(?:jpg|jpeg|png)",
                PurePosixPath(name).name,
                re.IGNORECASE,
            )
            if match:
                groups[sp].add(match[1])
            else:
                unknown[sp] += 1
    splits = sorted(groups)
    overlaps = {
        a + "__" + b: len(groups[a] & groups[b])
        for i, a in enumerate(splits)
        for b in splits[i + 1 :]
    }
    return {
        "status": "cross_split_submission_overlap"
        if any(overlaps.values())
        else "unknown_submission_ids"
        if any(unknown.values())
        else "no_observed_overlap",
        "overlaps": overlaps,
        "unknown": unknown,
        "all_ids_known": not any(unknown.values()),
        "policy": "retain source membership; no images dropped",
    }


def clue(root, counts):
    selected = {sp: [] for sp in ("train", "test", "valid")}
    for sp, selected_names in selected.items():
        directory = root / sp
        if directory.is_symlink():
            raise ValueError("source symlinks are not accepted")
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("source symlinks are not accepted")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                name = path.relative_to(root).as_posix()
                if len(name.split("/")) != 3:
                    raise ValueError("expected split/class/image layout")
                selected_names.append(name)
    classes = sorted({p.split("/")[1] for p in selected["train"]})
    if len(classes) != counts[2]:
        raise ValueError("CLUE training class count differs")
    rows = []
    for sp in ("train", "test"):
        for name in selected[sp]:
            cls = name.split("/")[1]
            if cls not in classes:
                raise ValueError("unknown CLUE evaluation class")
            rows.append(
                {
                    "split": sp,
                    "path": name,
                    "target": classes.index(cls),
                    "source": name,
                }
            )
    return (
        classes,
        rows,
        {
            "submission_groups": submission_groups(selected),
            "evaluation": "source test; valid diagnostic only",
        },
        {},
    )


def sun(root, counts, stack):
    protocol = root / "protocol_tar"
    raw = file_at(root, "protocol_tar/manifest.jsonl").read_bytes()
    rows = []
    names = {}
    archives = {}
    seen = set()
    for line in raw.decode("utf-8").splitlines():
        try:
            r = json.loads(line, object_pairs_hook=unique_object)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid SUN JSONL") from exc
        required = {
            "archive",
            "member",
            "class_name",
            "label",
            "split",
            "bytes",
            "content_sha256",
        }
        if not isinstance(r, dict) or not required <= r.keys():
            raise ValueError("invalid SUN row schema")
        label, cls, sp = r["label"], r["class_name"], r["split"]
        if type(label) is not int or not 0 <= label < counts[2]:
            raise ValueError("invalid SUN label")
        if not isinstance(cls, str) or not cls.strip() or cls != cls.strip():
            raise ValueError("invalid SUN class")
        if names.get(label, cls) != cls:
            raise ValueError("SUN label mapping conflict")
        names[label] = cls
        if sp not in ("train", "test"):
            raise ValueError("invalid SUN split")
        archive, member = relative(r["archive"]), relative(r["member"])
        sha = r["content_sha256"]
        if not isinstance(sha, str) or re.fullmatch("[0-9a-f]{64}", sha) is None:
            raise ValueError("invalid SUN hash")
        if (
            not archive.startswith("shards/")
            or not archive.endswith(".tar")
            or PurePosixPath(member).parent.as_posix() != "images"
            or PurePosixPath(member).stem != sha
            or PurePosixPath(member).suffix.lower()
            not in {".jpg", ".png", ".gif", ".bmp"}
        ):
            raise ValueError("invalid SUN archive or content-addressed member")
        if type(r["bytes"]) is not int or r["bytes"] < 1:
            raise ValueError("invalid SUN byte count")
        key = (archive, member)
        if key in seen:
            raise ValueError("duplicate SUN occurrence")
        seen.add(key)
        if archive not in archives:
            # ExitStack owns every opened shard until both validation and staging finish.
            tar = stack.enter_context(tarfile.open(file_at(protocol, archive), "r:"))  # noqa: SIM115
            index = {}
            for item in tar.getmembers():
                relative(item.name.rstrip("/") if item.isdir() else item.name)
                if item.name in index or not (item.isfile() or item.isdir()):
                    raise ValueError("ambiguous or linked TAR member")
                index[item.name] = item
            archives[archive] = (tar, index)
        tar, index = archives[archive]
        if member not in index or not index[member].isfile():
            raise ValueError("missing SUN member")
        if index[member].size != r["bytes"]:
            raise ValueError("SUN byte count mismatch")
        rows.append(
            {
                "split": sp,
                "path": f"images/{len(rows):08d}{PurePosixPath(member).suffix}",
                "target": label,
                "source": archive + "::" + member,
                "expected_sha256": sha,
                "archive": archive,
                "member": member,
            }
        )
    if set(names) != set(range(counts[2])):
        raise ValueError("SUN class coverage differs")
    classes = [names[i] for i in range(counts[2])]
    if len(set(classes)) != len(classes):
        raise ValueError("SUN class names are not unique")
    return (
        classes,
        rows,
        {
            "upstream_filename_join_verified": False,
            "label_order": "manifest numeric IDs; no relabeling",
            "evaluation": "redistributed split 1; upstream filenames unavailable",
        },
        {"manifest.jsonl": digest(raw)},
        archives,
    )


def convert(dataset, root, out, *, fixture_counts=None, eval_data_root=None):
    profiles = COUNTS | prepared_images.COUNTS
    if dataset not in profiles:
        raise ValueError("no verified native image profile")
    root, out = Path(root), Path(out)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source must be a directory")
    root = root.resolve()
    if out.resolve().is_relative_to(root):
        raise ValueError("output must be outside read-only sources")
    if out.exists() or out.is_symlink():
        raise FileExistsError("output already exists")
    evaluation = None
    if eval_data_root is not None:
        if dataset not in prepared_images.SUBSETS:
            raise ValueError("separate evaluation root is only for ImageNet subsets")
        evaluation = Path(eval_data_root)
        if evaluation.is_symlink():
            raise ValueError("evaluation source symlinks are not accepted")
        if not evaluation.is_dir():
            raise ValueError("evaluation source must be a directory")
        evaluation = evaluation.resolve()
        if out.resolve().is_relative_to(evaluation):
            raise ValueError("output must be outside read-only evaluation sources")
    counts = profiles[dataset] if fixture_counts is None else fixture_counts
    if len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("counts require positive train/evaluation/classes")
    with ExitStack() as stack:
        archives = {}
        if dataset in prepared_images.COUNTS:
            classes, rows, extra, hashes = prepared_images.membership(
                dataset,
                root,
                counts,
                fixture=fixture_counts is not None,
                eval_root=evaluation,
            )
        elif dataset == "sun397":
            classes, rows, extra, hashes, archives = sun(root, counts, stack)
            rows.sort(key=lambda r: r["archive"] + "/" + r["member"])
        else:
            classes, rows, extra, hashes = clue(root, counts)
        splits = {"train": [], "validation": []}
        data = {"schema_version": 1, "classes": classes, **splits}
        source_rows = []
        content = {"train": set(), "test": set()}

        def read(row):
            if "archive" not in row:
                return file_at(row.get("source_root", root), row["source"]).read_bytes()
            tar, index = archives[row["archive"]]
            handle = tar.extractfile(index[row["member"]])
            if handle is None:
                raise ValueError("missing SUN payload")
            with handle:
                return handle.read()

        for sp, expected in zip(("train", "test"), counts[:2]):
            items = [r for r in rows if r["split"] == sp]
            if len(items) != expected:
                raise ValueError("native split count differs")
            if dataset == "sun397" and {r["target"] for r in items} != set(
                range(counts[2])
            ):
                raise ValueError("SUN split class coverage differs")
            if (
                dataset == "sun397"
                and fixture_counts is None
                and set(Counter(r["target"] for r in items).values()) != {50}
            ):
                raise ValueError("SUN requires 50 occurrences per class per split")
            for row in items:
                raw = read(row)
                sha = digest(raw)
                checked_image(raw)
                if "expected_sha256" in row and sha != row["expected_sha256"]:
                    raise ValueError("SUN content hash mismatch")
                row["sha256"] = sha
                content[sp].add(sha)
                source_rows.append(
                    {"path": row["path"], "source": row["source"], "sha256": sha}
                )
                if "source_role" in row:
                    source_rows[-1]["source_role"] = row["source_role"]
                splits["train" if sp == "train" else "validation"].append(
                    {"path": row["path"], "target": row["target"]}
                )
        overlap = len(content["train"] & content["test"])
        if overlap and dataset in {"clue_footprint", "clue_stools"}:
            raise ValueError(
                "cross-split content overlap is not authorized for this CLUE profile"
            )
        evidence = dict(
            dataset=dataset,
            fixture=fixture_counts is not None,
            split_counts=list(counts),
            authenticity_verified=False,
            cross_split_content_hashes=overlap,
            content_overlap_policy="preserve and disclose; no filtering",
            annotation_sha256=hashes,
            **extra,
        )
        evidence["membership_sha256"] = digest(
            json.dumps(source_rows, sort_keys=True).encode()
        )
        data["split_evidence"] = json.dumps(evidence, sort_keys=True)

        def assets():
            for row in rows:
                raw = read(row)
                if digest(raw) != row["sha256"]:
                    raise ValueError("source changed during staging")
                yield row["path"], raw

        write(out, data, source_rows, assets())
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    if set(cfg) not in (
        {"dataset", "data_root"},
        {"dataset", "data_root", "eval_data_root"},
    ):
        raise ValueError(
            "config requires dataset/data_root and optional eval_data_root"
        )
    kwargs = (
        {"eval_data_root": cfg["eval_data_root"]} if "eval_data_root" in cfg else {}
    )
    convert(cfg["dataset"], cfg["data_root"], args.out, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
