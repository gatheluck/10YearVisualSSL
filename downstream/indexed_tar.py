"""Read-only indexed-tar inputs; frozen membership and no gray-image fallback."""
import gzip
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def validate_output(output, *inputs):
    output = Path(output).resolve()
    for value in inputs:
        source = Path(value).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output and original input locations overlap")


def checked_row(row, index):
    for key in ("sample_id", "synset", "member_name", "tar_relpath"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"invalid sample {index}: {key}")
    path = Path(row["tar_relpath"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid sample {index}: tar path escapes root")
    for key, minimum in (("offset", 0), ("size", 1)):
        if type(row.get(key)) is not int or row[key] < minimum:
            raise ValueError(f"invalid sample {index}: {key}")
    if "instance_id" in row and (type(row["instance_id"]) is not int or row["instance_id"] != index):
        raise ValueError(f"invalid sample {index}: noncontiguous instance_id")
    return row


def freeze(prepared_root, validation_root, output):
    prepared_root, validation_root, output = map(Path, (prepared_root, validation_root, output))
    validate_output(output, prepared_root, validation_root)
    if not validation_root.is_dir():
        raise FileNotFoundError("validation root missing; cannot verify filename exclusions")
    success = json.loads((prepared_root / "SUCCESS.json").read_text())
    names = set()
    for path in validation_root.rglob("*"):
        if path.is_file():
            names.update((path.name, path.stem))
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(); source_digest = hashlib.sha256(); count = excluded = source_count = 0
    with (prepared_root / "MANIFEST.jsonl").open("rb") as source, (output / "TRAIN_MANIFEST.jsonl").open("wb") as target, (output / "EXCLUSIONS.jsonl").open("w") as exclusions:
        for line in source:
            source_digest.update(line)
            row = json.loads(line)
            # Source identifiers may predate filtering; renumber only kept rows.
            row.pop("instance_id", None)
            checked_row(row, source_count)
            source_count += 1
            name = Path(row["member_name"]).name
            if name.startswith("ILSVRC2012_val_") or name in names or Path(name).stem in names:
                exclusions.write(json.dumps({"sample_id": row["sample_id"], "reason": "imagenet1k_val_filename"}) + "\n")
                excluded += 1
                continue
            row["instance_id"] = count
            data = (json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
            target.write(data); digest.update(data); count += 1
    if not count:
        raise ValueError("training population empty after exclusions")
    meta = {"n_train": count, "n_excluded": excluded, "source_n": source_count,
            "manifest_sha256": digest.hexdigest(), "source_manifest_sha256": source_digest.hexdigest(),
            "source_success_sha256": hashlib.sha256(json.dumps(success, sort_keys=True).encode()).hexdigest(),
            "validation_name_count": len(names), "content_overlap_checked": False}
    (output / "TRAIN_MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class IndexedTarDataset(Dataset):
    def __init__(self, manifest, prepared_root, transform):
        self.root = Path(prepared_root).resolve()
        self.transform = transform
        self.entries = []
        with Path(manifest).open() as handle:
            for i, line in enumerate(handle):
                row = checked_row(json.loads(line), i)
                self.entries.append((row["sample_id"], row["tar_relpath"], row["offset"], row["size"]))
        if not self.entries:
            raise ValueError("empty indexed manifest")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        identity, relative, offset, size = self.entries[index]
        try:
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("tar symlink escapes prepared root")
            with path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read(size)
            if len(payload) != size:
                raise ValueError("truncated indexed payload")
            if payload[:2] == b"\x1f\x8b":
                payload = gzip.decompress(payload)
            with Image.open(io.BytesIO(payload)) as im:
                image = im.convert("RGB")
        except Exception as exc:
            raise ValueError(f"unreadable indexed sample {identity}: {type(exc).__name__}") from exc
        return self.transform(image), index, 0
