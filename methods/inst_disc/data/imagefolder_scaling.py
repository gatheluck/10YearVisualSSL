"""Source-ordered ImageFolder membership; fail rather than substitute gray images."""
import hashlib
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from data.indexed_tar import validate_output


def freeze(train_root, validation_root, output):
    train_root, validation_root, output = map(Path, (train_root, validation_root, output))
    validate_output(output, train_root, validation_root)
    if not validation_root.is_dir():
        raise FileNotFoundError("validation root missing; cannot verify filename exclusions")
    folder = ImageFolder(train_root)
    names = {p.name for p in validation_root.rglob("*") if p.is_file()}
    rows, excluded = [], []
    for path, label in folder.samples:
        path = Path(path)
        relative = path.relative_to(train_root).as_posix()
        if path.name.startswith("ILSVRC2012_val_") or path.name in names:
            excluded.append({"relpath": relative, "reason": "imagenet1k_val_filename"})
            continue
        rows.append({"instance_id": len(rows), "relpath": relative,
                         "class_index": label, "class_name": folder.classes[label]})
    if not rows:
        raise ValueError("training population empty after exclusions")
    output.mkdir(parents=True, exist_ok=False)
    data = "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n" for row in rows).encode()
    (output / "TRAIN_MANIFEST.jsonl").write_bytes(data)
    (output / "EXCLUSIONS.jsonl").write_text("".join(json.dumps(row) + "\n" for row in excluded))
    meta = {"n_train": len(rows), "n_excluded": len(excluded), "n_classes": len(folder.classes),
            "source_n": len(folder.samples), "manifest_sha256": hashlib.sha256(data).hexdigest(),
            "content_overlap_checked": False}
    (output / "TRAIN_MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class FrozenImageFolder(Dataset):
    def __init__(self, manifest, train_root, transform):
        self.root = Path(train_root).resolve()
        self.transform = transform
        self.entries = []
        for i, line in enumerate(Path(manifest).read_text().splitlines()):
            row = json.loads(line)
            relative = row.get("relpath")
            if (type(row.get("instance_id")) is not int or row["instance_id"] != i
                    or type(row.get("class_index")) is not int or row["class_index"] < 0
                    or not isinstance(relative, str) or not relative
                    or Path(relative).is_absolute() or ".." in Path(relative).parts):
                raise ValueError("invalid frozen ImageFolder row or escaping path")
            self.entries.append(row)
        if not self.entries:
            raise ValueError("empty frozen ImageFolder manifest")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        path = (self.root / row["relpath"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("image symlink escapes training root")
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            raise ValueError(f"unreadable frozen image {index}: {type(exc).__name__}") from exc
        return self.transform(image), index, row["class_index"]
