"""Observed Flowers training uses trnid only; the companion says train+valid."""

import io
from collections import Counter

COUNTS = {"flowers102": (1020, 6149, 102)}


def membership(root, counts, *, fixture):
    from scipy.io import loadmat

    from downstream.native_images import digest, file_at

    hashes = {}

    def mat(name):
        raw = file_at(root, name).read_bytes()
        hashes[name] = digest(raw)
        try:
            return loadmat(io.BytesIO(raw), simplify_cells=True)
        except Exception as exc:
            raise ValueError("unreadable MAT annotation") from exc

    def vector(data, key):
        if key not in data:
            raise ValueError("missing annotation field")
        value = data[key]
        values = value.reshape(-1).tolist() if hasattr(value, "reshape") else [value]
        if not values or any(type(v) is not int or v < 1 for v in values):
            raise ValueError("annotation requires positive integer values")
        return values

    sets = mat("setid.mat")
    labels = vector(mat("imagelabels.mat"), "labels")
    if len(labels) != counts[0] * 2 + counts[1]:
        raise ValueError("annotation label count differs")
    if any(v > counts[2] for v in labels):
        raise ValueError("annotation label out of range")
    parts = {
        sp: vector(sets, key)
        for sp, key in (("train", "trnid"), ("valid", "valid"), ("test", "tstid"))
    }
    ids = [i for part in parts.values() for i in part]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate or overlapping annotation IDs")
    if set(ids) != set(range(1, len(labels) + 1)):
        raise ValueError("annotation ID coverage differs")
    if len(parts["valid"]) != counts[0]:
        raise ValueError("annotation validation count differs")
    rows = []
    for split, values in parts.items():
        targets = [labels[i - 1] for i in values]
        if set(targets) != set(range(1, counts[2] + 1)):
            raise ValueError("annotation split class coverage differs")
        if not fixture and split != "test" and set(Counter(targets).values()) != {10}:
            raise ValueError("Flowers train/valid require ten images per class")
        for ident in sorted(values):
            name = f"jpg/image_{ident:05d}.jpg"
            file_at(root, name)
            if split != "valid":
                rows.append(
                    {
                        "split": split,
                        "path": name,
                        "source": name,
                        "target": labels[ident - 1] - 1,
                    }
                )
    return (
        [str(i) for i in range(1, counts[2] + 1)],
        rows,
        {
            "official_train": "setid.mat:trnid",
            "official_evaluation": "setid.mat:tstid",
            "excluded_validation_count": len(parts["valid"]),
            "validation": "setid.mat:valid checked and excluded",
            "prepared_folders_used": False,
            "protocol_split_conflict": True,
            "membership_profile": "observed_flowers_trnid_v1",
            "historical_membership_verified": False,
        },
        hashes,
    )
