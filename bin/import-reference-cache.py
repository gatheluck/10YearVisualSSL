#!/usr/bin/env python3
"""Stage audited, hash-pinned validation caches in canonical sample order.

Profiles and provenance are trusted local inputs kept outside Git. Explicit
indices must come from the audited producer, never from sorting class labels.
This verifies cache integrity, not the scientific identity of its producer.
"""
import importlib.util
import json
from pathlib import Path


def _tool(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_cache(profile, output):
    import numpy as np
    writer = _tool("extract-features.py", "cache_writer")
    reference = _tool("extract-reference.py", "cache_reference")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    def read(record):
        path = Path(record["path"])
        if writer.sha256_of(path) != record["sha256"]:
            raise ValueError(f"source hash changed: {path}")
        return np.load(path, allow_pickle=False)

    expected = read(profile["expected_labels"])
    shards = profile["shards"]
    if not shards:
        raise ValueError("no cache shards")
    features, labels, indices = [], [], []
    for shard in shards:
        x, y, ix = (read(shard[key]) for key in ("features", "labels", "indices"))
        if y.dtype.kind not in "iu" or ix.dtype.kind not in "iu":
            raise ValueError("labels and indices must be integers")
        if x.ndim != 2 or x.shape[1] != profile["feat_dim"]:
            raise ValueError("cache feature width mismatch")
        if y.shape != (len(x),) or ix.shape != (len(x),):
            raise ValueError("cache shard row mismatch")
        features.append(x)
        labels.append(y)
        indices.append(ix)
    x, y = reference.restore_sample_order(np.concatenate(features), np.concatenate(labels),
                                         np.concatenate(indices), profile["count"])
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)
    reference.validate_features(x, y, expected, profile["count"])
    normalized = writer.apply_representation(x, "l2")
    if not np.allclose(np.linalg.norm(normalized, axis=1), 1., atol=1e-5):
        raise ValueError("features cannot be normalized to unit length")
    output.mkdir(parents=True, exist_ok=False)
    meta = dict(method=profile["id"], count=len(y), feat_dim=x.shape[1],
                representation="l2", source_profile=profile,
                arch=profile.get("method", profile["id"]))
    writer.save_features(output, normalized, y, meta)
    (output / "result.json").write_text(json.dumps(dict(status="ok", count=len(y), feat_dim=x.shape[1]), indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import_cache(json.loads(Path(args.profile).read_text()), args.out)
