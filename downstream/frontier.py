"""Evaluate explicit matched-subset predictions from the frontier protocol."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from downstream.reference_accounting import canonical, read_json

# Dataset interfaces, not model identities. Vocabulary IDs come from the caller.
TASKS = {
    "ImageNet-1k": (1000, "class_id", "top1"),
    "SSv2": (174, "class_id", "top1"),
    "COCO": (80, "category_ids", "category_presence_micro_f1"),
    "ADE20K": (150, "point_class_ids", "semantic_point_accuracy"),
    "NYUv2": (0, "pair_answers", "relative_depth_accuracy"),
}


def _object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("object has missing or unknown fields")


def _list(value, count=None):
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise ValueError("list population differs from the protocol")
    return value


def _labels(value, allowed, *, count=None, unique=False):
    values = _list(value, count)
    if any(type(v) is not int or v not in allowed for v in values):
        raise ValueError("label is not an allowed integer")
    if unique and len(set(values)) != len(values):
        raise ValueError("duplicate category IDs")
    return values


def _point(value):
    for coordinate in _list(value, 2):
        if (
            type(coordinate) not in (int, float)
            or not math.isfinite(coordinate)
            or not 0 <= coordinate <= 1
        ):
            raise ValueError("query coordinate must be finite and normalized")


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("input requires a lowercase SHA256 identity")


def _answers(value, dataset, allowed, *, target=False):
    if dataset in ("ImageNet-1k", "SSv2"):
        return _labels([value], allowed, count=1)
    if dataset == "NYUv2":
        return _labels(value, {0, 1, 2} if target else {0, 1, 2, 3}, count=20)
    return _labels(
        value,
        allowed,
        count=32 if dataset == "ADE20K" else None,
        unique=dataset == "COCO",
    )


def evaluate(manifest, responses):
    """Score fixed, already-labelled inputs; never construct new ground truth."""
    _object(manifest, {"schema_version", "dataset", "class_ids", "samples"})
    dataset = manifest["dataset"]
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(dataset, str)
        or dataset not in TASKS
    ):
        raise ValueError("unsupported frontier manifest")
    size, field, metric = TASKS[dataset]
    vocabulary = _list(manifest["class_ids"], size)
    if (
        any(type(v) is not int or v < 0 for v in vocabulary)
        or len(set(vocabulary)) != size
    ):
        raise ValueError("vocabulary must contain distinct nonnegative integer IDs")
    allowed = set(vocabulary)
    samples = _list(manifest["samples"], 500)
    seen = set()
    targets = []
    for sample in samples:
        keys = {
            "sample_id",
            "target",
            "frame_sha256" if dataset == "SSv2" else "input_sha256",
        }
        if dataset == "ADE20K":
            keys.add("points")
        if dataset == "NYUv2":
            keys.add("pairs")
        _object(sample, keys)
        identifier = sample["sample_id"]
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in seen
        ):
            raise ValueError("sample identifiers must be nonempty and unique")
        seen.add(identifier)
        hashes = (
            _list(sample["frame_sha256"], 16)
            if dataset == "SSv2"
            else [sample["input_sha256"]]
        )
        for digest in hashes:
            _sha(digest)
        if dataset == "ADE20K":
            for point in _list(sample["points"], 32):
                _point(point)
        if dataset == "NYUv2":
            for pair in _list(sample["pairs"], 20):
                for point in _list(pair, 2):
                    _point(point)
        targets.append(_answers(sample["target"], dataset, allowed, target=True))
    predictions = []
    for batch_index, response in enumerate(_list(responses, 5)):
        _object(response, {"dataset", "predictions"})
        if response["dataset"] != dataset:
            raise ValueError("response dataset differs from manifest")
        for offset, prediction in enumerate(_list(response["predictions"], 100)):
            _object(prediction, {"sample_id", field})
            if (
                prediction["sample_id"]
                != samples[batch_index * 100 + offset]["sample_id"]
            ):
                raise ValueError("prediction identifiers or order differ from manifest")
            predictions.append(_answers(prediction[field], dataset, allowed))
    if dataset == "COCO":
        tp = sum(len(set(t) & set(p)) for t, p in zip(targets, predictions))
        fp = sum(len(set(p) - set(t)) for t, p in zip(targets, predictions))
        fn = sum(len(set(t) - set(p)) for t, p in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            raise ValueError("category-presence Micro-F1 is undefined for empty totals")
        statistics = {"tp": tp, "fp": fp, "fn": fn}
        score = 100 * 2 * tp / denominator
    else:
        correct = sum(
            a == b for t, p in zip(targets, predictions) for a, b in zip(t, p)
        )
        decisions = sum(len(t) for t in targets)
        statistics = {"correct": correct, "decisions": decisions}
        score = 100 * correct / decisions
    return {
        "schema_version": 1,
        "dataset": dataset,
        "metric": metric,
        "score": score,
        "units": "percent",
        "samples": 500,
        "batches": 5,
        **statistics,
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "response_sha256": [
            hashlib.sha256(canonical(r)).hexdigest() for r in responses
        ],
        "canonical_eligible": False,
        "record_value": False,
        "paper_score_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--responses", nargs=5, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    raw_manifest = Path(args.manifest).read_bytes()
    raw_responses = [Path(path).read_bytes() for path in args.responses]
    report = evaluate(
        read_json(raw_manifest), [read_json(raw) for raw in raw_responses]
    )
    report["raw_manifest_sha256"] = hashlib.sha256(raw_manifest).hexdigest()
    report["raw_response_sha256"] = [
        hashlib.sha256(raw).hexdigest() for raw in raw_responses
    ]
    with Path(args.out).open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
