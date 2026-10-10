"""Audit explicitly pinned historical downstream records without changing them."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from downstream.accounting import summarize, validate_seeds

IDENTITY = frozenset(
    {
        "dataset",
        "model",
        "track",
        "protocol_id",
        "protocol_sha256",
        "checkpoint_identity",
        "recipe",
        "feature_layer",
        "evaluation_split",
        "effective_batch",
        "realized_lr",
        "world_size",
        "per_gpu_batch",
        "accumulation",
    }
)

BASIC5_IDENTITY = frozenset(
    {
        "dataset",
        "method_id",
        "protocol_id",
        "checkpoint_identity",
        "feature_layer",
        "effective_batch",
        "realized_lr",
        "world_size",
    }
)
BASIC5_EPOCH_IDENTITY = BASIC5_IDENTITY | {
    "campaign",
    "track",
    "protocol_sha256",
    "input_size",
    "eval_views",
    "per_gpu_batch",
    "accumulation",
}
BASIC5_SCHEDULED_IDENTITY = BASIC5_IDENTITY | {
    "protocol_hash",
    "input_resolution",
    "evaluation_views",
    "micro_batch_per_gpu",
    "grad_accum",
}
BASIC5_PROTOCOLS = {
    "LINEAR": "BASIC5_FAIR_v1",
    "ATTENTIVE": "BASIC5_ATTENTIVE_v1",
    "FINETUNE": "BASIC5_FINETUNE_v1",
}


def source_schema(cell):
    if "source_schema" not in cell:
        return "extended_lp_ap_v1"
    value = cell["source_schema"]
    if value not in ("basic5_epoch_v1", "basic5_scheduled_v1"):
        raise ValueError("unknown native result schema")
    return value


def artifacts(cell):
    if source_schema(cell) == "basic5_scheduled_v1":
        return ("result", "resume_meta")
    return ("result", "completion", "resume_meta")


class InvalidRecord(ValueError):
    """A fixed diagnostic that contains no source paths or arbitrary values."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise InvalidRecord("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise InvalidRecord("nonfinite JSON number")


def read_json(raw):
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise InvalidRecord("JSON object required")
    return value


def pinned(descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise InvalidRecord("artifact requires path and sha256")
    sha = descriptor["sha256"]
    raw = Path(descriptor["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise InvalidRecord("artifact hash mismatch")
    return read_json(raw)


def positive(value):
    return type(value) is int and value > 0


def validate_cell(cell):
    if not isinstance(cell, dict) or not cell:
        raise ValueError("cell must be a nonempty object")
    schema = source_schema(cell)
    required = {
        "cell_id",
        "identity",
        "expected_seeds",
        "epochs",
        "evaluation_count",
        "metric",
        "runs",
    }
    if schema != "extended_lp_ap_v1":
        required |= {"source_schema", "track"}
    if set(cell) != required:
        raise ValueError("cell has missing or unknown fields")
    if not isinstance(cell["cell_id"], str) or not cell["cell_id"].strip():
        raise ValueError("cell_id must be nonempty")
    validate_seeds(cell["expected_seeds"])
    if not positive(cell["epochs"]) or not positive(cell["evaluation_count"]):
        raise ValueError("epoch and evaluation counts must be positive integers")
    identity = cell["identity"]
    fields = {
        "extended_lp_ap_v1": IDENTITY,
        "basic5_epoch_v1": BASIC5_EPOCH_IDENTITY,
        "basic5_scheduled_v1": BASIC5_SCHEDULED_IDENTITY,
    }[schema]
    if not isinstance(identity, dict) or not fields <= identity.keys():
        raise ValueError("explicit reference identity fields are required")
    if schema == "extended_lp_ap_v1":
        if identity["track"] not in ("LINEAR", "ATTENTIVE"):
            raise ValueError("only native Extended LP/AP result schemas are accepted")
    else:
        if cell["track"] not in ("LINEAR", "ATTENTIVE", "FINETUNE"):
            raise ValueError("BasicFive track must be explicit")
        if identity["protocol_id"] != BASIC5_PROTOCOLS[cell["track"]]:
            raise ValueError("BasicFive track and native protocol disagree")
        if schema == "basic5_epoch_v1" and identity["track"] != cell["track"]:
            raise ValueError("BasicFive track identities disagree")
    if any(identity[k] is None for k in fields):
        raise ValueError("reference identity fields cannot be null")
    if (
        not isinstance(identity["checkpoint_identity"], dict)
        or not identity["checkpoint_identity"]
    ):
        raise ValueError("checkpoint identity must be explicit")
    hash_field = (
        "protocol_hash" if schema == "basic5_scheduled_v1" else "protocol_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(identity[hash_field])) is None:
        raise ValueError("protocol hash is required")
    canonical(identity)
    metric = cell["metric"]
    if (
        not isinstance(metric, dict)
        or set(metric) != {"field", "name", "unit", "direction"}
        or any(not isinstance(v, str) or not v.strip() for v in metric.values())
        or metric["direction"] not in ("higher", "lower")
    ):
        raise ValueError("explicit metric field/name/unit/direction required")
    if not isinstance(cell["runs"], list):
        raise TypeError("runs must be an explicit list")
    seen = set()
    for run in cell["runs"]:
        if not isinstance(run, dict) or set(run) != {
            "seed",
            "exit_status",
            *artifacts(cell),
        }:
            raise ValueError(
                "run requires seed, recorded exit and schema-specific pinned artifacts"
            )
        seed = run["seed"]
        if type(seed) is not int or seed not in cell["expected_seeds"] or seed in seen:
            raise ValueError("unexpected or duplicate seed; select attempts explicitly")
        seen.add(seed)


def inspect_run(cell, run):
    if type(run["exit_status"]) is not int or run["exit_status"] != 0:
        raise InvalidRecord("recorded successful exit required")
    result = pinned(run["result"])
    schema = source_schema(cell)
    if "completion" in artifacts(cell):
        completion = pinned(run["completion"])
        if canonical(result) != canonical(completion):
            raise InvalidRecord("result and completion disagree")
    meta = pinned(run["resume_meta"])
    if result.get("status") != "COMPLETED_VALID":
        raise InvalidRecord("result status is not completed")
    if type(result.get("seed")) is not int or result["seed"] != run["seed"]:
        raise InvalidRecord("result seed differs")
    for key, expected in cell["identity"].items():
        if key not in result or canonical(result[key]) != canonical(expected):
            raise InvalidRecord("reference identity differs")
    if (
        schema == "extended_lp_ap_v1"
        and result.get("protocol") != cell["identity"]["protocol_id"]
    ):
        raise InvalidRecord("protocol aliases disagree")
    epoch_fields = {
        "extended_lp_ap_v1": ("epochs", "final_scheduled_epoch"),
        "basic5_epoch_v1": ("epochs",),
        "basic5_scheduled_v1": ("epochs_scheduled", "epochs_run"),
    }[schema]
    for value in [*(result.get(k) for k in epoch_fields), meta.get("epoch")]:
        if type(value) is not int or value != cell["epochs"]:
            raise InvalidRecord("final scheduled epoch is not evidenced")
    final_flag = (
        "final_epoch_canonical"
        if schema == "basic5_scheduled_v1"
        else "final_epoch_only"
    )
    if result.get(final_flag) is not True:
        raise InvalidRecord("final-epoch selection is not declared")
    if schema != "extended_lp_ap_v1" and (
        result.get("smoke") is not False or result.get("canonical", True) is not True
    ):
        raise InvalidRecord("smoke or noncanonical record")
    if schema == "basic5_scheduled_v1":
        if result.get("canonical") is not True:
            raise InvalidRecord("canonical native selection is not declared")
        for key, expected in {
            "protocol": cell["identity"]["protocol_id"],
            "dataset": cell["identity"]["dataset"],
            "seed": run["seed"],
        }.items():
            if canonical(meta.get(key)) != canonical(expected):
                raise InvalidRecord("epoch metadata identity differs")
    metrics = result.get("metrics")
    count_types = (int, float) if schema == "basic5_scheduled_v1" else (int,)
    if (
        not isinstance(metrics, dict)
        or type(metrics.get("n")) not in count_types
        or metrics["n"] != cell["evaluation_count"]
    ):
        raise InvalidRecord("evaluated population differs")
    score = metrics.get(cell["metric"]["field"])
    if "n_all_zero_clips" in metrics and (
        type(metrics["n_all_zero_clips"]) is not int or metrics["n_all_zero_clips"] != 0
    ):
        raise InvalidRecord("blank input disclosure is invalid")
    if type(score) not in (int, float) or not math.isfinite(score):
        raise InvalidRecord("selected score is missing or nonfinite")
    return score


def audit(plan):
    if (
        not isinstance(plan, dict)
        or set(plan) != {"schema_version", "cells"}
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or not isinstance(plan["cells"], list)
        or not plan["cells"]
    ):
        raise ValueError("nonempty schema-version-1 cell plan required")
    identifiers = set()
    for cell in plan["cells"]:
        validate_cell(cell)
        if cell["cell_id"] in identifiers:
            raise ValueError("duplicate cell identifier")
        identifiers.add(cell["cell_id"])
    cells = []
    for cell in plan["cells"]:
        valid, invalid = [], []
        for run in cell["runs"]:
            try:
                score = inspect_run(cell, run)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # Neither input paths nor arbitrary values from source files are emitted.
                invalid.append(
                    {
                        "seed": run["seed"],
                        "reason": str(exc)
                        if isinstance(exc, InvalidRecord)
                        else "artifact_validation_failed",
                        "error_type": type(exc).__name__,
                    }
                )
            else:
                valid.append(
                    {
                        "seed": run["seed"],
                        "score": score,
                        "artifact_sha256": {
                            k: run[k]["sha256"] for k in artifacts(cell)
                        },
                    }
                )
        valid.sort(key=lambda item: item["seed"])
        seen = {r["seed"] for r in cell["runs"]}
        cells.append(
            {
                "cell_id": cell["cell_id"],
                **(
                    {"source_schema": cell["source_schema"], "track": cell["track"]}
                    if "source_schema" in cell
                    else {}
                ),
                "identity_sha256": hashlib.sha256(
                    canonical(cell["identity"])
                ).hexdigest(),
                "metric": dict(cell["metric"]),
                "expected_seeds": sorted(cell["expected_seeds"]),
                "epochs": cell["epochs"],
                "evaluation_count": cell["evaluation_count"],
                "completed_seeds": [r["seed"] for r in valid],
                "missing_seeds": sorted(set(cell["expected_seeds"]) - seen),
                "valid_runs": valid,
                "invalid_runs": invalid,
                "summary": summarize([r["score"] for r in valid]) if valid else None,
                "complete": len(valid) == len(cell["expected_seeds"]),
                "canonical_eligible": False,
                "record_value": False,
                "paper_score_verified": False,
            }
        )
    return {
        "schema_version": 1,
        "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "cells": cells,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit(read_json(Path(args.config).read_bytes()))
    with Path(args.out).open("x") as handle:
        handle.write(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
