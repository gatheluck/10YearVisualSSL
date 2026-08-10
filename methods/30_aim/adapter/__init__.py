"""Adapter for 30_aim, linear evaluation only (AIM; arXiv:2401.08541).

    python -m adapter --config <resolved.json> --out <dir>

An eval-only port -- a `linear_eval` stage and no step 1, the AIM sibling of
28_dinov2 / 36_franca. In the capture, AIM's "Step 1" is an as-is SSL comparison:
freeze the official pretrained AIM-600M (ViT-H/14) and fit a linear probe on
frozen features, because the from-scratch data (DFN-2B+, ~2B uncurated images) is
not public. That from-scratch autoregressive pretraining is the capture's excluded
step. So this port trains nothing and produces no `encoder.pt`; it probes a frozen,
downloaded backbone.

The model is the pinned upstream under `third_party/ml-aim` (Apple's official
`aim` package), imported and never copied. A real run needs the official AIM-600M
backbone checkpoint (a hash-pinned download named in the config as `ckpt`, fetched
by bin/fetch-weights.py against the backbone_artifact in provenance.json); the
hermetic smoke leaves `ckpt` empty and builds a random tiny AIM. The
representation is a genuine SSL ViT, so the number is comparable.

Licence note: the AIM code (apple/ml-aim) and the AIM-600M weights (apple/AIM) are
released under the Apple ML Research licence (apple-amlr), which restricts use to
non-commercial research. This port is used only for academic research; see
provenance.json and README.md. Nothing under that licence is copied into this
repository -- the code is a pinned submodule and the weights are a download.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "30_aim"
STAGES = ("linear_eval",)
METHOD_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = METHOD_DIR.parent.parent / "third_party" / "ml-aim" / "aim-v1"

# The pinned upstream, recorded in every manifest. Pinned directly (no fork): the
# frozen backbone forward needs no patch.
UPSTREAM = {
    "repo": "https://github.com/apple/ml-aim",
    "commit": "bd9e06893f3ad68bda56b34ac2158fda2680b7fc",
}

# Settings that select and load the backbone ...
MODEL_KEYS = frozenset({"name", "ckpt", "img_size", "patch_size", "embed_dim",
                        "num_blocks", "num_heads", "num_feature_layers"})
# ... and the probe's own hyperparameters.
PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr", "momentum",
                        "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | PROBE_KEYS
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

LINEAR_EVAL_METRIC_NAMES = {
    "best_top1_acc": "best_linear_probe_top1_accuracy",
    "best_top5_acc_at_best_top1": "best_linear_probe_top5_accuracy",
    "final_top1_acc": "final_linear_probe_top1_accuracy",
    "final_top5_acc": "final_linear_probe_top5_accuracy",
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}


class ConfigError(Exception):
    """A refusal, always naming what was refused."""


def _named(missing, unknown, where: str) -> None:
    if missing:
        raise ConfigError(
            f"{where}: missing {', '.join(sorted(missing))}. No default is "
            "filled in here -- the resolved config has to say what ran")
    if unknown:
        raise ConfigError(
            f"{where}: unknown {', '.join(sorted(unknown))}. A key that is "
            "ignored is a setting that never took effect")


def to_run_config(config: dict, out: Path) -> dict:
    """Validate the resolved config. There is one stage, so this only checks;
    the evaluation reads the config directly."""
    for key in ("output", "result_dir", "checkpoint_dir"):
        if key in config:
            raise ConfigError(
                f"config: {key} is set. The output location is not a setting: "
                "the contract fixes it at --out, and a config naming a "
                "directory would claim a location that was not used")

    stage = config.get("stage")
    if stage not in STAGES:
        raise ConfigError(
            f"config: stage is {stage!r}; the only stage this port has is "
            f"{STAGES[0]} (AIM's from-scratch pretraining on the unavailable "
            "DFN-2B+ is the capture's excluded step)")

    _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    _named(EVAL_TRAIN_KEYS - set(train), set(train) - EVAL_TRAIN_KEYS,
           "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")
    return {"stage": stage}


def _filter_numeric(raw: dict) -> tuple:
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    return metrics, unusable


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Probe the frozen backbone. Reads no encoder.pt; the backbone is the
    pinned upstream, loaded from the checkpoint named in the config (or random
    when none is given, for the hermetic smoke)."""
    to_run_config(config, out)          # validate before running
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_aim import run as _run
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config, official_dir=UPSTREAM_DIR) or {}
    metrics, unusable = _filter_numeric(raw)
    unusable += sum(1 for k in ("best_top1_acc", "final_top1_acc")
                    if k not in metrics)
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def body(ctx: adapterlib.Context) -> None:
    ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                      names=LINEAR_EVAL_METRIC_NAMES)


def _stage_of(config_path) -> str:
    import json
    try:
        return json.loads(Path(config_path).read_text(
            encoding="utf-8")).get("stage") or STAGES[0]
    except (OSError, ValueError, AttributeError):
        return STAGES[0]


def _absent_reason(config_path) -> "str | None":
    """CONTRACT section 3. This port trains nothing and produces no encoder; it
    probes a frozen, downloaded backbone. Saying so is required."""
    if _stage_of(config_path) != "linear_eval":
        return None
    return ("this port fits a linear probe on a frozen, pretrained backbone and "
            "produces a classifier, not an encoder; it trains nothing and the "
            "backbone it read is named in the config")


def main(argv: "list[str] | None" = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=_stage_of(a.config), body=body,
                              upstream=UPSTREAM,
                              encoder_absent_reason=_absent_reason(a.config))
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
