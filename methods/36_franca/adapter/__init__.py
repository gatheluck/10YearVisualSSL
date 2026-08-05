"""Adapter for 36_franca, linear evaluation only (Franca; arXiv:2507.14137).

    python -m adapter --config <resolved.json> --out <dir>

**The first eval-only port: a `linear_eval` stage and no step 1.** In the
capture, Franca's "Step 1" is a caveat eval -- freeze the official pretrained
Franca ViT-B/14 In21K backbone and fit a linear probe on frozen CLS features,
"analogous to DINOv2 ... not local Franca pretraining". The from-scratch SSL
pretraining is the capture's Step 2 (H100-class) and is excluded like every
method's step 2. So this port trains nothing and produces no `encoder.pt`; it
probes a frozen, downloaded backbone.

The model is the pinned upstream under `third_party/franca`, imported and never
copied, and pinned **directly** (no fork): the frozen forward has no hardcoded
device. A real run needs the official checkpoint (a hash-pinned download named in
the config as `ckpt`); the hermetic smoke leaves `ckpt` empty and builds a random
backbone, so nothing is downloaded. Unlike var's tokeniser probe, the
representation is a genuine SSL ViT, so the number is comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "36_franca"
STAGES = ("linear_eval",)
METHOD_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = METHOD_DIR.parent.parent / "third_party" / "franca"

# The pinned upstream, recorded in every manifest. Pinned directly (no fork):
# the frozen backbone forward needs no patch.
UPSTREAM = {
    "repo": "https://github.com/valeoai/Franca",
    "commit": "52653cdd2f94fc7e4dd12655cf326b181a48091d",
}

# Settings that select and load the backbone ...
MODEL_KEYS = frozenset({"name", "weights", "ckpt", "resolution", "feature_key",
                        "use_rasa_head"})
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
            f"{STAGES[0]} (Franca's from-scratch pretraining is the capture's "
            "excluded step 2)")

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
        from evaluate_linear_franca import run as _run
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


def main(argv: list[str] | None = None) -> int:
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
