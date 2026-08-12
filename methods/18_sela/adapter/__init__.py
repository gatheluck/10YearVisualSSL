"""Adapter for 18_sela, step 1 and linear evaluation (Asano et al., ICLR 2020).

    python -m adapter --config <resolved.json> --out <dir>

SeLa (Self-Labelling): a ResNet backbone with linear prototype heads is trained to
predict pseudo-labels computed by Sinkhorn-Knopp optimal transport (which forces a
balanced assignment over K clusters), with cross-entropy on the hard targets,
averaged over the heads (step 1). linear_eval then probes the frozen backbone
(2048-d). A self-contained re-implementation (the lab's own code) -- no submodule.
The capture's step 2 (ViT, which needs timm) is excluded, as in every port.

`encoder.pt` is the ResNet backbone (`backbone.*`); the prototype heads
(`top_layer.*`) are training machinery and are left out. `linear_eval` reads this
`encoder.pt`; the representation is the model this port trains, so the probe number
is a genuine, comparable linear probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "18_sela"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

MODEL_KEYS = frozenset({"arch", "image_size"})
CLUSTERING_KEYS = frozenset({"k", "num_heads", "nopts", "sinkhorn_max_iters",
                             "sinkhorn_tol", "lambda", "epsilon"})
STEP1_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers",
                              "learning_rate", "momentum", "weight_decay",
                              "lr_schedule", "lr_step_size", "lr_gamma",
                              "temperature", "assignment_mode"})
STEP1_TRAIN_KEYS = MODEL_KEYS | CLUSTERING_KEYS | STEP1_TRAIN_ONLY
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# The ResNet backbone. The prototype heads (top_layer) are excluded.
ENCODER_PREFIXES = ("backbone.",)

STEP1_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

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


def _clustering_section(train: dict) -> dict:
    return {"k": int(train["k"]), "num_heads": int(train["num_heads"]),
            "nopts": int(train["nopts"]),
            "sinkhorn_max_iters": int(train["sinkhorn_max_iters"]),
            "sinkhorn_tol": float(train["sinkhorn_tol"]),
            "lambda": float(train["lambda"]),
            "epsilon": float(train["epsilon"])}


def _training_section(train: dict) -> dict:
    return {"epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "num_workers": int(train["num_workers"]),
            "learning_rate": float(train["learning_rate"]),
            "momentum": float(train["momentum"]),
            "weight_decay": float(train["weight_decay"]),
            "lr_schedule": str(train["lr_schedule"]),
            "lr_step_size": int(train["lr_step_size"]),
            "lr_gamma": float(train["lr_gamma"]),
            "temperature": float(train["temperature"]),
            "assignment_mode": str(train["assignment_mode"])}


def to_run_config(config: dict, out: Path) -> dict:
    for key in ("output", "checkpoint"):
        if key in config:
            raise ConfigError(
                f"config: {key} is set. The output location is not a setting: "
                "the contract fixes it at --out, and a config naming a "
                "directory would claim a location that was not used")

    stage = config.get("stage")
    if stage not in STAGES:
        raise ConfigError(
            f"config: stage is {stage!r}; known stages are "
            f"{', '.join(STAGES)}")
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else STEP1_TRAIN_KEYS
    top = EVAL_TOP_KEYS if stage == "linear_eval" else TOP_KEYS
    _named(top - set(config), set(config) - top, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    _named(keys - set(train), set(train) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        return {"stage": stage}

    return {
        "seed": int(config["seed"]),
        "model": {"arch": str(train["arch"])},
        "clustering": _clustering_section(train),
        "data": {"data_root": str(config["data_root"]),
                 "image_size": int(train["image_size"])},
        "training": _training_section(train),
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    out = {k: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIXES)}
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIXES} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import ResNetSeLa
    from train_step1_sela import model_config
    # The prototype heads are excluded from encoder.pt, so k / num_heads (which
    # only shape them) are free here; the backbone is what loads back.
    model = ResNetSeLa(num_classes=1, num_heads=1, **model_config(config["train"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The "
            "prototype heads are expected to be missing; the backbone is not")
    return model


def _filter_numeric(raw: dict) -> tuple:
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    return metrics, unusable


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from train_step1_sela import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True,
                                                       exist_ok=True)
    raw = _run(args, run_config) or {}
    metrics, unusable = _filter_numeric(raw)
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    import torch
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_sela import run as _run
    state = torch.load(config["encoder"], map_location="cpu", weights_only=True)
    model = load_encoder(state, config)
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config, model=model) or {}
    metrics, unusable = _filter_numeric(raw)
    unusable += sum(1 for k in ("best_top1_acc", "final_top1_acc")
                    if k not in metrics)
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def body(ctx: adapterlib.Context) -> None:
    import torch
    if ctx.stage == "linear_eval":
        ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                          names=LINEAR_EVAL_METRIC_NAMES)
        return
    metrics = run_training(ctx.config, ctx.out)
    latest = Path(ctx.out) / WORK / "checkpoint_latest.pth"
    if not latest.is_file():
        raise RuntimeError(
            f"training finished but {latest} was not written; there is no "
            "encoder to hand over")
    state = torch.load(latest, map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model_state_dict"]),
               Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=STEP1_METRIC_NAMES)


def _stage_of(config_path) -> str:
    import json
    try:
        return json.loads(Path(config_path).read_text(
            encoding="utf-8")).get("stage") or STAGES[0]
    except (OSError, ValueError, AttributeError):
        return STAGES[0]


def _absent_reason(config_path) -> "str | None":
    if _stage_of(config_path) != "linear_eval":
        return None
    return ("this stage fits a linear probe on the frozen SeLa backbone and "
            "produces a classifier, not an encoder; it reads the encoder.pt "
            "named in the config")


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=_stage_of(a.config), body=body,
                              encoder_absent_reason=_absent_reason(a.config))
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
