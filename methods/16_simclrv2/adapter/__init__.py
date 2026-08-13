"""Adapter for 16_simclrv2, step 1 and linear evaluation (Chen et al., 2020).

    python -m adapter --config <resolved.json> --out <dir>

SimCLR v2, the paper-faithful ResNet-50 path: two augmented views feed a shared
ResNet-50 + 3-layer MLP projection head; the NT-Xent loss pulls the matching pair
together and pushes every other view in the batch apart, optimised with LARS
(step 1). linear_eval then probes the frozen ResNet-50 backbone (2048-d). A
self-contained re-implementation (the lab's own code) -- no submodule. The
capture's ViT step 2 (which needs timm) is excluded, as in every port.

`encoder.pt` is the ResNet-50 backbone (`encoder.*`); the projection head is
training machinery and is left out. `linear_eval` reads this `encoder.pt`; the
representation is the model this port trains, so the probe number is a genuine,
comparable linear probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "16_simclrv2"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

MODEL_KEYS = frozenset({"out_dim", "img_size", "width_multiplier"})
LOSS_KEYS = frozenset({"temperature"})
DATA_KEYS = frozenset({"color_jitter_strength"})
PRETRAIN_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                              "momentum", "weight_decay", "eta",
                              "warmup_epochs"})
PRETRAIN_TRAIN_KEYS = MODEL_KEYS | LOSS_KEYS | DATA_KEYS | PRETRAIN_TRAIN_ONLY
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

# Step-2 unified ViT-B/16 (`arch: vit`); the native ResNet-50 pretrain (no arch)
# is unchanged. The ViT ignores width_multiplier (a ResNet knob).
ARCHS = ("resnet", "vit")
VIT_MODEL_KEYS = frozenset({"out_dim", "img_size", "patch_size", "embed_dim",
                            "depth", "num_heads", "mlp_ratio", "drop_rate",
                            "attn_drop_rate"})
VIT_LOSS_KEYS = frozenset({"temperature"})
VIT_DATA_KEYS = frozenset({"color_jitter_strength"})
PRETRAIN_VIT_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                               "weight_decay", "warmup_epochs", "min_lr",
                               "clip_grad", "save_at_epochs"})
PRETRAIN_VIT_KEYS = (VIT_MODEL_KEYS | VIT_LOSS_KEYS | VIT_DATA_KEYS
                     | PRETRAIN_VIT_ONLY)
EVAL_VIT_KEYS = VIT_MODEL_KEYS | EVAL_PROBE_KEYS
_VIT_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# The ResNet-50 backbone. The projection head is excluded.
ENCODER_PREFIXES = ("encoder.",)

PRETRAIN_METRIC_NAMES = {
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


def _model_section(train: dict) -> dict:
    return {"out_dim": int(train["out_dim"]),
            "img_size": int(train["img_size"]),
            "width_multiplier": int(train["width_multiplier"])}


def _loss_section(train: dict) -> dict:
    return {"temperature": float(train["temperature"])}


def _data_section(train: dict, data_root: str) -> dict:
    return {"data_root": str(data_root), "img_size": int(train["img_size"]),
            "color_jitter_strength": float(train["color_jitter_strength"])}


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
    top = EVAL_TOP_KEYS if stage == "linear_eval" else TOP_KEYS
    _named(top - set(config), set(config) - top, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    arch = train.get("arch", "resnet")
    if arch not in ARCHS:
        raise ConfigError(
            f"config.train: arch is {arch!r}; expected one of "
            f"{', '.join(ARCHS)}")
    rest = {k: v for k, v in train.items() if k != "arch"}
    if stage == "linear_eval":
        keys = EVAL_VIT_KEYS if arch == "vit" else EVAL_TRAIN_KEYS
    else:
        keys = PRETRAIN_VIT_KEYS if arch == "vit" else PRETRAIN_TRAIN_KEYS
    _named(keys - set(rest), set(rest) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        return {"stage": stage}

    if arch == "vit":
        return {
            "seed": int(config["seed"]),
            "arch": "vit",
            "model": {k: (float(train[k]) if k in _VIT_FLOATS else int(train[k]))
                      for k in VIT_MODEL_KEYS},
            "loss": {"temperature": float(train["temperature"])},
            "data": {"data_root": str(config["data_root"]),
                     "img_size": int(train["img_size"]),
                     "color_jitter_strength":
                         float(train["color_jitter_strength"])},
            "training": {"epochs": int(train["epochs"]),
                         "batch_size": int(train["batch_size"]),
                         "num_workers": int(train["num_workers"]),
                         "lr": float(train["lr"]),
                         "weight_decay": float(train["weight_decay"]),
                         "warmup_epochs": int(train["warmup_epochs"]),
                         "min_lr": float(train["min_lr"]),
                         "clip_grad": float(train["clip_grad"]),
                         "save_at_epochs": [int(e) for e in
                                            train["save_at_epochs"]]},
            "output": {"checkpoint_dir": str(Path(out) / WORK)},
        }

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "loss": _loss_section(train),
        "data": _data_section(train, config["data_root"]),
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "num_workers": int(train["num_workers"]),
                     "lr": float(train["lr"]),
                     "momentum": float(train["momentum"]),
                     "weight_decay": float(train["weight_decay"]),
                     "eta": float(train["eta"]),
                     "warmup_epochs": int(train["warmup_epochs"])},
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
    train = config["train"]
    if train.get("arch", "resnet") == "vit":
        from models.vit_simclrv2 import build_vit_simclrv2
        from train_pretrain_vit_simclrv2 import model_kwargs as vit_mk
        model = build_vit_simclrv2(**vit_mk(train))
    else:
        from models import build_resnet_simclrv2
        from train_pretrain_simclrv2 import model_config
        model = build_resnet_simclrv2(**model_config(train))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The "
            "projection head is expected to be missing; the backbone is not")
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
    arch = config.get("train", {}).get("arch", "resnet")
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        if arch == "vit":
            from train_pretrain_vit_simclrv2 import run as _run
        else:
            from train_pretrain_simclrv2 import run as _run
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
        from evaluate_linear_simclrv2 import run as _run
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
    train = ctx.config.get("train", {})
    if train.get("arch", "resnet") == "vit":
        for n in train.get("save_at_epochs", []):
            ck = Path(ctx.out) / WORK / f"checkpoint_epoch_{int(n)}.pth"
            if ck.is_file():
                s = torch.load(ck, map_location="cpu", weights_only=False)
                torch.save(extract_encoder(s["model_state_dict"]),
                           Path(ctx.out) / f"encoder_epoch{int(n)}.pt")
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


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
    return ("this stage fits a linear probe on the frozen SimCLR v2 backbone and "
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
