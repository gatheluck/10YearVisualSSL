"""Adapter for 35_vjepa, step 1 and linear evaluation (V-JEPA; arXiv:2404.08471).

    python -m adapter --config <resolved.json> --out <dir>

V-JEPA (image adaptation): a context encoder and an EMA target encoder (the official
facebookresearch/jepa video ViT + predictor, run at num_frames=1 so the backbone is
an image ViT) are trained by latent prediction over 3D multi-block masks (step 1).
linear_eval then probes the frozen target encoder (its mean-pooled tokens). The ViT,
predictor and mask collator are the official code, pinned as the submodule
third_party/jepa and imported (never copied).

Scope: this ports the capture's step 2 (the from-scratch unified-comparison image
adaptation of V-JEPA on ImageNet). The capture's step 1 (a caveat probe of the
released VIDEO ViT-H/16) is not ported.

Licence note: the jepa code is CC BY-NC 4.0 (non-commercial). This port is used
only for academic research; see provenance.json and README.md. Nothing under that
licence is copied into this repository -- the code is a pinned submodule.

`encoder.pt` is the EMA target encoder (the representation V-JEPA eval uses); the
context encoder and the predictor are training machinery and are not saved. It loads
into a rebuilt V-JEPA encoder whose mean-pooled tokens the probe reads. `linear_eval`
reads this `encoder.pt`; the representation is the model this port trains, so the
probe number is comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "35_vjepa"
STAGES = ("step1", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

# The pinned upstream, recorded in every manifest: the V-JEPA ViT + predictor +
# mask collator are the official facebookresearch/jepa code, imported not copied.
UPSTREAM = {
    "repo": "https://github.com/facebookresearch/jepa",
    "commit": "51c59d518fc63c08464af6de585f78ac0c7ed4d5",
}

MODEL_KEYS = frozenset({"model_name", "crop_size", "patch_size", "num_frames",
                        "tubelet_size", "pred_depth", "pred_embed_dim",
                        "uniform_power", "use_mask_tokens",
                        "zero_init_mask_tokens", "use_sdpa"})
DATA_KEYS = frozenset({"num_workers", "use_color_jitter"})
LOSS_KEYS = frozenset({"loss_exp", "reg_coeff"})
MASK_KEYS = frozenset({"mask"})
TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "start_lr", "final_lr",
                           "weight_decay", "final_weight_decay", "warmup_epochs",
                           "beta1", "beta2", "eps", "clip_grad", "ema_start",
                           "ema_final"})
STEP1_TRAIN_KEYS = (MODEL_KEYS | DATA_KEYS | LOSS_KEYS | MASK_KEYS
                    | TRAINING_KEYS)
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# encoder.pt is the whole EMA target encoder (a bare ViT wrapper, no separate
# projection head), so nothing is stripped.
ENCODER_PREFIX = ""

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


def _model_section(t: dict) -> dict:
    return {"model_name": str(t["model_name"]), "crop_size": int(t["crop_size"]),
            "patch_size": int(t["patch_size"]), "num_frames": int(t["num_frames"]),
            "tubelet_size": int(t["tubelet_size"]),
            "pred_depth": int(t["pred_depth"]),
            "pred_embed_dim": int(t["pred_embed_dim"]),
            "uniform_power": bool(t["uniform_power"]),
            "use_mask_tokens": bool(t["use_mask_tokens"]),
            "zero_init_mask_tokens": bool(t["zero_init_mask_tokens"]),
            "use_sdpa": bool(t["use_sdpa"])}


def _training_section(t: dict) -> dict:
    return {"epochs": int(t["epochs"]), "batch_size": int(t["batch_size"]),
            "lr": float(t["lr"]), "start_lr": float(t["start_lr"]),
            "final_lr": float(t["final_lr"]),
            "weight_decay": float(t["weight_decay"]),
            "final_weight_decay": float(t["final_weight_decay"]),
            "warmup_epochs": int(t["warmup_epochs"]),
            "beta1": float(t["beta1"]), "beta2": float(t["beta2"]),
            "eps": float(t["eps"]), "clip_grad": float(t["clip_grad"]),
            "ema_start": float(t["ema_start"]),
            "ema_final": float(t["ema_final"])}


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
            f"config: stage is {stage!r}; known stages are {', '.join(STAGES)}")
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else STEP1_TRAIN_KEYS
    top = EVAL_TOP_KEYS if stage == "linear_eval" else TOP_KEYS
    _named(top - set(config), set(config) - top, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, not a mapping")
    _named(keys - set(train), set(train) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        return {"stage": stage}

    if not isinstance(train["mask"], list) or not train["mask"]:
        raise ConfigError("config.train: mask must be a non-empty list of mask "
                          "block configs")

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "data": {"data_root": str(config["data_root"]),
                 "num_workers": int(train["num_workers"]),
                 "use_color_jitter": bool(train["use_color_jitter"])},
        "mask": train["mask"],
        "loss": {"loss_exp": float(train["loss_exp"]),
                 "reg_coeff": float(train["reg_coeff"])},
        "training": _training_section(train),
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    out = {k[len(ENCODER_PREFIX):]: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIX)}
    if not out:
        raise RuntimeError(
            "the target encoder state is empty; there is nothing to save as "
            "encoder.pt")
    return out


def load_encoder(state_dict: dict, config: dict):
    import torch
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_vjepa_encoder
    train = config["train"]
    encoder = build_vjepa_encoder({k: train[k] for k in MODEL_KEYS},
                                  torch.device("cpu"))
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys the encoder does not have: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"encoder.pt is missing encoder weights: {missing[:5]}")
    return encoder


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
        from train_step1_vjepa import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
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
        from evaluate_linear_vjepa import run as _run
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
    torch.save(extract_encoder(state["target_encoder_state_dict"]),
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
    return ("this stage fits a linear probe on the frozen V-JEPA target encoder "
            "and produces a classifier, not an encoder; it reads the encoder.pt "
            "named in the config")


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
