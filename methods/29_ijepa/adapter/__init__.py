"""Adapter for 29_ijepa, step 1 and linear evaluation (Assran et al., 2023).

    python -m adapter --config <resolved.json> --out <dir>

I-JEPA (joint-embedding predictive architecture): a context encoder sees a large
context block of patches; a narrow predictor predicts the representations of
several masked target blocks; the targets come from an EMA target encoder (a
momentum copy of the context encoder), under a smooth-L1 loss in latent space
(step 1). linear_eval then probes the frozen target ViT encoder (its mean-pooled
patch tokens, embed_dim). A self-contained re-implementation (the lab's own code,
following facebookresearch/ijepa); I-JEPA ships its own ViT, so the port is
torch-only -- no timm. The capture's step 2 (ViT-B) is excluded, as in every port.

`encoder.pt` is the target ViT encoder (`target_encoder.*`, the prefix stripped so
it loads straight into a plain VisionTransformer). The context encoder, the
predictor and the mask token are training machinery and are left out. The target
encoder is the representation I-JEPA is evaluated on (the capture's own linear eval
uses the target encoder). `linear_eval` reads this `encoder.pt`; the representation
is the model this port trains, so the probe number is a genuine, comparable linear
probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "29_ijepa"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

MODEL_KEYS = frozenset({"name", "img_size", "patch_size"})
PREDICTOR_KEYS = frozenset({"pred_dim", "pred_depth"})
DATA_KEYS = frozenset({"augmentation", "use_horizontal_flip", "num_workers"})
MASKING_KEYS = frozenset({"num_enc_masks", "num_pred_masks", "allow_overlap",
                          "min_keep", "enc_mask_scale", "enc_mask_aspect",
                          "pred_mask_scale", "pred_mask_aspect"})
TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "start_lr", "final_lr",
                           "weight_decay", "final_wd", "warmup_epochs",
                           "clip_grad", "ipe_scale", "beta1", "beta2"})
EMA_KEYS = frozenset({"start_ema", "final_ema"})
PRETRAIN_TRAIN_KEYS = (MODEL_KEYS | PREDICTOR_KEYS | DATA_KEYS | MASKING_KEYS
                    | TRAINING_KEYS | EMA_KEYS)
# The additive unified ViT-B/16 Step-2 recipe (recipe: unified, a key in `train`;
# absent == the native step-1 path). The native I-JEPA trainer already implements
# the unified recipe -- it uses the lr directly (no batch/256 rescale), reads the
# arch and augmentation from the config, and its cosine weight-decay is constant
# when weight_decay == final_wd -- so the only additions are augmentation: step2
# (which ignores use_horizontal_flip, so that native-only key is dropped) and
# milestone checkpoints (save_at_epochs). The two key sets are disjoint, so a knob
# from one recipe on the other is refused by name.
RECIPES = ("native", "unified")
PRETRAIN_UNIFIED_KEYS = (PRETRAIN_TRAIN_KEYS - {"use_horizontal_flip"}
                         ) | {"save_at_epochs"}
EVAL_MODEL_KEYS = frozenset({"name", "img_size", "patch_size"})
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = EVAL_MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# The target ViT encoder. The context encoder, predictor and mask token are
# excluded; the prefix is stripped so encoder.pt loads into a plain ViT.
ENCODER_PREFIX = "target_encoder."

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


def _pair(value) -> list:
    return [float(s) for s in value]


def _model_section(train: dict) -> dict:
    return {"name": str(train["name"]), "img_size": int(train["img_size"]),
            "patch_size": int(train["patch_size"])}


def _predictor_section(train: dict) -> dict:
    return {"pred_dim": int(train["pred_dim"]),
            "pred_depth": int(train["pred_depth"])}


def _masking_section(train: dict) -> dict:
    return {"num_enc_masks": int(train["num_enc_masks"]),
            "num_pred_masks": int(train["num_pred_masks"]),
            "allow_overlap": bool(train["allow_overlap"]),
            "min_keep": int(train["min_keep"]),
            "enc_mask_scale": _pair(train["enc_mask_scale"]),
            "enc_mask_aspect": _pair(train["enc_mask_aspect"]),
            "pred_mask_scale": _pair(train["pred_mask_scale"]),
            "pred_mask_aspect": _pair(train["pred_mask_aspect"])}


def _training_section(train: dict) -> dict:
    return {"epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "lr": float(train["lr"]),
            "start_lr": float(train["start_lr"]),
            "final_lr": float(train["final_lr"]),
            "weight_decay": float(train["weight_decay"]),
            "final_wd": float(train["final_wd"]),
            "warmup_epochs": int(train["warmup_epochs"]),
            "clip_grad": float(train["clip_grad"]),
            "ipe_scale": float(train["ipe_scale"]),
            "beta1": float(train["beta1"]),
            "beta2": float(train["beta2"])}


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

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        _named(EVAL_TRAIN_KEYS - set(train), set(train) - EVAL_TRAIN_KEYS,
               "config.train")
        return {"stage": stage}

    # `recipe` selects the pretrain recipe; absent == native. It is a key in
    # `train`, consumed here and not passed on, so it is excluded from the check.
    recipe = train.get("recipe", "native")
    if recipe not in RECIPES:
        raise ConfigError(
            f"config.train: recipe is {recipe!r}; expected one of "
            f"{', '.join(RECIPES)}")
    keys = PRETRAIN_UNIFIED_KEYS if recipe == "unified" else PRETRAIN_TRAIN_KEYS
    rest = {k: v for k, v in train.items() if k != "recipe"}
    _named(keys - set(rest), set(rest) - keys, "config.train")

    # step-2 augmentation ignores use_horizontal_flip, so it is not a unified key;
    # the trainer still reads it, so a fixed False is passed on.
    use_flip = bool(train["use_horizontal_flip"]) if recipe != "unified" else False
    training = _training_section(train)
    if recipe == "unified":
        training["save_at_epochs"] = [int(e) for e in train["save_at_epochs"]]

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "predictor": _predictor_section(train),
        "data": {"data_root": str(config["data_root"]),
                 "augmentation": str(train["augmentation"]),
                 "use_horizontal_flip": use_flip,
                 "num_workers": int(train["num_workers"])},
        "masking": _masking_section(train),
        "training": training,
        "ema": {"start_ema": float(train["start_ema"]),
                "final_ema": float(train["final_ema"])},
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
            f"nothing under {ENCODER_PREFIX!r} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_ijepa_encoder
    train = config["train"]
    model = build_ijepa_encoder(str(train["name"]), img_size=int(train["img_size"]),
                                patch_size=int(train["patch_size"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this ViT encoder does not have: "
            f"{unexpected[:5]}")
    if missing:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {missing[:5]}. It should "
            "be the full target ViT encoder; the predictor is excluded")
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
        from train_pretrain_ijepa import run as _run
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
        from evaluate_linear_ijepa import run as _run
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
    if ctx.config.get("train", {}).get("recipe") == "unified":
        # The trainer writes checkpoint_epoch_{N}.pth at each save_at_epochs; hand
        # over encoder_epoch{N}.pt (the target encoder) for each so the 100/200/300
        # sweep can probe every frozen milestone.
        work = Path(ctx.out) / WORK
        for n in ctx.config["train"].get("save_at_epochs", []):
            ck = work / f"checkpoint_epoch_{int(n)}.pth"
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
    return ("this stage fits a linear probe on the frozen I-JEPA target ViT "
            "encoder and produces a classifier, not an encoder; it reads the "
            "encoder.pt named in the config")


def main(argv: "list[str] | None" = None) -> int:
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
