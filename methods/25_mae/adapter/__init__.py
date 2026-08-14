"""Adapter for 25_mae, step 1 and linear evaluation (He et al., 2021).

    python -m adapter --config <resolved.json> --out <dir>

MAE pretrains a masked autoencoder (step 1) and is evaluated by a linear probe on
the encoder's features (linear_eval). It is a **self-contained re-implementation**
-- the lab's own MAE code, torch-only, trained from scratch (no CC-BY-NC weights)
-- so there is no `third_party/` submodule. The capture's unified ViT-B/16 Step 2
(recipe: unified) is also ported additively: the same MAE objective on a ViT-B/16
encoder (vs the native ViT-L/16) under the unified recipe -- AdamW + a cosine LR
schedule with 10-epoch warmup (which the native fixed-LR trainer lacks) and
milestone checkpoints; the native ViT-L/16 path is byte-for-byte unchanged.

`encoder.pt` is the encoder side of the model: the patch embedding, the CLS
token, the encoder blocks and their norm. The decoder (`dec_*`, `mask_token`,
`dec_pred`) is reconstruction machinery and is excluded. `linear_eval` reads this
`encoder.pt`; the representation is the model this port trains, so the probe
number is a genuine, comparable linear probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "25_mae"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

# Architecture settings (build the model) ...
MODEL_KEYS = frozenset({"arch", "img_size", "patch_size", "enc_embed_dim",
                        "enc_depth", "enc_num_heads", "dec_embed_dim",
                        "dec_depth", "dec_num_heads", "mlp_ratio", "mask_ratio",
                        "norm_pix_loss"})
PRETRAIN_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                              "weight_decay"})
PRETRAIN_TRAIN_KEYS = MODEL_KEYS | PRETRAIN_TRAIN_ONLY
# The unified ViT-B/16 Step-2 recipe (recipe: unified), additive to the native
# ViT-L/16 recipe. The model arch is a shared key (vit_large natively, vit_base
# for unified). The native trainer has no LR schedule; the unified one ADDS a
# cosine schedule, so the unified-only training keys are warmup_epochs, min_lr and
# save_at_epochs. `recipe` is stripped before key validation.
RECIPES = ("native", "unified")
UNIFIED_TRAIN_ONLY = PRETRAIN_TRAIN_ONLY | {"warmup_epochs", "min_lr",
                                            "save_at_epochs"}
PRETRAIN_UNIFIED_KEYS = MODEL_KEYS | UNIFIED_TRAIN_ONLY
# The probe reads the architecture (to rebuild the model), the pooling, and its
# own hyperparameters.
EVAL_PROBE_KEYS = frozenset({"pool", "epochs", "batch_size", "num_workers",
                             "lr", "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
# linear_eval also names the encoder.pt to probe (from a pretrain run).
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# The encoder side: patch embed, CLS token, encoder blocks and norm. The decoder
# is excluded. (enc_pos_embed is a non-persistent sin-cos buffer, recomputed on
# build, so it is not in the state dict.)
ENCODER_PREFIXES = ("patch_embed.", "cls_token", "enc_blocks.", "enc_norm.")

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
    out = {"arch": str(train["arch"])}
    for k in ("img_size", "patch_size", "enc_embed_dim", "enc_depth",
              "enc_num_heads", "dec_embed_dim", "dec_depth", "dec_num_heads"):
        out[k] = int(train[k])
    out["mlp_ratio"] = float(train["mlp_ratio"])
    out["mask_ratio"] = float(train["mask_ratio"])
    out["norm_pix_loss"] = bool(train["norm_pix_loss"])
    return out


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
    # `recipe` selects the recipe; absent == native. Stripped before validation.
    recipe = train.get("recipe", "native")
    if recipe not in RECIPES:
        raise ConfigError(
            f"config.train: recipe is {recipe!r}; expected one of "
            f"{', '.join(RECIPES)}")
    rest = {k: v for k, v in train.items() if k != "recipe"}
    if stage == "linear_eval":
        keys = EVAL_TRAIN_KEYS
    else:
        keys = PRETRAIN_UNIFIED_KEYS if recipe == "unified" else PRETRAIN_TRAIN_KEYS
    _named(keys - set(rest), set(rest) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        return {"stage": stage}

    training = {"epochs": int(train["epochs"]),
                "batch_size": int(train["batch_size"]),
                "num_workers": int(train["num_workers"]),
                "lr": float(train["lr"]),
                "weight_decay": float(train["weight_decay"])}
    if recipe == "unified":
        training["warmup_epochs"] = int(train["warmup_epochs"])
        training["min_lr"] = float(train["min_lr"])
        training["save_at_epochs"] = [int(e) for e in train["save_at_epochs"]]

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "training": training,
        "data": {"data_root": str(config["data_root"])},
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
    """Rebuild the MAE and put the encoder weights back. The decoder is expected
    to be missing; the encoder is not."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_mae
    from train_pretrain_mae import model_kwargs
    train = config["train"]
    model = build_mae(train["arch"], **model_kwargs(train))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The decoder "
            "is expected to be missing; the encoder is not")
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
    recipe = config.get("train", {}).get("recipe", "native")
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        if recipe == "unified":
            from train_pretrain_vit_mae import run as _run
        else:
            from train_pretrain_mae import run as _run
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
    """Probe the trained encoder. Reads encoder.pt (the MAE encoder) named in
    the config."""
    import torch
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_mae import run as _run
    state = torch.load(config["encoder"], map_location="cpu", weights_only=True)
    mae = load_encoder(state, config)
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config, mae=mae) or {}
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
    if train.get("recipe") == "unified":
        # The unified trainer writes checkpoint_epoch_{N}.pth per milestone; hand
        # over encoder_epoch{N}.pt for each so the 100/200/300 sweep can probe
        # each frozen encoder.
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
    return ("this stage fits a linear probe on the frozen MAE encoder and "
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
