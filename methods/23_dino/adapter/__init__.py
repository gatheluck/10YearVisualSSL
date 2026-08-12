"""Adapter for 23_dino, step 1 and linear evaluation (Caron et al., 2021).

    python -m adapter --config <resolved.json> --out <dir>

DINO (self-distillation with no labels): a student (ViT + DINO head) sees all
crops; a teacher (an EMA copy of the student) sees the two global crops; a
centred + sharpened cross-entropy distils the teacher into the student (step 1).
linear_eval then probes the frozen teacher ViT backbone (its CLS feature,
embed_dim). A self-contained re-implementation (the lab's own code, following
facebookresearch/dino); DINO ships its own ViT, so the port is torch-only -- no
timm. The capture's step 2 (ViT-B) is excluded, as in every port.

`encoder.pt` is the teacher ViT backbone (`teacher.backbone.*`, the prefix
stripped so it loads straight into a plain VisionTransformer): the class token,
the position embedding, the patch-embed conv, the transformer blocks and the final
norm. The DINO head, the centre buffer and the whole student are training
machinery and are left out. The teacher is the representation DINO is known for
(the capture's own linear eval defaults to it). `linear_eval` reads this
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

METHOD = "23_dino"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

MODEL_KEYS = frozenset({"arch", "img_size"})
DINO_KEYS = frozenset({"out_dim", "hidden_dim", "bottleneck_dim",
                       "use_bn_in_head", "norm_last_layer", "n_local_crops",
                       "local_size", "global_crops_scale", "local_crops_scale",
                       "student_temp", "teacher_temp_init", "teacher_temp_final",
                       "teacher_temp_warmup_epochs", "momentum_teacher"})
PRETRAIN_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers",
                              "drop_path_rate", "lr", "min_lr", "warmup_epochs",
                              "weight_decay_start", "weight_decay_end",
                              "clip_grad", "freeze_last_layer"})
PRETRAIN_TRAIN_KEYS = MODEL_KEYS | DINO_KEYS | PRETRAIN_TRAIN_ONLY
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = frozenset({"arch", "img_size"}) | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# The teacher ViT backbone. The head, the centre buffer and the student are
# excluded; the prefix is stripped so encoder.pt loads into a plain ViT.
ENCODER_PREFIX = "teacher.backbone."

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


def _scale_pair(value) -> list:
    pair = [float(s) for s in value]
    if len(pair) != 2:
        raise ConfigError(f"config.train: expected a [min, max] scale, got {value!r}")
    return pair


def _model_section(train: dict) -> dict:
    return {"arch": str(train["arch"]), "img_size": int(train["img_size"])}


def _dino_section(train: dict) -> dict:
    return {"out_dim": int(train["out_dim"]),
            "hidden_dim": int(train["hidden_dim"]),
            "bottleneck_dim": int(train["bottleneck_dim"]),
            "use_bn_in_head": bool(train["use_bn_in_head"]),
            "norm_last_layer": bool(train["norm_last_layer"]),
            "n_local_crops": int(train["n_local_crops"]),
            "local_size": int(train["local_size"]),
            "global_crops_scale": _scale_pair(train["global_crops_scale"]),
            "local_crops_scale": _scale_pair(train["local_crops_scale"]),
            "student_temp": float(train["student_temp"]),
            "teacher_temp_init": float(train["teacher_temp_init"]),
            "teacher_temp_final": float(train["teacher_temp_final"]),
            "teacher_temp_warmup_epochs": int(train["teacher_temp_warmup_epochs"]),
            "momentum_teacher": float(train["momentum_teacher"])}


def _training_section(train: dict) -> dict:
    return {"epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "drop_path_rate": float(train["drop_path_rate"]),
            "lr": float(train["lr"]),
            "min_lr": float(train["min_lr"]),
            "warmup_epochs": int(train["warmup_epochs"]),
            "weight_decay_start": float(train["weight_decay_start"]),
            "weight_decay_end": float(train["weight_decay_end"]),
            "clip_grad": float(train["clip_grad"]),
            "freeze_last_layer": int(train["freeze_last_layer"])}


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
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else PRETRAIN_TRAIN_KEYS
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
        "model": _model_section(train),
        "dino": _dino_section(train),
        "data": {"data_root": str(config["data_root"]),
                 "num_workers": int(train["num_workers"])},
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
            f"nothing under {ENCODER_PREFIX!r} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_vit
    train = config["train"]
    model = build_vit(str(train["arch"]), img_size=int(train["img_size"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this ViT backbone does not have: "
            f"{unexpected[:5]}")
    if missing:
        raise RuntimeError(
            f"encoder.pt is missing backbone weights: {missing[:5]}. It should "
            "be the full teacher ViT trunk")
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
        from train_pretrain_dino import run as _run
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
        from evaluate_linear_dino import run as _run
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
    return ("this stage fits a linear probe on the frozen DINO teacher ViT "
            "backbone and produces a classifier, not an encoder; it reads the "
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
