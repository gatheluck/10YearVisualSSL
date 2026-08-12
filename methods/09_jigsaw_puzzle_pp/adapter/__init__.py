"""Adapter for 09_jigsaw_puzzle_pp: step 1, knowledge transfer, and linear
evaluation (Noroozi et al., CVPR 2018).

    python -m adapter --config <resolved.json> --out <dir>

Jigsaw++ solves a pretext of predicting which permutation reordered an image's
3x3 tiles -- with occlusion tiles from another image and frequent grayscale --
using a shared VGG16 encoder (step 1). The paper's knowledge-transfer stage is
also ported (knowledge_transfer): cluster the VGG16 conv4 features with faiss
k-means into pseudo-labels and train a standard AlexNet to classify them. Either
encoder is then probed with a linear layer (linear_eval). A self-contained
re-implementation (the lab's own code) -- no submodule. The clustering uses faiss
(GPU / x86_64-linux only), so knowledge_transfer runs there; step 1 and the
default `arch=vgg16` probe stay torch-only. The capture's step 2 (ViT) is
excluded, as in every port (see the method README).

`encoder.pt` is the shared VGG16 encoder (`encoder.*`) for step 1, or the AlexNet
conv trunk (`features.*`) for knowledge_transfer; the classification head is
training machinery and is left out. `linear_eval` reads an `encoder.pt`: it
probes the VGG16 encoder (`arch=vgg16`, the default) or the knowledge-transfer
AlexNet (`arch=alexnet_cluster_cls`). The representation is a model this port
trains, so the probe number is a genuine, comparable linear probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "09_jigsaw_puzzle_pp"
STAGES = ("pretrain", "knowledge_transfer", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

MODEL_KEYS = frozenset({"num_permutations", "dropout", "tile_size", "tile_gap",
                        "image_size", "grayscale_prob", "max_occlusions"})
PRETRAIN_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                              "momentum", "weight_decay"})
PRETRAIN_TRAIN_KEYS = MODEL_KEYS | PRETRAIN_TRAIN_ONLY
# Knowledge transfer (Noroozi et al. "Boosting SSL via Knowledge Transfer"):
# reads a VGG16 encoder.pt, clusters its conv4 features (faiss) into pseudo-labels
# and trains an AlexNet on them. Only `dropout` is needed to rebuild the VGG16 to
# load its encoder (only encoder.* is loaded); the rest configures the AlexNet.
KT_KEYS = frozenset({"num_clusters", "image_size", "dropout"})
KT_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                           "momentum", "weight_decay"})
KT_TRAIN_KEYS = KT_KEYS | KT_TRAIN_ONLY
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
# linear_eval probes either the VGG16 encoder (arch=vgg16, the default -- keys
# unchanged) or the knowledge-transfer AlexNet (arch=alexnet_cluster_cls).
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS                       # arch=vgg16
EVAL_ALEXNET_KEYS = frozenset({"arch", "dropout", "image_size"}) | EVAL_PROBE_KEYS
ARCHS = ("vgg16", "alexnet_cluster_cls")

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
KT_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# encoder.pt prefixes: the shared VGG16 encoder for step1/linear_eval, and the
# AlexNet conv trunk for the knowledge-transfer output.
ENCODER_PREFIXES = ("encoder.",)
ALEXNET_ENCODER_PREFIXES = ("features.",)

PRETRAIN_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "final_acc": "final_pretext_top1_accuracy",
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
    return {
        "num_permutations": int(train["num_permutations"]),
        "dropout": float(train["dropout"]),
        "tile_size": int(train["tile_size"]),
        "tile_gap": int(train["tile_gap"]),
        "image_size": int(train["image_size"]),
        "grayscale_prob": float(train["grayscale_prob"]),
        "max_occlusions": int(train["max_occlusions"]),
    }


def eval_arch(train: dict) -> str:
    """The encoder the probe reads: `vgg16` (default) or `alexnet_cluster_cls`."""
    arch = train.get("arch", "vgg16") if isinstance(train, dict) else "vgg16"
    if arch not in ARCHS:
        raise ConfigError(
            f"config.train: arch is {arch!r}; expected one of "
            f"{', '.join(ARCHS)}")
    return arch


def _training_section(train: dict) -> dict:
    return {"epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "num_workers": int(train["num_workers"]),
            "lr": float(train["lr"]),
            "momentum": float(train["momentum"]),
            "weight_decay": float(train["weight_decay"])}


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

    if stage == "linear_eval":
        top = EVAL_TOP_KEYS
        arch = eval_arch(config.get("train", {}))
        keys = EVAL_ALEXNET_KEYS if arch == "alexnet_cluster_cls" \
            else EVAL_TRAIN_KEYS
    elif stage == "knowledge_transfer":
        top, keys = KT_TOP_KEYS, KT_TRAIN_KEYS
    else:
        top, keys = TOP_KEYS, PRETRAIN_TRAIN_KEYS
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

    if stage == "knowledge_transfer":
        return {
            "seed": int(config["seed"]),
            "kt": {"num_clusters": int(train["num_clusters"]),
                   "image_size": int(train["image_size"]),
                   "dropout": float(train["dropout"])},
            "training": _training_section(train),
            "data": {"data_root": str(config["data_root"])},
            "output": {"checkpoint_dir": str(Path(out) / WORK)},
        }

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "training": _training_section(train),
        "data": {"data_root": str(config["data_root"])},
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict, prefixes=ENCODER_PREFIXES) -> dict:
    out = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    if not out:
        raise RuntimeError(
            f"nothing under {prefixes} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """Rebuild the encoder encoder.pt describes and load it. The VGG16 encoder
    (arch=vgg16, the default) or the knowledge-transfer AlexNet
    (arch=alexnet_cluster_cls)."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import (build_vgg16_jigsaw_pp_model,
                        build_alexnet_cluster_cls_model)
    from train_pretrain_jigsaw_pp import model_kwargs
    arch = eval_arch(config["train"])
    if arch == "alexnet_cluster_cls":
        model = build_alexnet_cluster_cls_model(
            dropout=float(config["train"]["dropout"]))
        prefixes = ALEXNET_ENCODER_PREFIXES
    else:
        model = build_vgg16_jigsaw_pp_model(**model_kwargs(config["train"]))
        prefixes = ENCODER_PREFIXES
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(prefixes)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The "
            "classifier/head is expected to be missing; the encoder is not")
    return model


def _load_vgg_encoder(encoder_path, dropout):
    """Load a step-1 VGG16 encoder.pt into a VGG16 model, for the KT stage."""
    import torch
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_vgg16_jigsaw_pp_model
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model = build_vgg16_jigsaw_pp_model(num_classes=701, dropout=float(dropout))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"the VGG16 encoder.pt carries unexpected keys: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"the VGG16 encoder.pt is missing encoder weights: {absent[:5]}")
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
        from train_pretrain_jigsaw_pp import run as _run
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
        from evaluate_linear_jigsaw_pp import run as _run
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


def run_knowledge_transfer(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from train_pretrain_cluster_cls import run as _run
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True,
                                                       exist_ok=True)
    vgg_model = _load_vgg_encoder(config["encoder"], config["train"]["dropout"])
    args = Namespace(config=None, data_path=None, encoder=None,
                     device=config["device"])
    raw = _run(args, run_config, vgg_model=vgg_model) or {}
    metrics, unusable = _filter_numeric(raw)
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def body(ctx: adapterlib.Context) -> None:
    import torch
    if ctx.stage == "linear_eval":
        ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                          names=LINEAR_EVAL_METRIC_NAMES)
        return
    if ctx.stage == "knowledge_transfer":
        metrics = run_knowledge_transfer(ctx.config, ctx.out)
        prefixes = ALEXNET_ENCODER_PREFIXES
    else:
        metrics = run_training(ctx.config, ctx.out)
        prefixes = ENCODER_PREFIXES
    latest = Path(ctx.out) / WORK / "checkpoint_latest.pth"
    if not latest.is_file():
        raise RuntimeError(
            f"training finished but {latest} was not written; there is no "
            "encoder to hand over")
    state = torch.load(latest, map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model_state_dict"], prefixes),
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
    return ("this stage fits a linear probe on the frozen encoder (VGG16 or the "
            "knowledge-transfer AlexNet) and produces a classifier, not an "
            "encoder; it reads the encoder.pt named in the config")


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
