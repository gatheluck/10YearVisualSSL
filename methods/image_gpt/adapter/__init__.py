"""Adapter for image_gpt, step 1 and linear evaluation (Chen et al., 2020).

    python -m adapter --config <resolved.json> --out <dir>

iGPT pretrains a causal transformer on colour-cluster tokens (step 1) and is
evaluated by a linear probe on a middle transformer layer (linear_eval). It is a
**self-contained re-implementation**, ported from the lab's ARSSL inline model,
the same treatment the six official methods got -- there is no submodule.

`encoder.pt` is the representation side of the model: the token and position
embeddings, the transformer blocks, and the final norm. The generative head
(`head`) is excluded. Unlike the generative ports, iGPT's `linear_eval` reads
this `encoder.pt` -- the representation is the model this port trains, so the
probe number is a genuine, comparable linear probe.

The colour clusters step 1 fits are saved beside `encoder.pt` (`clusters.npy`)
and passed to `linear_eval`, which must quantise images with the same clusters
the model was trained on.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "image_gpt"
STAGES = ("step1", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

# Architecture settings (build the model) ...
MODEL_KEYS = frozenset({"vocab_size", "img_size", "n_layer", "n_head",
                        "n_embd"})
# ... and the step-1 training settings.
STEP1_TRAIN_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                              "grad_clip"})
STEP1_TRAIN_KEYS = MODEL_KEYS | STEP1_TRAIN_ONLY
# The probe reads the architecture (to rebuild the model) plus its own
# hyperparameters.
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
# linear_eval also names the encoder to probe and the clusters to quantise with;
# both come from a step-1 run.
EVAL_TOP_KEYS = TOP_KEYS | {"encoder", "clusters"}

DEVICES = ("auto", "cuda", "cpu")
WORK = "work"
CLUSTERS = "clusters.npy"

# The representation side: token/position embeddings, the blocks and the final
# norm. The generative head is excluded, so encoder.pt is the representation
# network.
ENCODER_PREFIXES = ("token_embed.", "pos_embed.", "blocks.", "ln_f.")

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


def _model_section(train: dict) -> dict:
    return {
        "vocab_size": int(train["vocab_size"]),
        "img_size": int(train["img_size"]),
        "n_layer": int(train["n_layer"]),
        "n_head": int(train["n_head"]),
        "n_embd": int(train["n_embd"]),
    }


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the shape the trainer expects."""
    for key in ("checkpoint", "output"):
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
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else STEP1_TRAIN_KEYS
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
        # The evaluation reads the config directly; it takes no translated
        # run-config document, only a validated stage.
        return {"stage": stage}

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "num_workers": int(train["num_workers"]),
                     "lr": float(train["lr"]),
                     "grad_clip": float(train["grad_clip"])},
        "data": {"data_root": str(config["data_root"])},
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The representation side of the model (everything but the generative
    head)."""
    out = {k: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIXES)}
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIXES} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """The other half of extract_encoder: rebuild the model and put the weights
    back. The architecture comes from the config; the head is expected to be
    missing, the encoder is not."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_igpt
    from train_step1_igpt import model_kwargs
    model = build_igpt(**model_kwargs(config["train"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The head is "
            "expected to be missing; the representation is not")
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
        from train_step1_igpt import run as _run
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
    """Probe the trained model's middle layer. Reads encoder.pt (the model) and
    clusters.npy (the colour space it was trained on), both from a step-1 run."""
    import numpy as np
    import torch
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_igpt import run as _run
    state = torch.load(config["encoder"], map_location="cpu", weights_only=True)
    model = load_encoder(state, config)
    clusters = np.load(config["clusters"])
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config, model=model, clusters=clusters) or {}
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
    work = Path(ctx.out) / WORK
    latest = work / "checkpoint_latest.pth"
    if not latest.is_file():
        raise RuntimeError(
            f"training finished but {latest} was not written; there is no "
            "encoder to hand over")
    state = torch.load(latest, map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model_state_dict"]),
               Path(ctx.out) / "encoder.pt")
    clusters = work / CLUSTERS
    if not clusters.is_file():
        raise RuntimeError(
            f"training finished but {clusters} was not written; the probe "
            "cannot quantise without the clusters the model was trained on")
    shutil.copyfile(clusters, Path(ctx.out) / CLUSTERS)
    ctx.write_metrics(metrics, names=STEP1_METRIC_NAMES)


def _stage_of(config_path) -> str:
    import json
    try:
        return json.loads(Path(config_path).read_text(
            encoding="utf-8")).get("stage") or STAGES[0]
    except (OSError, ValueError, AttributeError):
        return STAGES[0]


def _absent_reason(config_path) -> "str | None":
    """CONTRACT section 3. linear_eval fits a classifier on the frozen model; it
    produces no encoder of its own, and saying so is required."""
    if _stage_of(config_path) != "linear_eval":
        return None
    return ("this stage fits a linear probe on the frozen model and produces a "
            "classifier, not an encoder; it reads the encoder.pt named in the "
            "config")


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
