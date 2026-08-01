"""Adapter for 17_swav, step 1 (Caron et al., 2020).

    python -m adapter --config <resolved.json> --out <dir>

Instead of comparing two views directly, SwAV assigns each view to a set of
learned prototypes and makes one view predict the other's assignment. A
Sinkhorn-Knopp normalisation keeps the prototypes evenly used, which is what
stops the representation collapsing.

**Two things here are about the shape of a configuration rather than about the
training.**

The multi-crop settings are four parallel lists -- the crop sizes, how many of
each, and the scale bounds -- and the loader asserts they are the same length.
A set of lists that do not line up is not a run, so it is refused here, by
name, rather than arriving as a bare `AssertionError` from inside the dataset.

And most of this original's settings are optional: it reads a dozen keys with
`cfg.get(...)` and a default behind them. Leaving them out of the contract's
config would describe a run whose Sinkhorn epsilon, warmup and prototype
freezing were whatever that version of the code defaulted to. They are all
declared, because the resolved config has to say what ran.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "17_swav"
STAGES = ("step1",)

# The four lists that describe multi-crop. They are read together because they
# only mean anything together.
CROP_KEYS = ("size_crops", "nmb_crops", "min_scale_crops", "max_scale_crops")

# Every setting the original reads, and no others. Most of them are optional
# in the original; none of them is optional here.
TRAIN_KEYS = frozenset({
    "epochs", "batch_size", "num_workers", "lr", "final_lr", "momentum",
    "weight_decay", "warmup_epochs", "start_warmup", "eta", "larc_clip",
    "freeze_prototypes_steps", "color_jitter_strength", "temperature",
    "sinkhorn_eps", "sinkhorn_iters", "out_dim", "hidden_mlp",
    "nmb_prototypes", "save_freq", "print_freq"}) | set(CROP_KEYS)
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

WORK = "work"

# `get_encoder()` wraps `self.encoder` with a flatten; the weights live under
# that prefix. The projection head and the prototypes are training machinery:
# the prototypes are learned cluster centres, not the representation.
ENCODER_PREFIX = "encoder."
DDP_PREFIX = "module."

STEP1_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
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


def check_crops(train: dict) -> None:
    """The four multi-crop lists, checked together.

    The loader asserts their lengths match, which arrives as a bare
    `AssertionError` from inside the dataset with nothing to say which setting
    was wrong. Refused here instead, by name.
    """
    for key in CROP_KEYS:
        if not isinstance(train[key], list):
            raise ConfigError(
                f"config.train: {key} is {type(train[key]).__name__}, not a "
                "list. The multi-crop settings are one list per crop size")
    lengths = {key: len(train[key]) for key in CROP_KEYS}
    if len(set(lengths.values())) != 1:
        raise ConfigError(
            "config.train: the multi-crop settings must be the same length, "
            "one entry per crop size; got "
            + ", ".join(f"{k}={v}" for k, v in lengths.items()))
    if not any(lengths.values()):
        raise ConfigError(
            "config.train: the multi-crop settings are empty, so there is at "
            "least one crop size missing and the loader would yield nothing")


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the shape the original expects."""
    for key in ("checkpoint", "output"):
        if key in config:
            raise ConfigError(
                f"config: {key} is set. The output location is not a setting: "
                "the contract fixes it at --out, and a config naming a "
                "directory would claim a location that was not used")
    _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")
    if config["stage"] not in STAGES:
        raise ConfigError(
            f"config: stage is {config['stage']!r}; known stages are "
            f"{', '.join(STAGES)}")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    _named(TRAIN_KEYS - set(train), set(train) - TRAIN_KEYS, "config.train")
    check_crops(train)

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    return {
        "model": {"out_dim": int(train["out_dim"]),
                  "hidden_mlp": int(train["hidden_mlp"]),
                  "nmb_prototypes": int(train["nmb_prototypes"])},
        "data": {"train_path": str(Path(config["data_root"]) / "train"),
                 "num_workers": int(train["num_workers"]),
                 "color_jitter_strength": float(
                     train["color_jitter_strength"]),
                 **{k: list(train[k]) for k in CROP_KEYS}},
        # The trainer looks for these under `loss`, not under `training`.
        "loss": {"temperature": float(train["temperature"]),
                 "sinkhorn_eps": float(train["sinkhorn_eps"]),
                 "sinkhorn_iters": int(train["sinkhorn_iters"])},
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "lr": float(train["lr"]),
                     "final_lr": float(train["final_lr"]),
                     "momentum": float(train["momentum"]),
                     "weight_decay": float(train["weight_decay"]),
                     "warmup_epochs": int(train["warmup_epochs"]),
                     "start_warmup": float(train["start_warmup"]),
                     "eta": float(train["eta"]),
                     "larc_clip": bool(train["larc_clip"]),
                     "freeze_prototypes_steps": int(
                         train["freeze_prototypes_steps"]),
                     "print_freq": int(train["print_freq"]),
                     "save_freq": int(train["save_freq"])},
        "checkpoint": {"save_dir": str(Path(out) / WORK)},
        "seed": int(config["seed"]),
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The encoder, and neither the projection head nor the prototypes.

    The prototypes are learned cluster centres -- training machinery, not the
    representation. Shipping them would change what `encoder.pt` means from
    one method to the next.
    """
    out = {}
    for key, value in state_dict.items():
        name = key[len(DDP_PREFIX):] if key.startswith(DDP_PREFIX) else key
        if name.startswith(ENCODER_PREFIX):
            out[name] = value
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIX!r} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """The other half of `extract_encoder`: put it back.

    The resolved config is required because the encoder is not
    self-describing: the projection head and the prototype count size the
    model it belongs to.
    """
    from models import build_resnet_swav
    train = config["train"]
    model = build_resnet_swav(
        out_dim=int(train["out_dim"]), hidden_mlp=int(train["hidden_mlp"]),
        nmb_prototypes=int(train["nmb_prototypes"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIX)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent[:5]}. The "
            "projection head and the prototypes are expected to be missing; "
            "the encoder is not")
    return model.get_encoder()


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        from train_step1_resnet import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                     exist_ok=True)
    raw = _run(args, run_config) or {}
    metrics, unusable = {}, 0
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            unusable += 1
            continue
        metrics[key] = value
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def latest_checkpoint(work: Path) -> Path:
    """The checkpoint the run finished on, chosen by epoch rather than by
    name: sorting would put epoch 9 after epoch 10."""
    found = list(Path(work).glob("checkpoint_epoch_*.pth"))
    if not found:
        raise RuntimeError(
            f"training finished but no checkpoint_epoch_*.pth is in {work}; "
            "there is no encoder to hand over")
    return max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def body(ctx: adapterlib.Context) -> None:
    import torch
    metrics = run_training(ctx.config, ctx.out)
    state = torch.load(latest_checkpoint(Path(ctx.out) / WORK),
                       map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["state_dict"]),
               Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=STEP1_METRIC_NAMES)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage="step1", body=body)
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
