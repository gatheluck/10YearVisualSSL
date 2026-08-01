"""Adapter for 20_simsiam, step 1 (Chen & He, 2020).

    python -m adapter --config <resolved.json> --out <dir>

**What this port had to solve that the earlier two did not is which module is
the encoder.** SimSiam trains three: a ResNet-50 backbone, a projector, and a
predictor. Only the backbone is the representation -- and that is read from
the original rather than decided here. `SimSiamResNet.get_encoder()` returns
`self.backbone`, and the original's own `evaluate_linear_official.py` builds
its frozen feature extractor from exactly that call. The projector and the
predictor are training machinery.

The second new thing is a metric with nowhere to go. The trainer reports
`z_std`, the standard deviation of the L2-normalised embeddings, which is
SimSiam's collapse monitor: healthy runs sit near `1/sqrt(dim)` and a
collapsed one near zero. It is a real measurement and it belongs to neither
family in the contract's vocabulary, so it is mapped to `None` -- kept under
the original's own name in `metrics_raw`, kept out of the comparable block.
Inventing a contract name would offer it for comparison against methods that
have no such quantity.

The checkpoint directory arriving inside the config is the same problem the
second port met, and gets the same answer: it is **refused** rather than
overridden, because a config naming a directory that was not used is a config
that lies about the run.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "20_simsiam"
STAGES = ("step1",)

# Every setting the original reads, and no others.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "base_lr",
                        "momentum", "weight_decay", "img_size", "dim",
                        "pred_dim", "save_freq", "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

# The original writes checkpoints, a copy of its config and TensorBoard events
# under `checkpoint.save_dir`. It gets a subdirectory of --out so that nothing
# escapes and every file still reaches the manifest.
WORK = "work"

# `get_encoder()` returns `self.backbone`, and the original's own linear
# evaluation loads exactly that. A checkpoint written under DDP carries a
# `module.` prefix, which the original's loader strips before loading.
ENCODER_PREFIX = "backbone."
DDP_PREFIX = "module."

# What the original calls its numbers, and what the contract calls them.
#
# `final_loss` is SimSiam's negative cosine similarity -- this method's own
# objective, sharing no scale with any other method's loss, so it is a pretext
# name. `final_z_std` is the collapse monitor and has no contract slot at all:
# `None` keeps it in `metrics_raw` and out of `metrics`, which is the whole
# point of that path.
STEP1_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "final_z_std": None,
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
    """Translate the resolved config into the shape the original expects.

    The original reads a nested mapping with `model`, `data`, `training` and
    `checkpoint` sections. The contract's config is flat and declares only
    what affects the result, so the two are not the same document and this is
    where they meet.
    """
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

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    return {
        "model": {"dim": int(train["dim"]),
                  "pred_dim": int(train["pred_dim"])},
        "data": {"train_path": str(Path(config["data_root"]) / "train"),
                 "img_size": int(train["img_size"]),
                 "num_workers": int(train["num_workers"])},
        # base_lr and batch_size are passed through rather than combined: the
        # linear scaling rule init_lr = base_lr * batch_size / 256 is the
        # original's, and recomputing it here would be a second implementation
        # of a rule that already exists.
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "base_lr": float(train["base_lr"]),
                     "momentum": float(train["momentum"]),
                     "weight_decay": float(train["weight_decay"]),
                     "print_freq": int(train["print_freq"]),
                     "save_freq": int(train["save_freq"])},
        "checkpoint": {"save_dir": str(Path(out) / WORK),
                       "allow_resume": False},
        "seed": int(config["seed"]),
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The backbone, and only the backbone.

    Read from the original: `get_encoder()` returns `self.backbone`, and its
    own linear evaluation loads that. Shipping the projector and predictor
    would change what `encoder.pt` means from one method to the next.
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

    The keys keep the `backbone.` prefix they had in the checkpoint, so they
    load into the whole model rather than into the submodule; `get_encoder()`
    then hands back the backbone the original's own linear evaluation uses.
    Whether a port strips its prefix is its own business -- what has to hold
    is that the two halves agree, which is what this makes checkable.

    **The encoder is not self-describing.** Its shapes come from the settings
    the run used, so rebuilding the model with library defaults produces a
    differently shaped one and `load_state_dict` reports a wall of size
    mismatches. The resolved config is therefore required, not optional --
    found by writing the round-trip test, which failed on exactly that.
    """
    from models import build_simsiam_resnet
    train = config["train"]
    model = build_simsiam_resnet(dim=int(train["dim"]),
                                 pred_dim=int(train["pred_dim"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIX)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing backbone weights: {absent[:5]}. The "
            "projector and predictor are expected to be missing; the backbone "
            "is not")
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
    """The checkpoint the run finished on.

    The original names them `checkpoint_epoch_<n>.pth` and writes one every
    `save_freq` epochs and one at the end, so the highest epoch is the final
    state. Sorting by name would put epoch 9 after epoch 10.
    """
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
