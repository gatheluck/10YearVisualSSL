"""Adapter for 21_barlow_twins, step 1 (Zbontar et al., 2021).

    python -m adapter --config <resolved.json> --out <dir>

**What was new here is mixed precision.**

The captured trainer offers three precisions -- fp32, bf16 and fp16 -- and
writes `device_type="cuda"` into its autocast and its `GradScaler`. On a CPU,
fp32 and bf16 exist and fp16 does not. Quietly running fp32 when fp16 was
asked for would report a run at a precision it never used, so the pair is
**refused by name**, the same way asking for a GPU that is not there is.

This method's augmentation also calls `random.random()` directly rather than
going through a torchvision transform, which the port before this one did
not, so `make_deterministic` has to seed `random` and not only torch.

**Its loader workers need nothing extra, and a first version of this port
wrongly said they did.** Torch's worker loop seeds `random` itself -- measured
after the fact, both by reading `_worker_loop` and by drawing from two runs
with no `worker_init_fn`. The `seed_worker` that had been added, and the change
it forced on the captured loader, were removed. The loader came across
untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "21_barlow_twins"
STAGES = ("step1",)

# Every setting the original reads, and no others.
# Read from the trainer, not guessed: it takes **two** learning rates, as
# LARS does in the paper, and `base_lr` is not one of them. A first attempt
# declared `base_lr` and the run failed on a key nothing reads.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr_weights",
                        "lr_biases", "weight_decay", "img_size", "projector",
                        "lambd", "warmup_epochs", "precision", "save_freq",
                        "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

# Three offered; one of them needs hardware the others do not.
PRECISIONS = ("fp32", "bf16", "amp_fp16")
GPU_ONLY_PRECISIONS = ("amp_fp16",)

# The original writes checkpoints, a copy of its config and TensorBoard events
# under `checkpoint.save_dir`. It gets a subdirectory of --out.
WORK = "work"

# `get_encoder()` returns `self.backbone`. A checkpoint written under DDP
# carries a `module.` prefix.
ENCODER_PREFIX = "backbone."
DDP_PREFIX = "module."

# What the original calls its numbers, and what the contract calls them. The
# redundancy-reduction objective is this method's own, so it is a pretext name
# and shares no scale with any other method's loss.
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

    device = config["device"]
    if device not in DEVICES:
        raise ConfigError(
            f"config: device is {device!r}; expected one of "
            f"{', '.join(DEVICES)}")

    precision = train["precision"]
    if precision not in PRECISIONS:
        raise ConfigError(
            f"config.train: precision is {precision!r}; expected one of "
            f"{', '.join(PRECISIONS)}")
    if precision in GPU_ONLY_PRECISIONS and device == "cpu":
        raise ConfigError(
            f"config: precision {precision!r} needs a GPU and device is "
            "'cpu'. Running fp32 instead would report a run at a precision "
            "it did not use; ask for bf16, which a cpu does have")

    return {
        "model": {"projector": str(train["projector"])},
        "data": {"train_path": str(Path(config["data_root"]) / "train"),
                 "img_size": int(train["img_size"]),
                 "num_workers": int(train["num_workers"])},
        # `lambd` lives in its own section, which is where the trainer looks
        # for it.
        "barlow": {"lambd": float(train["lambd"])},
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "lr_weights": float(train["lr_weights"]),
                     "lr_biases": float(train["lr_biases"]),
                     "weight_decay": float(train["weight_decay"]),
                     "warmup_epochs": int(train["warmup_epochs"]),
                     "precision": precision,
                     "print_freq": int(train["print_freq"]),
                     "save_freq": int(train["save_freq"])},
        "checkpoint": {"save_dir": str(Path(out) / WORK),
                       "allow_resume": False},
        "seed": int(config["seed"]),
    }


def to_args(config: dict, out: Path) -> Namespace:
    """Every argument the trainer reads, taken from its source rather than
    discovered one failure at a time.

    `end_epoch` is the captured pilot switch: it stops a run short of the
    configured epochs. The contract's config says how many epochs ran, so it
    is left unset here -- a second way to change the length of a run would be
    a setting outside the hash.
    """
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     end_epoch=None, device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The backbone, and only the backbone.

    Read from the original: `get_encoder()` returns `self.backbone`. The
    projector is training machinery.
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

    The keys keep the prefix they had, so they load into the whole model and
    `get_encoder()` hands back the backbone. The resolved config is required
    because the encoder is not self-describing: its shapes come from the
    settings the run used.
    """
    from models import build_barlow_resnet
    model = build_barlow_resnet(projector=str(config["train"]["projector"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIX)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing backbone weights: {absent[:5]}. The "
            "projector is expected to be missing; the backbone is not")
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
