"""Adapter for 2_vae, step 1 (Kingma & Welling, 2013).

    python -m adapter --config <resolved.json> --out <dir>

**The new problem this port had to solve is where the output goes.** The
captured config carries an absolute path on the cluster as
`output.checkpoint_dir`, and the original writes its checkpoints and its
TensorBoard events there. Two things are wrong with that here: the contract
says an adapter writes only under `--out`, and a machine's path in a published
config is reproducible nowhere else.

So the shipped config has no output path at all, and this builds one under
`--out`. A config that tries to set it is **refused** rather than overridden
quietly -- overriding would leave a config claiming an output location that
was not used.

Everything else follows the first port: the original's own training loop is
called, not reimplemented, and every setting is declared so that a key the
stage never reads cannot sit in a config claiming to have had an effect.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "2_vae"
STAGES = ("step1",)

# Every setting the original reads, and no others.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr", "beta",
                        "latent_dim", "hidden_dim", "img_size", "save_freq",
                        "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

# The original writes checkpoints and TensorBoard events here. It gets a
# subdirectory of --out so that nothing escapes and everything is listed.
WORK = "work"
ENCODER_PREFIXES = ("encoder.", "fc_mu.", "fc_logvar.")

# What the original calls its numbers, and what the contract calls them.
# `final_loss` is this method's own objective -- reconstruction plus beta
# times the KL divergence -- so it is a pretext number and shares no scale
# with any other method's loss (CONTRACT, metric vocabulary).
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
    """Translate the resolved config into the shape the original expects.

    The original reads a nested mapping with `model`, `data`, `training` and
    `output` sections. The contract's config is flat and declares only what
    affects the result, so the two are not the same document and this is
    where they meet.
    """
    if "output" in config:
        raise ConfigError(
            "config: output is set. The output location is not a setting: the "
            "contract fixes it at --out, and a config naming a directory would "
            "claim a location that was not used")
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
        "model": {"latent_dim": int(train["latent_dim"]),
                  "hidden_dim": int(train["hidden_dim"])},
        "data": {"img_size": int(train["img_size"]),
                 "batch_size": int(train["batch_size"]),
                 "num_workers": int(train["num_workers"]),
                 "augmentation_type": "none"},
        "training": {"epochs": int(train["epochs"]),
                     "lr": float(train["lr"]),
                     "min_lr": float(train["lr"]),
                     "weight_decay": 0.0,
                     "beta": float(train["beta"]),
                     "grad_clip": 0.0,
                     "optimizer": "adam",
                     "betas": [0.9, 0.999],
                     "lr_scheduler": "constant",
                     "warmup_epochs": 0,
                     "print_freq": int(train["print_freq"]),
                     "save_freq": int(train["save_freq"])},
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=str(config["data_root"]),
                     distributed=False, resume="", local_rank=0, world_size=1,
                     seed=int(config["seed"]))


def load_encoder(state_dict: dict, config: dict):
    """The other half of `extract_encoder`: put it back.

    **This method's encoder is not one submodule**, so its keys keep their own
    names -- `encoder.`, `fc_mu.` and `fc_logvar.` -- and there is no single
    prefix that could be stripped. The pair still has to agree, which is what
    this makes checkable: the model is built, the saved keys are loaded into
    it, and anything the file failed to provide is an error rather than a
    silently default-initialised weight.

    **The encoder is not self-describing.** Its shapes come from the settings
    the run used, so rebuilding the model with library defaults produces a
    differently shaped one and `load_state_dict` reports a wall of size
    mismatches. The resolved config is therefore required, not optional --
    found by writing the round-trip test, which failed on exactly that.
    """
    from models.vae_cnn import VAE_CNN
    train = config["train"]
    model = VAE_CNN(latent_dim=int(train["latent_dim"]),
                    image_size=int(train["img_size"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent}. The decoder is "
            "expected to be missing; the encoder is not")
    return model


def extract_encoder(state_dict: dict) -> dict:
    """The encoder half of the model.

    A VAE is an encoder and a decoder, and the contract asks for the encoder.
    Handing over the decoder as well would change what `encoder.pt` means from
    one method to the next. The projections to the latent mean and log
    variance belong to the encoder side and come with it.
    """
    out = {k: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIXES)}
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIXES} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        from train_step1_cnn import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True,
                                                       exist_ok=True)
    raw = _run(args, run_config) or {}
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def body(ctx: adapterlib.Context) -> None:
    import torch
    metrics = run_training(ctx.config, ctx.out)
    latest = Path(ctx.out) / WORK / "checkpoint_latest.pth"
    if not latest.is_file():
        raise RuntimeError(
            f"training finished but {latest} was not written; there is no "
            "encoder to hand over")
    state = torch.load(latest, map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model_state_dict"]),
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
