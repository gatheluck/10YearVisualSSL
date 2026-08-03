"""Adapter for 04_context_encoder, step 1 and linear evaluation (Pathak et al., 2016).

    python -m adapter --config <resolved.json> --out <dir>

Context Encoder learns features by inpainting: a centre hole is cut from the
image and a conv encoder-decoder is trained to fill it, optionally with a
centre-hole adversarial discriminator. The representation the rest of the
project wants is the encoder plus its 4096-d bottleneck; the decoder and the
discriminator are training machinery.

**Only step 1 is here.** The capture's step 1 is the AlexNet architecture; its
step 2 is a ViT variant (with its own two-optimiser, bfloat16, always-adversarial
protocol) that -- like every other method's step 2 -- was not brought across.
Dropping it also drops the `timm` dependency it needed.

`encoder.pt` holds the encoder and the bottleneck (`encoder.*` and `fc.*`); the
decoder (`decoder_fc`, `decoder`) and the discriminator are left out. The
original's own linear evaluation reads the representation as
`model(x) -> (_, features)` -- the bottleneck output -- so `load_encoder`
rebuilds the model and the evaluation runs it that way.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "04_context_encoder"
STAGES = ("step1", "linear_eval")

# Every setting the step-1 trainer reads, and no others.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr", "momentum",
                        "weight_decay", "warmup_epochs", "img_size", "mask_size",
                        "loss_type", "use_adversarial", "adversarial_weight",
                        "save_freq", "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})

# The second stage freezes the encoder and fits a linear head; it reads its own
# small set and needs the encoder to load. The AlexNet architecture is fixed
# (channels=3), so unlike the ViT ports no model-shape key is needed to rebuild.
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
EVAL_TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay", "img_size"})

DEVICES = ("auto", "cuda", "cpu")
LOSS_TYPES = ("l1", "l2", "smooth_l1")

WORK = "work"

# The representation is the encoder conv stack plus the 4096-d bottleneck. The
# decoder and the discriminator are training machinery and are left out of
# encoder.pt. A DDP checkpoint would carry a `module.` prefix.
ENCODER_PREFIXES = ("encoder.", "fc.")
DDP_PREFIX = "module."

# What the trainer calls its numbers, and what the contract calls them. The
# reconstruction and adversarial components are real measurements that belong to
# no family in the vocabulary, so they map to `None`: kept under their own names
# in metrics_raw, kept out of the comparable block.
STEP1_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "final_recon_loss": None,
    "final_adv_loss": None,
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

# The downstream numbers -- all four comparable linear-probe accuracies, because
# this original's evaluation records a best top-5 as well as the rest.
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


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the nested shape the trainer reads."""
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
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else TRAIN_KEYS
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
        # The evaluation takes flags, not a document; `eval_args` builds them.
        return {"stage": stage}

    if train["loss_type"] not in LOSS_TYPES:
        raise ConfigError(
            f"config.train: loss_type is {train['loss_type']!r}; expected one "
            f"of {', '.join(LOSS_TYPES)}")

    return {
        # InpaintingDataset joins root + split itself, so train_path is the
        # parent of train/ and val/, not the train/ directory.
        "data": {"train_path": str(config["data_root"]),
                 "num_workers": int(train["num_workers"]),
                 "img_size": int(train["img_size"]),
                 "mask_size": int(train["mask_size"])},
        "training": {"epochs": int(train["epochs"]),
                     "batch_size": int(train["batch_size"]),
                     "lr": float(train["lr"]),
                     "momentum": float(train["momentum"]),
                     "weight_decay": float(train["weight_decay"]),
                     "warmup_epochs": int(train["warmup_epochs"]),
                     "loss_type": str(train["loss_type"]),
                     "use_adversarial": bool(train["use_adversarial"]),
                     "adversarial_weight": float(train["adversarial_weight"]),
                     "save_freq": int(train["save_freq"]),
                     "print_freq": int(train["print_freq"])},
        "checkpoint": {"save_dir": str(Path(out) / WORK)},
        "seed": int(config["seed"]),
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def eval_args(config: dict, out: Path) -> Namespace:
    """The flags the original's evaluation reads. `model_type` is fixed to
    `alexnet`: the ViT and official Caffe paths are step 2 / not brought
    across."""
    to_run_config(config, out)          # validate before building arguments
    train = config["train"]
    return Namespace(
        checkpoint=str(config["encoder"]), model_type="alexnet",
        data_path=str(config["data_root"]), img_size=int(train["img_size"]),
        batch_size=int(train["batch_size"]), num_workers=int(train["num_workers"]),
        epochs=int(train["epochs"]), lr=float(train["lr"]),
        momentum=float(train["momentum"]),
        weight_decay=float(train["weight_decay"]),
        save_dir=str(Path(out) / WORK), gpu=0,
        device=str(config["device"]), seed=int(config["seed"]))


def extract_encoder(state_dict: dict) -> dict:
    """The encoder and the bottleneck, and neither the decoder nor the
    discriminator.

    The original's linear evaluation reads `model(x) -> (_, features)`, where
    `features` is the bottleneck output, so both the conv encoder (`encoder.`)
    and the bottleneck (`fc.`) are the representation. The decoder head and the
    discriminator are training machinery.
    """
    out = {}
    for key, value in state_dict.items():
        name = key[len(DDP_PREFIX):] if key.startswith(DDP_PREFIX) else key
        if name.startswith(ENCODER_PREFIXES):
            out[name] = value
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIXES} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """The other half of `extract_encoder`: put it back into the model.

    The AlexNet architecture is fixed (channels=3), so no config shape is
    needed. The keys keep their prefixes and load into the whole model; the
    decoder is expected to be missing (it is not shipped), the encoder and the
    bottleneck are not.
    """
    from models import create_model
    model = create_model("alexnet", channels=3)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder/bottleneck weights: {absent[:5]}. "
            "The decoder is expected to be missing; the encoder is not")
    return model


def _filter_numeric(raw: dict) -> tuple:
    metrics, unusable = {}, 0
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            unusable += 1
            continue
        metrics[key] = value
    return metrics, unusable


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        from train_step1 import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["checkpoint"]["save_dir"]).mkdir(parents=True, exist_ok=True)
    raw = _run(args, run_config) or {}
    metrics, unusable = _filter_numeric(raw)
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def latest_checkpoint(work: Path) -> Path:
    """The checkpoint the run finished on, chosen by epoch rather than by name:
    sorting the strings would put epoch 9 after epoch 10."""
    found = list(Path(work).glob("checkpoint_epoch_*.pth"))
    if not found:
        raise RuntimeError(
            f"training finished but no checkpoint_epoch_*.pth is in {work}; "
            "there is no encoder to hand over")
    return max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Freeze the encoder the previous stage produced, and fit a linear head."""
    import torch
    if _run is None:
        from evaluate_linear import run as _run
    args = eval_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    state = torch.load(config["encoder"], map_location="cpu", weights_only=True)
    encoder = load_encoder(state, config)
    raw = _run(args, encoder=encoder, in_dim=None) or {}
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
    state = torch.load(latest_checkpoint(Path(ctx.out) / WORK),
                       map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model_state_dict"]),
               Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=STEP1_METRIC_NAMES)


def _stage_of(config_path) -> str:
    """The stage, read before adapterlib parses the config."""
    import json
    try:
        return json.loads(Path(config_path).read_text(
            encoding="utf-8")).get("stage") or STAGES[0]
    except (OSError, ValueError, AttributeError):
        return STAGES[0]      # adapterlib will report the real problem


def _absent_reason(config_path) -> "str | None":
    """CONTRACT section 3. This stage fits a classifier on a frozen encoder; it
    produces no encoder of its own, and saying so is required."""
    if _stage_of(config_path) != "linear_eval":
        return None
    return ("this stage evaluates a frozen encoder and produces a linear "
            "classifier; the encoder it read is named in the config")


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
