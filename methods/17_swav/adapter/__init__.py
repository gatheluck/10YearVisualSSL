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
STAGES = ("pretrain", "linear_eval")

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

# The second stage freezes an encoder and fits a linear head. It reads its own
# small set, and needs `out_dim`/`hidden_mlp`/`nmb_prototypes` to rebuild the
# model `load_encoder` loads the backbone into -- the encoder is not
# self-describing, the same reason the first stage's round trip needs them.
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
EVAL_TRAIN_KEYS = frozenset({"epochs", "batch_size", "lr", "weight_decay",
                             "num_workers", "img_size", "out_dim", "hidden_mlp",
                             "nmb_prototypes"})

# The unified ViT-B/16 Step-2 path (arch: vit), additive to the native ResNet-50
# path. The multi-crop lists, loss (temperature, Sinkhorn) and prototype/head
# dims are shared; the ViT adds its trunk dimensions and an AdamW/cosine recipe
# with milestone checkpoints. The native and ViT key sets are disjoint so a knob
# from one path cannot leak into the other.
ARCHS = ("resnet", "vit")
VIT_MODEL_KEYS = frozenset({"out_dim", "hidden_mlp", "nmb_prototypes",
                            "image_size", "patch_size", "embed_dim", "depth",
                            "num_heads", "mlp_ratio", "drop_rate",
                            "attn_drop_rate"})
VIT_LOSS_KEYS = frozenset({"temperature", "sinkhorn_iters", "sinkhorn_eps"})
VIT_DATA_KEYS = frozenset({"num_workers", "color_jitter_strength"}) | set(CROP_KEYS)
PRETRAIN_VIT_ONLY = frozenset({"epochs", "batch_size", "lr", "weight_decay",
                               "warmup_epochs", "min_lr", "print_freq",
                               "freeze_prototypes_steps", "save_at_epochs"})
PRETRAIN_VIT_KEYS = (VIT_MODEL_KEYS | VIT_LOSS_KEYS | VIT_DATA_KEYS
                     | PRETRAIN_VIT_ONLY)
EVAL_VIT_KEYS = frozenset({"image_size", "patch_size", "embed_dim", "depth",
                           "num_heads", "mlp_ratio", "drop_rate",
                           "attn_drop_rate", "epochs", "batch_size", "lr",
                           "weight_decay", "num_workers"})
_VIT_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")

DEVICES = ("auto", "cuda", "cpu")

# ResNet-50's pooled feature width, which the original's own evaluation also
# uses for the resnet path.
BACKBONE_DIM = 2048

WORK = "work"

# `get_encoder()` wraps `self.encoder` with a flatten; the weights live under
# that prefix. The projection head and the prototypes are training machinery:
# the prototypes are learned cluster centres, not the representation.
ENCODER_PREFIX = "encoder."
DDP_PREFIX = "module."

PRETRAIN_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

# The downstream numbers. Every one is a `linear_probe` name, because this
# stage measures classification against real labels -- the number this project
# exists to compare. Three, not four: this original's evaluation reports a best
# top-1 and a final top-1 and top-5, but no best top-5.
LINEAR_EVAL_METRIC_NAMES = {
    "best_top1_acc": "best_linear_probe_top1_accuracy",
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
    arch = train.get("arch", "resnet")
    if arch not in ARCHS:
        raise ConfigError(
            f"config.train: arch is {arch!r}; expected one of "
            f"{', '.join(ARCHS)}")
    rest = {k: v for k, v in train.items() if k != "arch"}
    if stage == "linear_eval":
        keys = EVAL_VIT_KEYS if arch == "vit" else EVAL_TRAIN_KEYS
    else:
        keys = PRETRAIN_VIT_KEYS if arch == "vit" else TRAIN_KEYS
    _named(keys - set(rest), set(rest) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        # The evaluation takes flags, not a document; `eval_args` builds them.
        return {"stage": stage}

    check_crops(rest)          # multi-crop is a pretrain setting

    if arch == "vit":
        return {
            "arch": "vit",
            "model": {k: (float(train[k]) if k in _VIT_FLOATS else int(train[k]))
                      for k in VIT_MODEL_KEYS},
            "data": {"train_path": str(Path(config["data_root"]) / "train"),
                     "num_workers": int(train["num_workers"]),
                     "color_jitter_strength":
                         float(train["color_jitter_strength"]),
                     **{k: list(train[k]) for k in CROP_KEYS}},
            "loss": {"temperature": float(train["temperature"]),
                     "sinkhorn_eps": float(train["sinkhorn_eps"]),
                     "sinkhorn_iters": int(train["sinkhorn_iters"])},
            "training": {"epochs": int(train["epochs"]),
                         "batch_size": int(train["batch_size"]),
                         "lr": float(train["lr"]),
                         "weight_decay": float(train["weight_decay"]),
                         "warmup_epochs": int(train["warmup_epochs"]),
                         "min_lr": float(train["min_lr"]),
                         "print_freq": int(train["print_freq"]),
                         "freeze_prototypes_steps":
                             int(train["freeze_prototypes_steps"]),
                         "save_at_epochs": [int(e) for e in
                                            train["save_at_epochs"]]},
            "checkpoint": {"save_dir": str(Path(out) / WORK)},
            "seed": int(config["seed"]),
        }

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


def eval_args(config: dict, out: Path) -> Namespace:
    """The flags the original's evaluation reads. `model_type` is fixed to
    `resnet`: the ViT is step 2, which this port does not include."""
    to_run_config(config, out)          # validate before building arguments
    train = config["train"]
    is_vit = train.get("arch", "resnet") == "vit"
    img_size = int(train["image_size"] if is_vit else train["img_size"])
    return Namespace(
        checkpoint=str(config["encoder"]),
        model_type="vit" if is_vit else "resnet",
        data_path=str(config["data_root"]),
        batch_size=int(train["batch_size"]), epochs=int(train["epochs"]),
        lr=float(train["lr"]), weight_decay=float(train["weight_decay"]),
        num_workers=int(train["num_workers"]), img_size=img_size,
        save_dir=str(Path(out) / WORK), gpu=0, resume_linear="",
        device=str(config["device"]), seed=int(config["seed"]))


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
    train = config["train"]
    if train.get("arch", "resnet") == "vit":
        from models import build_vit_swav
        from train_pretrain_vit_swav import model_kwargs as vit_mk
        model = build_vit_swav(**vit_mk(train))
    else:
        from models import build_resnet_swav
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
    arch = config.get("train", {}).get("arch", "resnet")
    if _run is None:
        if arch == "vit":
            from train_pretrain_vit_swav import run as _run
        else:
            from train_pretrain_resnet import run as _run
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


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Freeze the encoder the previous stage produced, and fit a linear head.

    The encoder is built here from `encoder.pt` and handed in, rather than
    rebuilt inside the original's loader from a whole training checkpoint:
    `load_encoder` already knows how to read one, so it is used rather than
    duplicated.
    """
    import torch
    if _run is None:
        from evaluate_linear import run as _run
    args = eval_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    state = torch.load(config["encoder"], map_location="cpu",
                       weights_only=True)
    encoder = load_encoder(state, config)
    # The frozen feature width: ViT's CLS embedding (embed_dim) for arch: vit,
    # ResNet-50's pooled 2048 otherwise.
    train = config["train"]
    in_dim = (int(train["embed_dim"]) if train.get("arch", "resnet") == "vit"
              else BACKBONE_DIM)
    raw = _run(args, encoder=encoder, in_dim=in_dim) or {}
    metrics, unusable = {}, 0
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            unusable += 1
            continue
        metrics[key] = value
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
    train = ctx.config.get("train", {})
    if train.get("arch", "resnet") == "vit":
        # The ViT trainer writes checkpoint_latest.pth (final) and a
        # checkpoint_epoch_{N}.pth at each milestone; hand over encoder.pt for
        # the final state and encoder_epoch{N}.pt for each milestone probe.
        latest = work / "checkpoint_latest.pth"
        if not latest.is_file():
            raise RuntimeError(
                f"training finished but {latest} was not written; there is no "
                "encoder to hand over")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        torch.save(extract_encoder(state["state_dict"]),
                   Path(ctx.out) / "encoder.pt")
        for n in train.get("save_at_epochs", []):
            ck = work / f"checkpoint_epoch_{int(n)}.pth"
            if ck.is_file():
                s = torch.load(ck, map_location="cpu", weights_only=False)
                torch.save(extract_encoder(s["state_dict"]),
                           Path(ctx.out) / f"encoder_epoch{int(n)}.pt")
    else:
        state = torch.load(latest_checkpoint(work),
                           map_location="cpu", weights_only=False)
        torch.save(extract_encoder(state["state_dict"]),
                   Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


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
