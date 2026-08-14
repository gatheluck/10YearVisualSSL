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
STAGES = ("pretrain", "linear_eval")

# Every setting the original reads, and no others.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "base_lr",
                        "momentum", "weight_decay", "img_size", "dim",
                        "pred_dim", "save_freq", "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})

# The second stage reads a different set. It needs an encoder to freeze, and
# none of step 1's optimiser settings: a key a stage never reads is a setting
# claiming an effect it never had.
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
EVAL_TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "optimizer", "weight_decay", "img_size", "dim",
                             "pred_dim", "print_freq"})

# The unified ViT-B/16 Step-2 path (arch: vit), additive to the native ResNet-50
# path. Its model dimensions and AdamW/cosine recipe are its own; the native and
# ViT key sets are disjoint so a knob from one path cannot leak into the other.
ARCHS = ("resnet", "vit")
VIT_MODEL_KEYS = frozenset({"dim", "pred_dim", "img_size", "patch_size",
                            "embed_dim", "depth", "num_heads", "mlp_ratio",
                            "drop_rate", "attn_drop_rate"})
PRETRAIN_VIT_ONLY = frozenset({"epochs", "batch_size", "num_workers", "lr",
                               "weight_decay", "warmup_epochs", "min_lr",
                               "save_at_epochs"})
PRETRAIN_VIT_KEYS = VIT_MODEL_KEYS | PRETRAIN_VIT_ONLY
EVAL_VIT_KEYS = VIT_MODEL_KEYS | frozenset({"epochs", "batch_size",
                                            "num_workers", "lr", "optimizer",
                                            "weight_decay", "print_freq"})
_VIT_FLOATS = ("mlp_ratio", "drop_rate", "attn_drop_rate")

DEVICES = ("auto", "cuda", "cpu")

# The original writes checkpoints, a copy of its config and TensorBoard events
# under `checkpoint.save_dir`. It gets a subdirectory of --out so that nothing
# escapes and every file still reaches the manifest.
WORK = "work"

# `get_encoder()` returns `self.backbone`, and the original's own linear
# evaluation loads exactly that. A checkpoint written under DDP carries a
# `module.` prefix, which the original's loader strips before loading.
ENCODER_PREFIX = "backbone."

# ResNet-50's pooled feature width, which the original's own evaluation also
# hard-codes for the resnet path.
BACKBONE_DIM = 2048
DDP_PREFIX = "module."

# What the original calls its numbers, and what the contract calls them.
#
# `final_loss` is SimSiam's negative cosine similarity -- this method's own
# objective, sharing no scale with any other method's loss, so it is a pretext
# name. `final_z_std` is the collapse monitor and has no contract slot at all:
# `None` keeps it in `metrics_raw` and out of `metrics`, which is the whole
# point of that path.
# The downstream numbers. Every one is a `linear_probe` name, because this
# stage measures classification against real labels -- which is the number
# this project exists to compare.
#
# **Three, not four.** The first port's evaluation also reports a best top-5;
# this original does not, and inventing one would be a number nothing
# measured.
LINEAR_EVAL_METRIC_NAMES = {
    "best_top1_acc": "best_linear_probe_top1_accuracy",
    "final_top1_acc": "final_linear_probe_top1_accuracy",
    "final_top5_acc": "final_linear_probe_top5_accuracy",
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

PRETRAIN_METRIC_NAMES = {
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
        # The evaluation takes flags, not a document. Validation above is the
        # part that matters here; `eval_args` builds what it actually reads.
        return {"stage": stage}

    if arch == "vit":
        # The unified ViT-B/16 Step-2 recipe: AdamW + warmup/cosine, milestone
        # checkpoints. The schedule (lr, warmup, min_lr) is applied directly by
        # the trainer, so nothing is recomputed here.
        return {
            "arch": "vit",
            "model": {k: (float(train[k]) if k in _VIT_FLOATS else int(train[k]))
                      for k in VIT_MODEL_KEYS},
            "data": {"train_path": str(Path(config["data_root"]) / "train"),
                     "img_size": int(train["img_size"]),
                     "num_workers": int(train["num_workers"])},
            "training": {"epochs": int(train["epochs"]),
                         "batch_size": int(train["batch_size"]),
                         "lr": float(train["lr"]),
                         "weight_decay": float(train["weight_decay"]),
                         "warmup_epochs": int(train["warmup_epochs"]),
                         "min_lr": float(train["min_lr"]),
                         "save_at_epochs": [int(e) for e in
                                            train["save_at_epochs"]]},
            "checkpoint": {"save_dir": str(Path(out) / WORK),
                           "allow_resume": False},
            "seed": int(config["seed"]),
        }

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


def eval_args(config: dict, out: Path) -> Namespace:
    """The arguments the original's evaluation reads.

    Its inputs are command-line flags rather than a nested mapping, so unlike
    step 1 there is no config document to build -- this is where the contract's
    flat config meets an argparse namespace.
    """
    to_run_config(config, out)          # validate before building arguments
    train = config["train"]
    model_type = "vit" if train.get("arch", "resnet") == "vit" else "resnet"
    return Namespace(
        checkpoint=str(config["encoder"]), model_type=model_type,
        data_path=str(config["data_root"]),
        batch_size=int(train["batch_size"]), epochs=int(train["epochs"]),
        lr=float(train["lr"]), optimizer=str(train["optimizer"]),
        weight_decay=float(train["weight_decay"]),
        num_workers=int(train["num_workers"]),
        save_dir=str(Path(out) / WORK), gpu=0, resume_linear="",
        device=str(config["device"]), img_size=int(train["img_size"]),
        seed=int(config["seed"]))


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
    train = config["train"]
    if train.get("arch", "resnet") == "vit":
        from models import build_simsiam_vit
        from train_pretrain_vit_simsiam import model_kwargs as vit_mk
        model = build_simsiam_vit(**vit_mk(train))
    else:
        from models import build_simsiam_resnet
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
    arch = config.get("train", {}).get("arch", "resnet")
    if _run is None:
        if arch == "vit":
            from train_pretrain_vit_simsiam import run as _run
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


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Freeze the encoder the previous stage produced, and fit a linear head.

    The encoder is built here rather than inside the original's loader: the
    contract's artifact is `encoder.pt`, the backbone alone, while the
    captured loader rebuilds the whole SimSiam model from a training
    checkpoint with `strict=True`. `load_encoder` already knows how to read
    one, so it is used rather than duplicated.
    """
    import torch
    if _run is None:
        from evaluate_linear_official import run as _run
    args = eval_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    state = torch.load(config["encoder"], map_location="cpu",
                       weights_only=True)
    encoder = load_encoder(state, config)
    # The frozen feature width: ViT's CLS embedding (embed_dim) for arch: vit,
    # ResNet-50's pooled 2048 otherwise. FrozenBackboneLinear flattens the
    # encoder output, so the linear head must be sized to match.
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
    train = ctx.config.get("train", {})
    work = Path(ctx.out) / WORK
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
    """CONTRACT section 3. This stage fits a classifier on a frozen encoder;
    it produces no encoder of its own, and saying so is required."""
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
