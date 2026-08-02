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
STAGES = ("step1", "linear_eval")

# Every setting the original reads, and no others.
# Read from the trainer, not guessed: it takes **two** learning rates, as
# LARS does in the paper, and `base_lr` is not one of them. A first attempt
# declared `base_lr` and the run failed on a key nothing reads.
TRAIN_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr_weights",
                        "lr_biases", "weight_decay", "img_size", "projector",
                        "lambd", "warmup_epochs", "precision", "save_freq",
                        "print_freq"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})

# The second stage freezes an encoder and fits a linear head. It needs the
# encoder to load, and none of step 1's optimiser or precision settings: a key
# a stage never reads is a setting claiming an effect it never had. `projector`
# is declared because the encoder is not self-describing -- `load_encoder`
# rebuilds the model with it, the same way the first stage's round trip does.
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
EVAL_TRAIN_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers",
                             "img_size", "projector"})
DEVICES = ("auto", "cuda", "cpu")

# ResNet-50's pooled feature width, which the original's own evaluation also
# uses for the resnet path.
BACKBONE_DIM = 2048

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

# The downstream numbers. Every one is a `linear_probe` name, because this
# stage measures classification against real labels -- the number this project
# exists to compare. Three, not four: this original's evaluation reports a best
# top-1 and a final top-1 and top-5, but no best top-5, and inventing one would
# be a number nothing measured.
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
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else TRAIN_KEYS
    _named(top - set(config), set(config) - top, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    _named(keys - set(train), set(train) - keys, "config.train")

    device = config["device"]
    if device not in DEVICES:
        raise ConfigError(
            f"config: device is {device!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        # The evaluation takes flags, not a document; `eval_args` builds them.
        return {"stage": stage}

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


def eval_args(config: dict, out: Path) -> Namespace:
    """The flags the original's evaluation reads.

    Its inputs are command-line flags rather than a nested mapping, so unlike
    step 1 there is no config document to build -- this is where the contract's
    flat config meets an argparse namespace. `model_type` is fixed to `resnet`:
    the ViT is step 2, which this port does not include.
    """
    to_run_config(config, out)          # validate before building arguments
    train = config["train"]
    return Namespace(
        checkpoint=str(config["encoder"]), model_type="resnet",
        data_path=str(config["data_root"]),
        batch_size=int(train["batch_size"]), epochs=int(train["epochs"]),
        lr=float(train["lr"]), num_workers=int(train["num_workers"]),
        img_size=int(train["img_size"]), save_dir=str(Path(out) / WORK),
        gpu=0, device=str(config["device"]), seed=int(config["seed"]))


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
    raw = _run(args, encoder=encoder, in_dim=BACKBONE_DIM) or {}
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
    state = torch.load(latest_checkpoint(Path(ctx.out) / WORK),
                       map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["state_dict"]),
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
