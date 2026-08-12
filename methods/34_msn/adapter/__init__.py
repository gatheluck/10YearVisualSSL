"""Adapter for 34_msn, step 1 and linear evaluation (MSN; arXiv:2204.07141).

    python -m adapter --config <resolved.json> --out <dir>

MSN (Masked Siamese Networks): an anchor ViT sees patch-dropped multi-crop views
and an EMA target ViT sees one un-dropped view; both are matched to learnable
prototypes via a soft-nearest-neighbour classifier under an MSN cross-entropy plus
a me-max regulariser (step 1). linear_eval then probes the frozen anchor backbone
(its CLS token). The ViT and the MSN loss are the official facebookresearch/msn
code, pinned as the submodule third_party/msn and imported (never copied); only
the multi-view augmentation is reimplemented (the upstream one trips the pinned
Pillow) and the trainer is single-process.

Licence note: the msn code is released under CC BY-NC 4.0 (non-commercial). This
port is used only for academic research; see provenance.json and README.md.
Nothing under that licence is copied into this repository -- the code is a pinned
submodule.

`encoder.pt` is the anchor ViT trunk (the `fc.*` projection head is training
machinery and is excluded); it loads into a bare ViT whose `forward_features`
returns the CLS token at embed_dim. `linear_eval` reads this `encoder.pt`; the
representation is the model this port trains, so the probe number is comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "34_msn"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

# The pinned upstream, recorded in every manifest: the ViT and MSN loss are the
# official facebookresearch/msn code, imported from third_party/msn (not copied).
UPSTREAM = {
    "repo": "https://github.com/facebookresearch/msn",
    "commit": "4388dc1eadbe3042b85d3296d41b9b207656e043",
}

MODEL_KEYS = frozenset({"img_size", "patch_size", "embed_dim", "depth",
                        "num_heads", "mlp_ratio", "use_bn", "hidden_dim",
                        "output_dim", "drop_path_rate"})
DATA_KEYS = frozenset({"focal_size", "rand_crop_scale", "focal_crop_scale",
                       "color_jitter", "rand_views", "focal_views", "patch_drop",
                       "num_workers", "label_smoothing"})
CRITERION_KEYS = frozenset({"num_proto", "temperature", "start_sharpen",
                            "final_sharpen", "me_max", "memax_weight", "use_ent",
                            "ent_weight", "use_sinkhorn"})
TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "start_lr", "final_lr",
                           "warmup", "weight_decay", "final_weight_decay",
                           "clip_grad", "ema_start", "ema_final"})
STEP1_TRAIN_KEYS = MODEL_KEYS | DATA_KEYS | CRITERION_KEYS | TRAINING_KEYS

EVAL_MODEL_KEYS = frozenset({"img_size", "patch_size", "embed_dim", "depth",
                             "num_heads", "mlp_ratio", "drop_path_rate"})
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = EVAL_MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# encoder.pt is the ViT trunk. The MSN projection head is excluded.
ENCODER_EXCLUDE_PREFIXES = ("fc.",)

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


def _is_encoder_key(key: str) -> bool:
    return not any(key.startswith(p) for p in ENCODER_EXCLUDE_PREFIXES)


def _model_section(t: dict) -> dict:
    return {"img_size": int(t["img_size"]), "patch_size": int(t["patch_size"]),
            "embed_dim": int(t["embed_dim"]), "depth": int(t["depth"]),
            "num_heads": int(t["num_heads"]), "mlp_ratio": float(t["mlp_ratio"]),
            "use_bn": bool(t["use_bn"]), "hidden_dim": int(t["hidden_dim"]),
            "output_dim": int(t["output_dim"]),
            "drop_path_rate": float(t["drop_path_rate"])}


def _data_section(config: dict, t: dict) -> dict:
    return {"data_root": str(config["data_root"]),
            "focal_size": int(t["focal_size"]),
            "rand_crop_scale": list(t["rand_crop_scale"]),
            "focal_crop_scale": list(t["focal_crop_scale"]),
            "color_jitter": float(t["color_jitter"]),
            "rand_views": int(t["rand_views"]),
            "focal_views": int(t["focal_views"]),
            "patch_drop": float(t["patch_drop"]),
            "num_workers": int(t["num_workers"]),
            "label_smoothing": float(t["label_smoothing"])}


def _criterion_section(t: dict) -> dict:
    return {"num_proto": int(t["num_proto"]),
            "temperature": float(t["temperature"]),
            "start_sharpen": float(t["start_sharpen"]),
            "final_sharpen": float(t["final_sharpen"]),
            "me_max": bool(t["me_max"]),
            "memax_weight": float(t["memax_weight"]),
            "use_ent": bool(t["use_ent"]),
            "ent_weight": float(t["ent_weight"]),
            "use_sinkhorn": bool(t["use_sinkhorn"])}


def _training_section(t: dict) -> dict:
    return {"epochs": int(t["epochs"]), "batch_size": int(t["batch_size"]),
            "lr": float(t["lr"]), "start_lr": float(t["start_lr"]),
            "final_lr": float(t["final_lr"]), "warmup": int(t["warmup"]),
            "weight_decay": float(t["weight_decay"]),
            "final_weight_decay": float(t["final_weight_decay"]),
            "clip_grad": float(t["clip_grad"]),
            "ema_start": float(t["ema_start"]),
            "ema_final": float(t["ema_final"])}


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
            f"config: stage is {stage!r}; known stages are {', '.join(STAGES)}")
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else STEP1_TRAIN_KEYS
    top = EVAL_TOP_KEYS if stage == "linear_eval" else TOP_KEYS
    _named(top - set(config), set(config) - top, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, not a mapping")
    _named(keys - set(train), set(train) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        return {"stage": stage}

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "data": _data_section(config, train),
        "criterion": _criterion_section(train),
        "training": _training_section(train),
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    out = {k: v for k, v in state_dict.items() if _is_encoder_key(k)}
    if not out:
        raise RuntimeError(
            "nothing left after excluding the projection head; the model layout "
            "changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_msn_backbone
    train = config["train"]
    # Build the eval backbone from the adapter's own key set (EVAL_MODEL_KEYS ==
    # build_msn_backbone's kwargs), so load_encoder depends only on `models` and
    # not on the shared-name import chain in evaluate_linear_msn (which would trip
    # the in-process suite where several methods share a `models`/`data` package).
    backbone = build_msn_backbone(**{k: train[k] for k in EVAL_MODEL_KEYS})
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys the backbone does not have: {unexpected[:5]}")
    absent = [k for k in missing if _is_encoder_key(k)]
    if absent:
        raise RuntimeError(f"encoder.pt is missing backbone weights: {absent[:5]}")
    return backbone


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
        from train_step1_msn import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
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
        from evaluate_linear_msn import run as _run
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


def body(ctx: adapterlib.Context) -> None:
    import torch
    if ctx.stage == "linear_eval":
        ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                          names=LINEAR_EVAL_METRIC_NAMES)
        return
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
    return ("this stage fits a linear probe on the frozen MSN backbone and "
            "produces a classifier, not an encoder; it reads the encoder.pt "
            "named in the config")


def main(argv: "list[str] | None" = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=_stage_of(a.config), body=body,
                              upstream=UPSTREAM,
                              encoder_absent_reason=_absent_reason(a.config))
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
