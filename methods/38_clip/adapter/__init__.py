"""Adapter for 38_clip: the as-is Step-1 probe and the label-text Step-2 pretrain.

    python -m adapter --config <resolved.json> --out <dir>

Two comparisons live here, selected by the stage (and, for the eval, a `recipe`
key):

- **Step 1 (as-is)** -- `linear_eval` with no `recipe`: freeze the official
  pretrained OpenAI CLIP **ViT-B/32** (a sha256-pinned download, built through the
  pinned `openai/CLIP` under `third_party/CLIP`) and probe its pooled image
  embedding. CLIP's 400M image-text training data is not public, so the as-is row
  reuses the released checkpoint. This trains nothing and produces no `encoder.pt`.

- **Step 2 (label-text adaptation)** -- `pretrain` trains a CLIP **ViT-B/16** from
  scratch on ImageNet-1k, pairing each labeled image with an official OpenAI
  ImageNet class-name prompt (symmetric image-text contrastive loss), then
  `linear_eval` with `recipe: unified` freezes the trained image tower
  (`encoder.pt`) and probes it. **This is a supervised label-text adaptation, not
  unlabeled VSSL**: every config, checkpoint and result records
  `supervised_label_text_adaptation=true` / `main_vssl_comparability=false`, and it
  must not be reported as a comparable self-supervised ImageNet result.

`encoder.pt` (Step 2) is the trained CLIP image tower (`visual.*`, the prefix
stripped so it loads into a plain `VisionTransformer`); the text tower and the
logit scale are training machinery. Milestone encoders (`encoder_epoch{N}.pt`) are
written for the 100/200/300 probe sweep.

Licence: openai/CLIP is MIT (commercial use permitted). Nothing under it is copied
into this repository; the model constructor and the BPE tokenizer are imported from
the pinned submodule through PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "38_clip"
STAGES = ("pretrain", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = METHOD_DIR.parent.parent / "third_party" / "CLIP"

# The pinned upstream. Its commit is cross-checked against the checked-out
# submodule and provenance.json by tests/test_method_38_clip.py.
UPSTREAM = {
    "repo": "https://github.com/openai/CLIP",
    "commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"
RECIPES = ("native", "unified")

# ── Step-2 pretrain config (nested, as the capture groups it) ────────────────
P_DEFINITION_KEYS = frozenset({"source_method", "source_code_commit", "adaptation",
                               "supervised_label_text_adaptation",
                               "main_vssl_comparability"})
P_MODEL_KEYS = frozenset({"embed_dim", "image_resolution", "vision_layers",
                          "vision_width", "vision_patch_size", "context_length",
                          "vocab_size", "transformer_width", "transformer_heads",
                          "transformer_layers"})
P_DATA_KEYS = frozenset({"image_size", "num_workers"})
P_PROMPTS_KEYS = frozenset({"use_official_imagenet", "templates"})
P_TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "min_lr", "beta1",
                             "beta2", "eps", "weight_decay", "warmup_epochs",
                             "clip_grad_norm", "save_at_epochs"})
PRETRAIN_TOP_KEYS = frozenset({"stage", "seed", "data_root", "device",
                               "definition", "model", "data", "prompts", "training"})

# ── Step-1 (as-is download) linear_eval config ───────────────────────────────
EVAL_MODEL_KEYS = frozenset({"ckpt", "resolution", "patch_size", "width", "layers",
                             "heads", "output_dim"})
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = EVAL_MODEL_KEYS | EVAL_PROBE_KEYS
# ── Step-2 (unified, trained-encoder) linear_eval config ─────────────────────
EVAL_UNIFIED_MODEL_KEYS = frozenset({"resolution", "patch_size", "width", "layers",
                                     "heads", "output_dim"})
EVAL_UNIFIED_KEYS = EVAL_UNIFIED_MODEL_KEYS | EVAL_PROBE_KEYS
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_UNIFIED_TOP_KEYS = TOP_KEYS | {"encoder"}

# encoder.pt (Step 2) is the trained CLIP image tower.
ENCODER_PREFIX = "visual."

PRETRAIN_METRIC_NAMES = {
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


def _block(config: dict, name: str, keys: frozenset) -> dict:
    value = config[name]
    if not isinstance(value, dict):
        raise ConfigError(f"config: {name} is {type(value).__name__}, not a mapping")
    _named(keys - set(value), set(value) - keys, f"config.{name}")
    return value


def _check_disclosure(definition: dict) -> None:
    """The Step-2 supervision disclosure is not optional (README, provenance)."""
    if definition.get("supervised_label_text_adaptation") is not True:
        raise ConfigError(
            "config.definition: supervised_label_text_adaptation must be true. "
            "CLIP Step 2 is a supervised label-text adaptation and must be "
            "disclosed as such -- it is not unlabeled VSSL")
    if definition.get("main_vssl_comparability") is not False:
        raise ConfigError(
            "config.definition: main_vssl_comparability must be false. The CLIP "
            "Step-2 number is a supervised-adaptation reference, not a comparable "
            "self-supervised result")


def _eval_recipe(config: dict) -> str:
    recipe = config.get("train", {}).get("recipe", "native")
    if recipe not in RECIPES:
        raise ConfigError(
            f"config.train: recipe is {recipe!r}; expected one of "
            f"{', '.join(RECIPES)}")
    return recipe


def to_run_config(config: dict, out: Path) -> dict:
    for key in ("output", "result_dir", "checkpoint_dir", "checkpoint"):
        if key in config:
            raise ConfigError(
                f"config: {key} is set. The output location is not a setting: "
                "the contract fixes it at --out, and a config naming a "
                "directory would claim a location that was not used")

    stage = config.get("stage")
    if stage not in STAGES:
        raise ConfigError(
            f"config: stage is {stage!r}; known stages are {', '.join(STAGES)}")
    if config.get("device") not in DEVICES:
        raise ConfigError(
            f"config: device is {config.get('device')!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        recipe = _eval_recipe(config)
        if recipe == "unified":
            _named(EVAL_UNIFIED_TOP_KEYS - set(config),
                   set(config) - EVAL_UNIFIED_TOP_KEYS, "config")
            train = config["train"]
            if not isinstance(train, dict):
                raise ConfigError("config: train is not a mapping")
            rest = {k: v for k, v in train.items() if k != "recipe"}
            _named(EVAL_UNIFIED_KEYS - set(rest), set(rest) - EVAL_UNIFIED_KEYS,
                   "config.train")
        else:
            _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")
            train = config["train"]
            if not isinstance(train, dict):
                raise ConfigError("config: train is not a mapping")
            _named(EVAL_TRAIN_KEYS - set(train), set(train) - EVAL_TRAIN_KEYS,
                   "config.train")
        return {"stage": stage}

    # pretrain (the label-text Step 2)
    _named(PRETRAIN_TOP_KEYS - set(config), set(config) - PRETRAIN_TOP_KEYS,
           "config")
    definition = _block(config, "definition", P_DEFINITION_KEYS)
    _check_disclosure(definition)
    _block(config, "model", P_MODEL_KEYS)
    _block(config, "data", P_DATA_KEYS)
    _block(config, "prompts", P_PROMPTS_KEYS)
    _block(config, "training", P_TRAINING_KEYS)

    run = {k: (dict(config[k]) if isinstance(config[k], dict) else config[k])
           for k in ("definition", "model", "data", "prompts", "training")}
    run["seed"] = int(config["seed"])
    run["data"]["train_path"] = str(Path(config["data_root"]) / "train")
    run["output"] = {"checkpoint_dir": str(Path(out) / WORK)}
    return run


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    out = {k[len(ENCODER_PREFIX):]: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIX)}
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIX!r} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """Rebuild the CLIP image tower (Step 2) from encoder.pt."""
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_clip_visual
    train = config["train"]
    model = build_clip_visual({
        "resolution": int(train["resolution"]),
        "patch_size": int(train["patch_size"]),
        "width": int(train["width"]),
        "layers": int(train["layers"]),
        "heads": int(train["heads"]),
        "output_dim": int(train["output_dim"]),
    })
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this CLIP image tower does not have: "
            f"{unexpected[:5]}")
    if missing:
        raise RuntimeError(
            f"encoder.pt is missing image-tower weights: {missing[:5]}. It should "
            "be the full visual tower; the text tower and logit scale are excluded")
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
        from train_pretrain_vit_clip import run as _run
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
    to_run_config(config, out)          # validate before running
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_clip import run as _run
    recipe = config.get("train", {}).get("recipe", "native")
    if recipe == "unified":
        # Step 2: probe the from-scratch trained image tower (encoder.pt).
        state = torch.load(config["encoder"], map_location="cpu", weights_only=True)
        model = load_encoder(state, config)
        raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
                   config=config, model=model) or {}
    else:
        # Step 1 (as-is): the official downloaded ViT-B/32 image tower.
        raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
                   config=config, official_dir=UPSTREAM_DIR) or {}
    metrics, unusable = _filter_numeric(raw)
    unusable += sum(1 for k in ("best_top1_acc", "final_top1_acc")
                    if k not in metrics)
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def latest_checkpoint(work: Path) -> Path:
    latest = work / "checkpoint_latest.pth"
    if not latest.is_file():
        raise RuntimeError(
            f"training finished but {latest} was not written; there is no "
            "encoder to hand over")
    return latest


def body(ctx: adapterlib.Context) -> None:
    import torch
    if ctx.stage == "linear_eval":
        ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                          names=LINEAR_EVAL_METRIC_NAMES)
        return
    metrics = run_training(ctx.config, ctx.out)
    work = Path(ctx.out) / WORK
    state = torch.load(latest_checkpoint(work), map_location="cpu",
                       weights_only=False)
    torch.save(extract_encoder(state["model"]), Path(ctx.out) / "encoder.pt")
    for n in ctx.config.get("training", {}).get("save_at_epochs", []):
        ck = work / f"checkpoint_epoch_{int(n)}.pth"
        if ck.is_file():
            s = torch.load(ck, map_location="cpu", weights_only=False)
            torch.save(extract_encoder(s["model"]),
                       Path(ctx.out) / f"encoder_epoch{int(n)}.pt")
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


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
    return ("this stage fits a linear probe on a frozen backbone and produces a "
            "classifier, not an encoder; the backbone it read (an official "
            "download for Step 1, or the trained encoder.pt for Step 2) is named "
            "in the config")


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
