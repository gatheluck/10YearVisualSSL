"""Adapter for 24_beit, step 1 and linear evaluation (BEiT; arXiv:2106.08254).

    python -m adapter --config <resolved.json> --out <dir>

BEiT (masked image modeling): a dVAE tokenizer turns each image into discrete
visual tokens; a random block of patches is replaced by a shared mask token in
the ViT input; the ViT predicts the visual tokens at the masked positions
(cross-entropy over the dVAE vocabulary) (step 1). linear_eval then probes the
frozen BEiT backbone (its mean-pooled patch tokens, embed_dim). A self-contained
re-implementation (the lab's own code); the ViT is its own (no timm). The dVAE
tokenizer is the frozen OpenAI DALL-E encoder for a real run (a hash-pinned
download named in provenance.json as tokenizer_artifact, unpickled by the `dall_e`
code pinned as the third_party/dall_e submodule and imported lazily); the hermetic
smoke uses a random tokenizer, so nothing is downloaded and no submodule is
imported. The capture's step 2 (ViT fine-tuning) is excluded, as in every port.

`encoder.pt` is the BEiT backbone trunk (patch_embed, cls_token, pos_embed, blocks,
norm); the shared mask token and the MIM head are training machinery and are left
out. `linear_eval` reads this `encoder.pt`; the representation is the model this
port trains, so the probe number is a genuine, comparable linear probe.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "24_beit"
STAGES = ("step1", "linear_eval")
METHOD_DIR = Path(__file__).resolve().parent.parent

# The pinned code this run imports: the OpenAI DALL-E dVAE tokenizer, unpickled
# for a real run by the `third_party/dall_e` submodule (the BEiT ViT itself is
# the lab's own code -- not from here). Recorded in the run manifest because the
# contract records `upstream` for every submodule-using method, so a run says
# which pinned code produced it; kept in step with provenance.json (the same
# repo+commit). The hermetic smoke imports nothing, but the field states what a
# real run would use. See docs/CONTRACT.md and provenance.json.
UPSTREAM = {"repo": "https://github.com/openai/DALL-E",
            "commit": "5be4b236bc3ade6943662354117a0e83752cc322"}

MODEL_KEYS = frozenset({"img_size", "patch_size", "vocab_size", "embed_dim",
                        "depth", "num_heads", "mlp_ratio", "drop_path_rate",
                        "init_values"})
TOKENIZER_KEYS = frozenset({"ckpt", "token_size", "input_is_mapped"})
DATA_KEYS = frozenset({"num_workers"})
MASKING_KEYS = frozenset({"num_masking_patches", "min_num_patches"})
TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "beta1", "beta2",
                           "eps", "weight_decay", "warmup_epochs", "clip_grad"})
STEP1_TRAIN_KEYS = (MODEL_KEYS | TOKENIZER_KEYS | DATA_KEYS | MASKING_KEYS
                    | TRAINING_KEYS)
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS

TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
EVAL_TOP_KEYS = TOP_KEYS | {"encoder"}
DEVICES = ("auto", "cuda", "cpu")
WORK = "work"

# encoder.pt is the BEiT backbone trunk (patch_embed, cls_token, pos_embed,
# blocks, norm). It has no single keep-prefix, so the convention is stated as the
# families to leave out: the MIM head (head.*) and the shared mask token
# (mask_token). Everything else is the encoder. Matched by startswith, so
# "mask_token" also drops the exact key.
ENCODER_EXCLUDE_PREFIXES = ("head.", "mask_token")

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


def _model_section(train: dict) -> dict:
    return {"img_size": int(train["img_size"]),
            "patch_size": int(train["patch_size"]),
            "vocab_size": int(train["vocab_size"]),
            "embed_dim": int(train["embed_dim"]),
            "depth": int(train["depth"]),
            "num_heads": int(train["num_heads"]),
            "mlp_ratio": float(train["mlp_ratio"]),
            "drop_path_rate": float(train["drop_path_rate"]),
            "init_values": float(train["init_values"])}


def _training_section(train: dict) -> dict:
    return {"epochs": int(train["epochs"]),
            "batch_size": int(train["batch_size"]),
            "lr": float(train["lr"]),
            "beta1": float(train["beta1"]),
            "beta2": float(train["beta2"]),
            "eps": float(train["eps"]),
            "weight_decay": float(train["weight_decay"]),
            "warmup_epochs": int(train["warmup_epochs"]),
            "clip_grad": float(train["clip_grad"])}


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
            f"config: stage is {stage!r}; known stages are "
            f"{', '.join(STAGES)}")
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else STEP1_TRAIN_KEYS
    top = EVAL_TOP_KEYS if stage == "linear_eval" else TOP_KEYS
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
        return {"stage": stage}

    return {
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "tokenizer": {"ckpt": str(train["ckpt"] or ""),
                      "token_size": int(train["token_size"]),
                      "input_is_mapped": bool(train["input_is_mapped"])},
        "data": {"data_root": str(config["data_root"]),
                 "num_workers": int(train["num_workers"])},
        "masking": {"num_masking_patches": int(train["num_masking_patches"]),
                    "min_num_patches": int(train["min_num_patches"])},
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
            "nothing left after excluding the mask token and MIM head; the "
            "model layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from models import build_beit
    from train_step1_beit import MODEL_ARGS
    train = config["train"]
    model = build_beit(**{k: train[k] for k in MODEL_ARGS})
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected[:5]}")
    absent = [k for k in missing if _is_encoder_key(k)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing backbone weights: {absent[:5]}. The mask "
            "token and MIM head are expected to be missing; the trunk is not")
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
        from train_step1_beit import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["output"]["checkpoint_dir"]).mkdir(parents=True,
                                                       exist_ok=True)
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
        from evaluate_linear_beit import run as _run
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
    return ("this stage fits a linear probe on the frozen BEiT backbone and "
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
