"""Adapter for var, step 1 (Tian et al., 2024).

    python -m adapter --config <resolved.json> --out <dir>

The model is the pinned upstream (`github.com/FoundationVision/VAR`) under
`third_party/var`, imported and never copied. It is a `submodule+adapter` port:
the upstream is pinned **directly** (no fork), because its model runs on a CPU or
a GPU unmodified -- unlike mar, whose model forced a device patch. The manifest
records `upstream = {repo, commit}`.

Everything else follows the earlier ports: the training loop is called rather
than reimplemented, every setting is declared, and the output goes only under
`--out`.

`encoder.pt` is the **representation side** of the VAR model -- the token and
class embeddings, the positional and level embeddings, and the transformer
blocks. The generative output head (`head`, `head_nm`) is left out, so
`encoder.pt` means the same "the representation network" it means in every other
port. Which representation a downstream probe should read from a generative model
is a separate, deferred question (CONTRACT section 7); this port ships no
`linear_eval` stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "var"
STAGES = ("step1",)
METHOD_DIR = Path(__file__).resolve().parent.parent

# The pinned upstream, recorded in every manifest. Pinned directly (no fork):
# VAR's model needs no patch. Kept in step with `third_party/var` (the gitlink)
# and `provenance.json`.
UPSTREAM = {
    "repo": "https://github.com/FoundationVision/VAR",
    "commit": "78b95394fc5896192e3a003e4b295f8ea743c48f",
}

# The representation side of VAR. `class_emb`, `pos_start`, `pos_1LC`, `lvl_1L`
# and `attn_bias_for_masking` are single tensors, so they are whole-name
# prefixes; the rest are module prefixes. The generative head (`head`,
# `head_nm`) is excluded.
ENCODER_PREFIXES = ("word_embed.", "class_emb", "lvl_embed.", "lvl_1L",
                    "pos_start", "pos_1LC", "blocks.", "attn_bias_for_masking")

# Settings that build the model (its architecture) ...
MODEL_KEYS = frozenset({"patch_nums", "vocab_size", "Cvae", "ch", "num_classes",
                        "depth", "shared_aln", "attn_l2_norm"})
# ... and settings that steer the training. `vqvae_ckpt` selects the tokeniser:
# a path for a real run, empty for the hermetic smoke's random VQVAE.
TRAIN_ONLY_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "weight_decay", "grad_clip", "vqvae_ckpt"})
TRAIN_KEYS = MODEL_KEYS | TRAIN_ONLY_KEYS
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

WORK = "work"

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


def _model_section(train: dict) -> dict:
    return {
        "patch_nums": [int(x) for x in train["patch_nums"]],
        "vocab_size": int(train["vocab_size"]),
        "Cvae": int(train["Cvae"]),
        "ch": int(train["ch"]),
        "num_classes": int(train["num_classes"]),
        "depth": int(train["depth"]),
        "shared_aln": bool(train["shared_aln"]),
        "attn_l2_norm": bool(train["attn_l2_norm"]),
    }


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the shape the trainer expects."""
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
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "training": {"epochs": int(train["epochs"]),
                     "lr": float(train["lr"]),
                     "weight_decay": float(train["weight_decay"]),
                     "grad_clip": float(train["grad_clip"])},
        "data": {"data_root": str(config["data_root"]),
                 "batch_size": int(train["batch_size"]),
                 "num_workers": int(train["num_workers"]),
                 "vqvae_ckpt": str(train["vqvae_ckpt"])},
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     seed=int(config["seed"]), device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The representation side of the model (everything but the generative
    head). Handing over the head would change what `encoder.pt` means from one
    method to the next."""
    out = {k: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIXES)}
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIXES} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """The other half of `extract_encoder`: put it back.

    The model is not self-describing -- its shapes come from the settings the
    run used -- so the resolved config is required. The head keys are expected
    to be missing; an absent *encoder* key is an error. The VQVAE that
    `build_vae_var` also constructs is random here and unused: only the VAR
    representation is loaded and compared.
    """
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from train_step1_var import _load_upstream, model_kwargs
    build_vae_var = _load_upstream()
    import torch
    _vae, model = build_vae_var(device=torch.device("cpu"),
                                flash_if_available=False, fused_if_available=False,
                                **model_kwargs(config["train"]))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this model does not have: {unexpected}")
    absent = [k for k in missing if k.startswith(ENCODER_PREFIXES)]
    if absent:
        raise RuntimeError(
            f"encoder.pt is missing encoder weights: {absent}. The head is "
            "expected to be missing; the representation is not")
    return model


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from train_step1_var import run as _run
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
                              stage="step1", body=body, upstream=UPSTREAM)
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
