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
port.

The `linear_eval` stage is the project's answer, for VAR, to the question
CONTRACT section 7 left open -- which representation a downstream probe reads
from a generative model. The lab's ARSSL harness probes VAR's **VQVAE
tokeniser** (its encoder's continuous features, average-pooled), *not* the VAR
transformer this port trains. So `linear_eval` reads no `encoder.pt`; it builds
the tokeniser from the config and needs the pretrained tokeniser weights
(`vqvae_ckpt`, a download via `bin/fetch-weights.py`) for a real number. The
number therefore describes the fixed tokeniser, not VAR's learned
representation; `docs/EVAL_DOWNLOAD.md` records this and why.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "var"
STAGES = ("pretrain", "linear_eval")
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

# The linear_eval stage probes the VQVAE tokeniser (see the module docstring),
# so it needs the model architecture (to build the VQVAE), the tokeniser
# checkpoint (`vqvae_ckpt`), and the probe's own hyperparameters. It reads no
# `encoder.pt`: the representation it evaluates is the tokeniser, not the VAR
# transformer step 1 produces. So the top-level keys are the same as step 1's
# (no `encoder` key, unlike methods whose probe reads their own encoder.pt).
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay", "img_size"})
EVAL_TRAIN_KEYS = MODEL_KEYS | EVAL_PROBE_KEYS | frozenset({"vqvae_ckpt"})

WORK = "work"

PRETRAIN_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

# The downstream numbers: all four comparable linear-probe accuracies, because
# the ARSSL probe records top-5 as well as top-1, and best as well as final.
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
    stage = config["stage"]
    if stage not in STAGES:
        raise ConfigError(
            f"config: stage is {stage!r}; known stages are "
            f"{', '.join(STAGES)}")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    keys = EVAL_TRAIN_KEYS if stage == "linear_eval" else TRAIN_KEYS
    _named(keys - set(train), set(train) - keys, "config.train")

    if config["device"] not in DEVICES:
        raise ConfigError(
            f"config: device is {config['device']!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        # The evaluation reads the config directly (data_root, train.*); it
        # takes no translated run-config document, only a validated stage.
        return {"stage": stage}

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

    Like `build_vqvae`, the `build_vae_var` call is wrapped in
    `restore_default_init` so the upstream's global `reset_parameters` no-op does
    not leak to the other methods sharing this process (the test suite holds them
    all at once).
    """
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from train_pretrain_var import (_load_upstream, model_kwargs,
                                    restore_default_init)
    build_vae_var = _load_upstream()
    import torch
    with restore_default_init():
        _vae, model = build_vae_var(
            device=torch.device("cpu"),
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
        from train_pretrain_var import run as _run
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


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Fit a linear probe on the frozen VQVAE tokeniser's features.

    This deliberately does **not** read `encoder.pt`: VAR's downstream
    representation, as the lab measured it, is the VQVAE encoder, not the VAR
    transformer step 1 produced (see the module docstring and
    `docs/EVAL_DOWNLOAD.md`). The evaluator builds the tokeniser from the
    config; a real run points `vqvae_ckpt` at the pretrained tokeniser."""
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_var import run as _run
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config) or {}
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
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
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


def _stage_of(config_path) -> str:
    """The stage, read before adapterlib parses the config, so `main` can tell
    it which stage is running and whether an encoder is expected."""
    import json
    try:
        return json.loads(Path(config_path).read_text(
            encoding="utf-8")).get("stage") or STAGES[0]
    except (OSError, ValueError, AttributeError):
        return STAGES[0]      # adapterlib will report the real problem


def _absent_reason(config_path) -> "str | None":
    """CONTRACT section 3. linear_eval fits a classifier on the frozen VQVAE
    tokeniser; it produces no encoder of its own, and saying so is required."""
    if _stage_of(config_path) != "linear_eval":
        return None
    return ("this stage fits a linear probe on the frozen VQVAE tokeniser and "
            "produces a classifier, not an encoder; it reads no encoder.pt "
            "(VAR's probed representation is the tokeniser, not the transformer)")


def main(argv: list[str] | None = None) -> int:
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
