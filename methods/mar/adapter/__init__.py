"""Adapter for mar, step 1 (Li et al., 2024).

    python -m adapter --config <resolved.json> --out <dir>

**This is the first port whose model is a pinned submodule rather than code
copied into the method.** The upstream (`github.com/gatheluck/mar`, a fork of
`LTH14/mar` carrying a two-line device patch, DESIGN section 2.8) lives under
`third_party/mar` and is imported, never vendored -- so the manifest records
`upstream = {repo, commit}` and the contract can say exactly which code ran.

Everything else follows the earlier ports: the training loop is called rather
than reimplemented, every setting is declared so a key the stage never reads
cannot sit in a config claiming to have had an effect, and the output goes
only under `--out`.

`encoder.pt` is the **MAE-encoder side** of the MAR model -- the tokens'
projection, the encoder blocks and norm, the class embedding and the learned
buffers used by `forward_mae_encoder`. The decoder and the diffusion loss head
are left out, so `encoder.pt` means the same "the representation network" it
means in every other port. Which representation a *downstream probe* should
read from a generative model is a separate, deferred question (CONTRACT
section 7); this port does not answer it, and ships no `linear_eval` stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "mar"
STAGES = ("pretrain",)
METHOD_DIR = Path(__file__).resolve().parent.parent

# The pinned upstream, recorded in every manifest. The commit is the fork's
# device patch; the URL is the fork, not LTH14/mar (DESIGN section 2.8). Kept
# in step with `third_party/mar` (the gitlink) and `provenance.json`.
UPSTREAM = {
    "repo": "https://github.com/gatheluck/mar",
    "commit": "e8d163b7274ce2ef933ee0d40fcc45abaffc42fe",
}

# The MAE-encoder side of MAR. `encoder_pos_embed_learned` and `fake_latent`
# are single parameters, so they are whole-name prefixes; the rest are module
# prefixes. Everything the decoder or the diffusion loss owns is excluded.
ENCODER_PREFIXES = ("z_proj.", "z_proj_ln.", "encoder_pos_embed_learned",
                    "encoder_blocks.", "encoder_norm.", "class_emb.",
                    "fake_latent")

# Settings that build the model (its architecture) ...
MODEL_KEYS = frozenset({"img_size", "vae_stride", "patch_size", "vae_embed_dim",
                        "class_num", "buffer_size", "diffloss_d", "diffloss_w",
                        "mask_ratio_min", "label_drop_prob"})
# ... and settings that steer the training. Together, and no others.
TRAIN_ONLY_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "weight_decay", "grad_clip"})
TRAIN_KEYS = MODEL_KEYS | TRAIN_ONLY_KEYS
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")

# The trainer writes its checkpoints here. A subdirectory of --out, so nothing
# escapes and everything is listed.
WORK = "work"

# What the original calls its number, and what the contract calls it. `step1`
# is a pretext stage: the loss is MAR's own masked-autoregressive objective,
# on no scale shared with any other method (adapterlib.METRIC_VOCABULARY).
PRETRAIN_METRIC_NAMES = {
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
        "img_size": int(train["img_size"]),
        "vae_stride": int(train["vae_stride"]),
        "patch_size": int(train["patch_size"]),
        "vae_embed_dim": int(train["vae_embed_dim"]),
        "class_num": int(train["class_num"]),
        "buffer_size": int(train["buffer_size"]),
        "diffloss_d": int(train["diffloss_d"]),
        "diffloss_w": int(train["diffloss_w"]),
        "mask_ratio_min": float(train["mask_ratio_min"]),
        "label_drop_prob": float(train["label_drop_prob"]),
    }


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the shape the trainer expects.

    The contract's config is flat and declares only what affects the result;
    the trainer reads a nested mapping with `model`, `training`, `data` and
    `output` sections. This is where the two meet.
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
        "seed": int(config["seed"]),
        "model": _model_section(train),
        "training": {"epochs": int(train["epochs"]),
                     "lr": float(train["lr"]),
                     "weight_decay": float(train["weight_decay"]),
                     "grad_clip": float(train["grad_clip"])},
        "data": {"cached_path": str(config["data_root"]),
                 "batch_size": int(train["batch_size"]),
                 "num_workers": int(train["num_workers"])},
        "output": {"checkpoint_dir": str(Path(out) / WORK)},
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    # device is validated above and forwarded here: validating it and then not
    # passing it on is how a trainer comes to ignore it and sniff the hardware
    # instead (see docs/GPU.md).
    return Namespace(config=None, data_path=None, resume=None,
                     seed=int(config["seed"]), device=config["device"])


def extract_encoder(state_dict: dict) -> dict:
    """The MAE-encoder side of the model.

    MAR is an encoder, a decoder and a diffusion loss head. The contract asks
    for the encoder; handing over the rest would change what `encoder.pt` means
    from one method to the next.
    """
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
    run used -- so the resolved config is required, not optional: rebuilding
    with library defaults produces a differently shaped model and
    `load_state_dict` reports a wall of size mismatches. The decoder keys are
    expected to be missing; an absent *encoder* key is an error.
    """
    if str(METHOD_DIR) not in sys.path:
        sys.path.insert(0, str(METHOD_DIR))
    from train_pretrain_mar import _load_upstream, model_kwargs
    mar_base, _dgd, _cached = _load_upstream()
    model = mar_base(**model_kwargs(config["train"]))
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


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from train_pretrain_mar import run as _run
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
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage="pretrain", body=body, upstream=UPSTREAM)
    except (adapterlib.AdapterError, ConfigError) as exc:
        print(f"  *** {exc}", file=sys.stderr)
        return 2
