"""Adapter for 1_context_prediction, step 1 (official-style track).

    python -m adapter --config <resolved.json> --out <dir>

**This does not train anything itself.** It translates the resolved config
into the arguments `train_step1_alexnet_official.py` already takes, calls its
`run()`, and arranges the results under `--out` with the names the contract
fixes. A second training loop here would put the same rule in two places -- the
root of past defects in this project -- and would let optimizer or DDP details
drift away from what was validated on the cluster.

Three things it refuses rather than guesses:

- **A missing setting.** No defaults are filled in. The resolved config exists
  to say what ran, and a value supplied here would not be in it
- **An unknown setting.** A misspelled key that is quietly ignored is a
  setting that never took effect while the config claims it did
- **`resume`.** Not supported yet in this adapter. Accepting it and ignoring
  it would silently produce a different run from the one asked for
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "1_context_prediction"

# **The stage comes from the config, not from a flag.** The contract fixes the
# adapter's arguments at exactly two, and says anything else that affects the
# result belongs in the config (CONTRACT section 2). A --stage flag would be an
# input that `config_sha256` does not cover.
#
# Each stage declares exactly the keys it reads, so a setting the stage never
# looks at cannot sit in a config claiming to have had an effect.
STAGES = {
    "step1": {
        "top": frozenset({"seed", "data_root", "device", "train"}),
        "train": frozenset({"max_steps", "batch_size", "num_workers", "lr",
                            "save_every_steps", "eval_every_steps",
                            "eval_batches"}),
    },
    "linear_eval": {
        "top": frozenset({"seed", "data_root", "device", "train", "encoder"}),
        "train": frozenset({"epochs", "batch_size", "feature_batch_size",
                            "num_workers", "lr", "img_size"}),
    },
}
DEVICES = ("auto", "cuda", "cpu")
ENCODER_PREFIX = "encoder."

# What the pretext evaluation is expected to report. Named, because `run()`
# also returns `global_step`, and a step count is a number without being a
# result: on its own it kept metrics.json looking populated when the
# evaluation had been skipped entirely.
EVAL_METRICS = ("val_loss", "val_acc1")

# What the original calls its numbers, and what the contract calls them.
#
# **`val_acc1` is the pretext task's accuracy, not a downstream one.** The
# model here is built with `num_classes=8` and predicts which of eight
# relative positions a patch sits in. Mapping it to a linear-probe name would
# put it in the same column as real classification accuracy, and the column
# would look right (CONTRACT, metric vocabulary).
STEP1_METRIC_NAMES = {
    "val_loss": "final_pretext_loss",
    "val_acc1": "final_pretext_top1_accuracy",
    "global_step": "steps_completed",
    "metrics_unavailable": "metrics_unavailable",
}

# The original writes run_config.json, progress.jsonl and its checkpoints into
# save_dir under its own names. It gets a subdirectory of --out so that
# nothing escapes and every file still ends up in the manifest.
WORK = "work"


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


def stage_of(config: dict) -> str:
    """Which stage the config asks for. Refused rather than defaulted."""
    stage = config.get("stage")
    if stage is None:
        raise ConfigError(
            "config: missing stage. It is not defaulted: the stage decides "
            f"what runs, so it belongs inside config_sha256. Known stages: "
            f"{', '.join(sorted(STAGES))}")
    if stage not in STAGES:
        raise ConfigError(
            f"config: stage is {stage!r}; known stages are "
            f"{', '.join(sorted(STAGES))}")
    return stage


def to_args(config: dict, out: Path) -> Namespace:
    """Translate the resolved config into the original's arguments."""
    if "resume" in config:
        raise ConfigError(
            "resume is not supported by this adapter yet. The original "
            "supports it; until this adapter records what was resumed from, "
            "accepting the key would hide it")
    stage = stage_of(config)
    top_keys = STAGES[stage]["top"] | {"stage"}
    _named(top_keys - set(config), set(config) - top_keys, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    train_keys = STAGES[stage]["train"]
    _named(train_keys - set(train), set(train) - train_keys, "config.train")

    device = config["device"]
    if device not in DEVICES:
        raise ConfigError(
            f"config: device is {device!r}; expected one of "
            f"{', '.join(DEVICES)}")

    common = dict(data_path=str(config["data_root"]),
                  save_dir=str(Path(out) / WORK),
                  seed=int(config["seed"]), device=device, gpu=0)
    if stage == "step1":
        return Namespace(
            **common, resume="", allow_resume=False,
            max_steps=int(train["max_steps"]),
            batch_size=int(train["batch_size"]),
            num_workers=int(train["num_workers"]),
            lr=float(train["lr"]),
            save_every_steps=int(train["save_every_steps"]),
            eval_every_steps=int(train["eval_every_steps"]),
            eval_batches=int(train["eval_batches"]),
        )
    return Namespace(
        **common, checkpoint=None, encoder=str(config["encoder"]),
        epochs=int(train["epochs"]),
        batch_size=int(train["batch_size"]),
        feature_batch_size=int(train["feature_batch_size"]),
        num_workers=int(train["num_workers"]),
        lr=float(train["lr"]),
        img_size=int(train["img_size"]),
    )


def _usable_metrics(raw: dict) -> dict:
    """Keep the numbers; count what is missing rather than hide it.

    `evaluate_pretext` returns None for its values when it saw no samples.
    Writing None would break the contract, and dropping it without a word
    would make a run that never evaluated look like one that did.

    A metric that is absent counts the same as one that is unusable. Without
    that, removing the final evaluation altogether left `global_step` behind
    and nothing noticed.
    """
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    unusable += sum(1 for k in EVAL_METRICS if k not in metrics)
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def extract_encoder(state_dict: dict) -> dict:
    """The encoder's parameters alone, with the prefix stripped.

    The contract asks for the encoder's weights. Handing over the pretext head
    as well (`fc7`, `bn7`, `fc8`, `fc9`) would give downstream users something
    to strip and would change what `encoder.pt` means from one method to the
    next.
    """
    out = {k[len(ENCODER_PREFIX):]: v for k, v in state_dict.items()
           if k.startswith(ENCODER_PREFIX)}
    if not out:
        raise RuntimeError(
            f"no parameters under {ENCODER_PREFIX!r} in the checkpoint; the "
            "model layout changed and encoder.pt would have been empty")
    return out


def load_final_state(out: Path, _load=None) -> dict:
    """The checkpoint the training run finished on.

    `_load` exists so the absence can be tested without torch.
    """
    final = Path(out) / WORK / "final.pth"
    if not final.is_file():
        raise RuntimeError(
            f"training finished but {final} was not written; there is no "
            "encoder to hand over")
    if _load is None:
        import torch
        _load = torch.load
    return _load(final, map_location="cpu", weights_only=False)


def run_training(config: dict, out: Path, _run=None) -> dict:
    """Call the original run for this stage, and return contract-fit metrics.

    `_run` exists so the translation can be tested without a GPU or a dataset;
    it defaults to the original for the stage the config names.
    """
    stage = stage_of(config)
    if _run is None:
        if stage == "step1":
            from train_step1_alexnet_official import run as _run
        else:
            from evaluate_linear_official import run as _run
    args = to_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    raw = _run(args)
    if stage == "step1":
        return _usable_metrics(raw)
    return _eval_metrics(raw)


# The accuracies the linear evaluation reports. Named, for the same reason
# EVAL_METRICS is named for step 1: a run that produced none must not look
# populated because some other number happened to be present.
LINEAR_EVAL_METRICS = ("best_top1_acc", "best_top5_acc",
                       "final_top1_acc", "final_top5_acc")

# These are downstream classification against real labels, so they are the
# numbers that may be compared across methods.
LINEAR_EVAL_METRIC_NAMES = {
    "best_top1_acc": "best_linear_probe_top1_accuracy",
    "best_top5_acc": "best_linear_probe_top5_accuracy",
    "final_top1_acc": "final_linear_probe_top1_accuracy",
    "final_top5_acc": "final_linear_probe_top5_accuracy",
    "metrics_unavailable": "metrics_unavailable",
}


def _eval_metrics(results: dict) -> dict:
    metrics, unusable = {}, 0
    for k in LINEAR_EVAL_METRICS:
        v = results.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


# This stage evaluates an encoder and produces a classifier. CONTRACT section
# 3 allows producing no encoder only when the reason is recorded.
NO_ENCODER_REASON = ("this stage evaluates a frozen encoder and produces a "
                     "linear classifier; the encoder it read is named in "
                     "work/results.json")


def body(ctx: adapterlib.Context) -> None:
    import torch
    metrics = run_training(ctx.config, ctx.out)
    if stage_of(ctx.config) == "step1":
        state = load_final_state(ctx.out)
        torch.save(extract_encoder(state["state_dict"]),
                   Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=(
        STEP1_METRIC_NAMES if stage_of(ctx.config) == "step1"
        else LINEAR_EVAL_METRIC_NAMES))


def main(argv: list[str] | None = None) -> int:
    # cuBLAS reads this once, when the CUDA context is created. Setting it
    # later has no effect, so it happens here rather than inside the training
    # run: without it, cuBLAS reductions are free to vary between runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        import json
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        stage = stage_of(cfg)
        return adapterlib.run(
            config=a.config, out=a.out, method=METHOD, stage=stage, body=body,
            encoder_absent_reason=(None if stage == "step1"
                                   else NO_ENCODER_REASON))
    except (adapterlib.AdapterError, ConfigError) as exc:
        # A refusal, not a run result. Leave no manifest behind.
        print(f"  *** {exc}", file=sys.stderr)
        return 2
