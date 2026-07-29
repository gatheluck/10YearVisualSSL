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
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "1_context_prediction"
STAGE = "step1"

# Every training setting the original takes, and no others. Kept as a set so
# that a missing one and an unknown one are both named rather than absorbed.
TRAIN_KEYS = frozenset({
    "max_steps", "batch_size", "num_workers", "lr",
    "save_every_steps", "eval_every_steps", "eval_batches",
})
TOP_KEYS = frozenset({"seed", "data_root", "device", "train"})
DEVICES = ("auto", "cuda", "cpu")
ENCODER_PREFIX = "encoder."

# What the pretext evaluation is expected to report. Named, because `run()`
# also returns `global_step`, and a step count is a number without being a
# result: on its own it kept metrics.json looking populated when the
# evaluation had been skipped entirely.
EVAL_METRICS = ("val_loss", "val_acc1")

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


def to_args(config: dict, out: Path) -> Namespace:
    """Translate the resolved config into the original's arguments."""
    if "resume" in config:
        raise ConfigError(
            "resume is not supported by this adapter yet. The original "
            "supports it; until this adapter records what was resumed from, "
            "accepting the key would hide it")
    _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")

    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError(f"config: train is {type(train).__name__}, "
                          "not a mapping")
    _named(TRAIN_KEYS - set(train), set(train) - TRAIN_KEYS, "config.train")

    device = config["device"]
    if device not in DEVICES:
        raise ConfigError(
            f"config: device is {device!r}; expected one of "
            f"{', '.join(DEVICES)}")

    return Namespace(
        data_path=str(config["data_root"]),
        save_dir=str(Path(out) / WORK),
        seed=int(config["seed"]),
        device=device,
        gpu=0,
        resume="",
        allow_resume=False,
        max_steps=int(train["max_steps"]),
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        lr=float(train["lr"]),
        save_every_steps=int(train["save_every_steps"]),
        eval_every_steps=int(train["eval_every_steps"]),
        eval_batches=int(train["eval_batches"]),
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
    """Call the original training run and return metrics fit for the contract.

    `_run` exists so the translation can be tested without a GPU or a dataset;
    it defaults to the original.
    """
    if _run is None:
        from train_step1_alexnet_official import run as _run
    args = to_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    return _usable_metrics(_run(args))


def body(ctx: adapterlib.Context) -> None:
    import torch
    metrics = run_training(ctx.config, ctx.out)
    state = load_final_state(ctx.out)
    torch.save(extract_encoder(state["state_dict"]),
               Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        return adapterlib.run(config=a.config, out=a.out, method=METHOD,
                              stage=STAGE, body=body)
    except (adapterlib.AdapterError, ConfigError) as exc:
        # A refusal, not a run result. Leave no manifest behind.
        print(f"  *** {exc}", file=sys.stderr)
        return 2
