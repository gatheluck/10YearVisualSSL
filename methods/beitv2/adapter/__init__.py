"""Adapter for beitv2: the as-is frozen-backbone linear probe (no pretraining).

    python -m adapter --config <resolved.json> --out <dir>

BEiT v2 (Peng, Dong, Bao, Ye & Wei, 2022) is ported **eval-only**: a single
`linear_eval` stage freezes the official self-supervised pretrained backbone
(`beitv2_base_patch16_224_pt1k`) and fits a linear probe on its pooled patch-token
feature. BEiT v2's from-scratch pretraining (a VQ-KD visual tokenizer + masked
image modelling, many GPU-days) is the excluded step, so the port reuses the
released checkpoint, exactly as the capture's as-is comparison does. The probed
representation is a genuine learned SSL feature, so the number is comparable -- the
autoregressive/MIM sibling of `eva02` / `36_franca`.

The backbone class is the author's, imported from the pinned `microsoft/unilm`
submodule (`third_party/unilm/beit2`, **MIT**, imported not copied), so this adapter
records `UPSTREAM` and passes it to `adapterlib.run` -- the run manifest then names
the pinned code that produced the result (CONTRACT: upstream is recorded for
submodule-using methods). The weights are a sha256-pinned download recorded as
`backbone_artifact` in provenance.json and fetched by `bin/fetch-weights.py`. This
stage trains nothing and produces no `encoder.pt` -- it records
`encoder_absent_reason` and the backbone it read is named in the config
(`train.ckpt`). CI stays hermetic: with an empty `train.ckpt` the smoke builds a
random tiny BEiT-style ViT, so nothing is downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "beitv2"
STAGES = ("linear_eval",)
METHOD_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = METHOD_DIR.parent.parent / "third_party" / "unilm"

# The pinned upstream. Its commit is cross-checked against the checked-out
# submodule and provenance.json by tests/test_method_beitv2.py.
UPSTREAM = {
    "repo": "https://github.com/microsoft/unilm",
    "commit": "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5",
}

DEVICES = ("auto", "cuda", "cpu")

# ── linear_eval config ───────────────────────────────────────────────────────
# The architecture keys let the hermetic smoke build a random tiny BEiT-style ViT;
# the shipped config pins the official base (embed 768, depth 12) and names the
# downloaded ckpt. The rel-pos-bias / mean-pooling architecture constants are fixed
# in the backbone, not settings (see models/beitv2_backbone.py).
EVAL_MODEL_KEYS = frozenset({"name", "ckpt", "img_size", "patch_size",
                             "embed_dim", "depth", "num_heads"})
EVAL_PROBE_KEYS = frozenset({"epochs", "batch_size", "num_workers", "lr",
                             "momentum", "weight_decay"})
EVAL_TRAIN_KEYS = EVAL_MODEL_KEYS | EVAL_PROBE_KEYS
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "train"})

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

    _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")
    train = config["train"]
    if not isinstance(train, dict):
        raise ConfigError("config: train is not a mapping")
    _named(EVAL_TRAIN_KEYS - set(train), set(train) - EVAL_TRAIN_KEYS,
           "config.train")
    return {"stage": stage}


def _filter_numeric(raw: dict) -> tuple:
    metrics, unusable = {}, 0
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            unusable += 1
            continue
        metrics[k] = v
    return metrics, unusable


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    to_run_config(config, out)          # validate before running
    if _run is None:
        if str(METHOD_DIR) not in sys.path:
            sys.path.insert(0, str(METHOD_DIR))
        from evaluate_linear_beitv2 import run as _run
    raw = _run(Namespace(config=None, data_path=None, device=config["device"]),
               config=config) or {}
    metrics, unusable = _filter_numeric(raw)
    unusable += sum(1 for k in ("best_top1_acc", "final_top1_acc")
                    if k not in metrics)
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def body(ctx: adapterlib.Context) -> None:
    ctx.write_metrics(run_linear_eval(ctx.config, ctx.out),
                      names=LINEAR_EVAL_METRIC_NAMES)


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
            "classifier, not an encoder; the downloaded backbone it read is named "
            "in the config (train.ckpt)")


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
