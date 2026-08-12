"""Adapter for 27_ibot, step 1 and linear evaluation (Zhou et al., 2021).

    python -m adapter --config <resolved.json> --out <dir>

iBOT trains a student and a teacher ViT with a shared DINOHead: each masked
image patch and the [CLS] token predict the teacher's assignment, and the
teacher is an EMA of the student. The encoder the rest of the project wants is
the backbone; the heads and the centering buffers are training machinery.

**Which backbone is the encoder is read from the original's own recipe, not
decided here.** The model exposes `get_encoder()` returning the student, but
every official linear-evaluation script probes `--checkpoint_key teacher`, and
the paper's reported numbers are the teacher's (the EMA). So `encoder.pt` holds
the teacher ViT: `ENCODER_PREFIX = "teacher."` selects it from the checkpoint
(the `"teacher."` prefix does not match `"teacher_head."`), and the linear
evaluation freezes exactly that.

**Two shapes of setting need care.** The multi-crop scales arrive as
`[low, high]` lists, refused here by name if they are not, rather than as a
bare error from inside the loader. And most of the original's settings are read
with a default behind them; leaving one out of the contract's config would
describe a run whose temperature schedule or masking ratio was whatever the
code happened to default to. Every key the trainer reads is declared, so the
resolved config says what ran. The two pilot-only truncation knobs
(`stop_after_epochs`, `max_steps_per_epoch`) are deliberately not part of the
contract config: they are never set, so a contract run is never truncated.

The checkpoint directory in the captured config was an absolute path on the
cluster. It is refused rather than overridden: a config naming a directory that
was not used is a config that lies about the run.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import adapterlib

METHOD = "27_ibot"
STAGES = ("pretrain", "linear_eval")

# Step 1 is ViT-Small/16. `vit_base` is step 2's backbone, and step 2 has no
# official-style variant in the capture, so it was not brought across.
ARCHS = ("vit_small",)
ARCH_EMBED_DIM = {"vit_small": 384}

# Every setting the trainer reads, grouped as the original groups them. None is
# optional here even where the original reads it with a default: a default that
# fires silently is a setting the resolved config never recorded.
MODEL_KEYS = frozenset({"arch", "patch_size", "embed_dim", "drop_path_rate"})
DATA_KEYS = frozenset({"global_size", "local_size", "n_global_crops",
                       "n_local_crops", "global_crops_scale",
                       "local_crops_scale", "num_workers"})
IBOT_KEYS = frozenset({"out_dim", "head_hidden_dim", "head_bottleneck_dim",
                       "head_nlayers", "shared_head", "norm_last_layer",
                       "student_temp", "teacher_temp", "teacher_patch_temp",
                       "teacher_temp_warmup", "teacher_patch_temp_warmup",
                       "teacher_temp_warmup_epochs", "lambda_token",
                       "pred_ratio", "pred_ratio_var", "pred_shape",
                       "pred_start_epoch", "teacher_momentum_start",
                       "teacher_momentum_end", "center_momentum",
                       "center_momentum_patch"})
TRAINING_KEYS = frozenset({"epochs", "batch_size", "lr", "min_lr",
                           "weight_decay_start", "weight_decay_end",
                           "warmup_epochs", "grad_clip", "freeze_last_layer",
                           "checkpoint_health", "fail_fast_after_epoch",
                           "print_freq", "save_freq"})
HEALTH_KEYS = frozenset({"min_total_loss", "min_component_loss",
                         "max_total_loss"})
TOP_KEYS = frozenset({"stage", "seed", "data_root", "device",
                      "model", "data", "ibot", "training"})

# The four settings written as `[low, high]` lists, checked together because a
# malformed one only fails deep inside the loader with nothing to say.
RANGE_KEYS = (("data", "global_crops_scale"), ("data", "local_crops_scale"),
              ("ibot", "pred_ratio"), ("ibot", "pred_ratio_var"))

# The second stage freezes an encoder and fits a linear head; it reads its own
# small set of flags and needs the encoder to load.
EVAL_MODEL_KEYS = frozenset({"arch", "patch_size", "n_last_blocks",
                             "avgpool_patchtokens"})
EVAL_KEYS = frozenset({"epochs", "batch_size", "lr", "num_workers"})
EVAL_TOP_KEYS = frozenset({"stage", "seed", "data_root", "device", "encoder",
                           "model", "eval"})

DEVICES = ("auto", "cuda", "cpu")

# The original writes checkpoints, a config copy and TensorBoard events under
# `checkpoint.save_dir`. It gets a subdirectory of --out so nothing escapes and
# every file still reaches the manifest.
WORK = "work"

# `encoder.pt` holds the teacher ViT, the backbone the official probe uses. The
# checkpoint stores the whole iBOT model under a `module.` prefix when written
# by DDP; the teacher keys are lifted out and the prefix stripped so the file
# is a plain ViT state_dict.
ENCODER_PREFIX = "teacher."
DDP_PREFIX = "module."

# What the original calls its numbers, and what the contract calls them.
#
# `final_loss` is iBOT's own objective, sharing no scale with another method's
# loss, so it is a pretext name. Its two components (`cls`, `patch`) are real
# measurements that belong to no family in the vocabulary: mapped to `None`,
# they stay in `metrics_raw` and out of the comparable block. Inventing
# contract names for them would offer them for comparison against methods that
# have no such quantities.
PRETRAIN_METRIC_NAMES = {
    "final_loss": "final_pretext_loss",
    "final_cls_loss": None,
    "final_patch_loss": None,
    "epochs": "epochs_completed",
    "metrics_unavailable": "metrics_unavailable",
}

# The downstream numbers. Every one is a `linear_probe` name, because this
# stage measures classification against real labels -- the number this project
# exists to compare. iBOT's evaluation reports a best top-5 as well as the
# rest, so all four comparable slots are filled.
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
        raise ConfigError(
            f"config: {name} is {type(value).__name__}, not a mapping")
    _named(keys - set(value), set(value) - keys, f"config.{name}")
    return value


def check_ranges(config: dict) -> None:
    """The `[low, high]` settings, refused by name if malformed.

    The loader and the loss read these as two-element sequences; a scalar or a
    three-element list fails deep inside with nothing to say which one was
    wrong.
    """
    for block, key in RANGE_KEYS:
        value = config[block][key]
        if not isinstance(value, list) or len(value) != 2 or \
                not all(isinstance(v, (int, float)) and
                        not isinstance(v, bool) for v in value):
            raise ConfigError(
                f"config.{block}: {key} is {value!r}; expected a "
                "[low, high] pair of numbers")


def to_run_config(config: dict, out: Path) -> dict:
    """Translate the resolved config into the nested shape the original reads.

    The original's config is already grouped into `model`, `data`, `ibot` and
    `training`, so the contract keeps that shape rather than flattening it. The
    two differences are the ones the contract fixes: the data path is built
    from `data_root`, and the output path is `--out`, never a config key.
    """
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

    if config.get("device") not in DEVICES:
        raise ConfigError(
            f"config: device is {config.get('device')!r}; expected one of "
            f"{', '.join(DEVICES)}")

    if stage == "linear_eval":
        _named(EVAL_TOP_KEYS - set(config), set(config) - EVAL_TOP_KEYS,
               "config")
        model = _block(config, "model", EVAL_MODEL_KEYS)
        _block(config, "eval", EVAL_KEYS)
        if model["arch"] not in ARCHS:
            raise ConfigError(
                f"config.model: arch is {model['arch']!r}; expected one of "
                f"{', '.join(ARCHS)}")
        # The evaluation takes flags, not a document; `eval_args` builds them.
        return {"stage": stage}

    _named(TOP_KEYS - set(config), set(config) - TOP_KEYS, "config")
    model = _block(config, "model", MODEL_KEYS)
    data = _block(config, "data", DATA_KEYS)
    ibot = _block(config, "ibot", IBOT_KEYS)
    training = _block(config, "training", TRAINING_KEYS)
    if not isinstance(training["checkpoint_health"], dict):
        raise ConfigError("config.training: checkpoint_health is not a mapping")
    _named(HEALTH_KEYS - set(training["checkpoint_health"]),
           set(training["checkpoint_health"]) - HEALTH_KEYS,
           "config.training.checkpoint_health")

    if model["arch"] not in ARCHS:
        raise ConfigError(
            f"config.model: arch is {model['arch']!r}; expected one of "
            f"{', '.join(ARCHS)}")
    expected_dim = ARCH_EMBED_DIM[model["arch"]]
    if model["embed_dim"] != expected_dim:
        raise ConfigError(
            f"config.model: embed_dim is {model['embed_dim']!r}, but "
            f"{model['arch']} has embed_dim {expected_dim}; a config that "
            "disagrees with its architecture would misdescribe the run")
    check_ranges(config)

    return {
        "model": dict(model),
        "data": {**data,
                 "train_path": str(Path(config["data_root"]) / "train")},
        "ibot": dict(ibot),
        "training": dict(training),
        "checkpoint": {"save_dir": str(Path(out) / WORK)},
        "seed": config["seed"],
    }


def to_args(config: dict, out: Path) -> Namespace:
    to_run_config(config, out)          # validate before building arguments
    return Namespace(config=None, data_path=None, resume=None,
                     device=config["device"])


def eval_args(config: dict, out: Path) -> Namespace:
    """The flags the original's evaluation reads.

    The teacher backbone and the online protocol are the official recipe, fixed
    here rather than exposed: a run that quietly probed the student, or scored
    cached features, would not be the evaluation the numbers are compared
    against.
    """
    to_run_config(config, out)          # validate before building arguments
    model = config["model"]
    ev = config["eval"]
    return Namespace(
        checkpoint=str(config["encoder"]), model_type=str(model["arch"]),
        data_path=str(config["data_root"]), patch_size=int(model["patch_size"]),
        batch_size=int(ev["batch_size"]), epochs=int(ev["epochs"]),
        lr=float(ev["lr"]), num_workers=int(ev["num_workers"]),
        save_dir=str(Path(out) / WORK), resume_linear="", gpu=0,
        device=str(config["device"]), seed=int(config["seed"]),
        checkpoint_key="teacher", n_last_blocks=int(model["n_last_blocks"]),
        avgpool_patchtokens=int(model["avgpool_patchtokens"]),
        eval_mode="online", allow_unverified_checkpoint=False)


def extract_encoder(state_dict: dict) -> dict:
    """The teacher ViT, and nothing else.

    Read from the original's recipe: the official probe evaluates the teacher
    (`--checkpoint_key teacher`). The student, the two heads and the centering
    buffers are training machinery. The `teacher.` prefix is stripped so the
    file is a plain ViT state_dict; `"teacher."` does not match
    `"teacher_head."`, so the head does not come along.
    """
    out = {}
    for key, value in state_dict.items():
        name = key[len(DDP_PREFIX):] if key.startswith(DDP_PREFIX) else key
        if name.startswith(ENCODER_PREFIX):
            out[name[len(ENCODER_PREFIX):]] = value
    if not out:
        raise RuntimeError(
            f"nothing under {ENCODER_PREFIX!r} in the checkpoint; the model "
            "layout changed and encoder.pt would have been empty")
    return out


def load_encoder(state_dict: dict, config: dict):
    """The other half of `extract_encoder`: put it back into a bare ViT.

    **The encoder is not self-describing.** The architecture and patch size
    come from the resolved config; a ViT built with library defaults would be a
    differently shaped one and `load_state_dict` would report a wall of
    mismatches. Required, not optional -- found by writing the round-trip test.
    """
    from models import vit_small
    builders = {"vit_small": vit_small}
    model_cfg = config["model"]
    arch = model_cfg["arch"]
    if arch not in builders:
        raise RuntimeError(
            f"config.model.arch is {arch!r}; only {', '.join(builders)} can be "
            "rebuilt here (vit_base belongs to step 2, not brought across)")
    model = builders[arch](patch_size=int(model_cfg["patch_size"]),
                           use_mask_token=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"encoder.pt carries keys this ViT does not have: {unexpected[:5]}")
    if missing:
        raise RuntimeError(
            f"encoder.pt is missing the teacher backbone weights: "
            f"{missing[:5]}. It should be a complete ViT state_dict")
    return model


def _filter_numeric(raw: dict) -> tuple:
    """Keep only real numbers; count everything a manifest cannot carry."""
    metrics, unusable = {}, 0
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            unusable += 1
            continue
        metrics[key] = value
    return metrics, unusable


def run_training(config: dict, out: Path, _run=None) -> dict:
    if _run is None:
        from train_pretrain import run as _run
    args = to_args(config, out)
    run_config = to_run_config(config, out)
    Path(run_config["checkpoint"]["save_dir"]).mkdir(parents=True,
                                                     exist_ok=True)
    raw = _run(args, run_config) or {}
    metrics, unusable = _filter_numeric(raw)
    if "final_loss" not in metrics:
        unusable += 1
    if unusable:
        metrics["metrics_unavailable"] = unusable
    return metrics


def latest_checkpoint(work: Path) -> Path:
    """The checkpoint the run finished on, chosen by epoch rather than by name:
    sorting the strings would put epoch 9 after epoch 10."""
    found = list(Path(work).glob("checkpoint_epoch_*.pth"))
    if not found:
        raise RuntimeError(
            f"training finished but no checkpoint_epoch_*.pth is in {work}; "
            "there is no encoder to hand over")
    return max(found, key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def run_linear_eval(config: dict, out: Path, _run=None) -> dict:
    """Freeze the teacher encoder the previous stage produced, fit a linear
    head.

    The encoder is built here from `encoder.pt` and handed in, rather than
    rebuilt inside the original's loader from a whole training checkpoint:
    `load_encoder` already knows how to read one, so it is used rather than
    duplicated. `in_dim` is left to the evaluation, which derives it from the
    architecture and the feature protocol -- one place that owns that rule.
    """
    import torch
    if _run is None:
        from evaluate_linear import run as _run
    args = eval_args(config, out)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    state = torch.load(config["encoder"], map_location="cpu",
                       weights_only=True)
    encoder = load_encoder(state, config)
    raw = _run(args, encoder=encoder, in_dim=None) or {}
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
    # The iBOT checkpoint stores the whole model under the key `model`.
    state = torch.load(latest_checkpoint(Path(ctx.out) / WORK),
                       map_location="cpu", weights_only=False)
    torch.save(extract_encoder(state["model"]),
               Path(ctx.out) / "encoder.pt")
    ctx.write_metrics(metrics, names=PRETRAIN_METRIC_NAMES)


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
