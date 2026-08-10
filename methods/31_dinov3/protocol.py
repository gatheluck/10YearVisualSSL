"""Immutable DINOv3 Step 2 protocol and optimizer-step schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


SCHEDULE_EPOCHS = 300
CORE_EPOCHS = 250
GRAM_STAGE_EPOCHS = 50
MILESTONE_EPOCHS = {100, 200, 300}

WORLD_SIZE = 8
BATCH_SIZE_PER_GPU = 128
PHYSICAL_GLOBAL_BATCH = 1024

BASE_LR = 6.0e-4
MIN_LR = 1.0e-6
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 0.05

CORE_TEACHER_MOMENTUM = 0.994
GRAM_TEACHER_MOMENTUM = 0.999
TEACHER_TEMP_START = 0.04
TEACHER_TEMP_END = 0.07
TEACHER_TEMP_WARMUP_EPOCHS = 25
GRAM_WEIGHT_START = 0.0
GRAM_WEIGHT_END = 2.0
DINO_LOCAL_WEIGHT_START = 1.0
DINO_LOCAL_WEIGHT_END = 0.5
GRAM_RAMP_FRACTION = 0.25

STEP2_PROTOCOL = "dinov3_vit_b16_300ep_physical1024_core250_gram50_step_schedules_v5"
GRAM_RELEASED_COMPRESSED = "released_compressed_300"
GRAM_CORE_ONLY = "core_only"
GRAM_EMA_ABLATION = "ema_gram_ablation"
GRAM_MODES = {GRAM_RELEASED_COMPRESSED, GRAM_CORE_ONLY, GRAM_EMA_ABLATION}

OFFICIAL_GRAM_UPDATE_EPOCHS = (1010, 1020, 1030)
GRAM_TEACHER_UPDATE_EPOCHS = tuple(
    math.floor(epoch * SCHEDULE_EPOCHS / 1200 + 0.5)
    for epoch in OFFICIAL_GRAM_UPDATE_EPOCHS
)


@dataclass(frozen=True)
class StepSchedule:
    lr: float
    teacher_temp: float
    teacher_momentum: float
    gram_weight: float
    dino_local_weight: float


def _linear(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return end
    progress = min(max(index / (count - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _cosine(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return end
    progress = min(max(index / (count - 1), 0.0), 1.0)
    interpolation = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * interpolation


def optimizer_step_schedule(
    optimizer_step: int,
    steps_per_epoch: int,
    *,
    stage_step: int | None = None,
) -> StepSchedule:
    """Return values for the next optimizer update.

    ``optimizer_step`` is zero based and counts completed updates. ``stage_step``
    is used only by the isolated forced-Gram pilot to exercise epoch-251 logic.
    """
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = SCHEDULE_EPOCHS * steps_per_epoch
    if not 0 <= optimizer_step < total_steps:
        raise ValueError(
            f"optimizer_step must be in [0, {total_steps}), got {optimizer_step}"
        )

    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    if optimizer_step < warmup_steps:
        lr = _linear(0.0, BASE_LR, optimizer_step, warmup_steps)
    else:
        lr = _cosine(
            BASE_LR,
            MIN_LR,
            optimizer_step - warmup_steps,
            total_steps - warmup_steps,
        )

    temp_warmup_steps = TEACHER_TEMP_WARMUP_EPOCHS * steps_per_epoch
    teacher_temp = (
        _linear(
            TEACHER_TEMP_START,
            TEACHER_TEMP_END,
            optimizer_step,
            temp_warmup_steps,
        )
        if optimizer_step < temp_warmup_steps
        else TEACHER_TEMP_END
    )

    schedule_step = optimizer_step if stage_step is None else stage_step
    core_steps = CORE_EPOCHS * steps_per_epoch
    if schedule_step < core_steps:
        teacher_momentum = CORE_TEACHER_MOMENTUM
        gram_weight = GRAM_WEIGHT_START
        dino_local_weight = DINO_LOCAL_WEIGHT_START
    else:
        teacher_momentum = GRAM_TEACHER_MOMENTUM
        ramp_steps = max(1, math.ceil(steps_per_epoch * GRAM_RAMP_FRACTION))
        ramp_index = schedule_step - core_steps
        gram_weight = _cosine(
            GRAM_WEIGHT_START, GRAM_WEIGHT_END, ramp_index, ramp_steps
        )
        dino_local_weight = _cosine(
            DINO_LOCAL_WEIGHT_START,
            DINO_LOCAL_WEIGHT_END,
            ramp_index,
            ramp_steps,
        )

    return StepSchedule(
        lr=lr,
        teacher_temp=teacher_temp,
        teacher_momentum=teacher_momentum,
        gram_weight=gram_weight,
        dino_local_weight=dino_local_weight,
    )


def _require_values(section: str, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"Step2 {section}.{key} must be exactly {value!r}, "
                f"got {actual.get(key)!r}"
            )


def validate_protocol_config(cfg: dict[str, Any]) -> None:
    """Reject any canonical setting that changes the audited Step 2 contract."""
    if cfg.get("protocol_id") != STEP2_PROTOCOL:
        raise ValueError(
            f"Step2 config protocol_id must be {STEP2_PROTOCOL!r}, "
            f"got {cfg.get('protocol_id')!r}"
        )

    _require_values(
        "model",
        cfg.get("model", {}),
        {
            "arch": "vit_base_patch16",
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "n_register_tokens": 4,
            "drop_path_rate": 0.1,
            "use_rope": True,
            "rope_base": 100.0,
            "rope_rescale_coords": 2.0,
            "dino_out_dim": 65536,
            "ibot_out_dim": 65536,
            "dino_head_hidden_dim": 2048,
            "dino_head_bottleneck_dim": 256,
            "ibot_head_hidden_dim": 2048,
            "ibot_head_bottleneck_dim": 256,
        },
    )
    _require_values(
        "data",
        cfg.get("data", {}),
        {
            "n_global_crops": 2,
            "n_local_crops": 8,
            "global_size": 224,
            "local_size": 96,
            "global_scale": [0.32, 1.0],
            "local_scale": [0.05, 0.32],
        },
    )
    _require_values(
        "training",
        cfg.get("training", {}),
        {
            "epochs": SCHEDULE_EPOCHS,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "lr": BASE_LR,
            "min_lr": MIN_LR,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "teacher_momentum_core": CORE_TEACHER_MOMENTUM,
            "teacher_momentum_gram": GRAM_TEACHER_MOMENTUM,
            "grad_clip": 3.0,
            "ibot_mask_ratio_min": 0.10,
            "ibot_mask_ratio_max": 0.50,
            "ibot_mask_sample_probability": 0.50,
            "koleo_loss_weight": 0.1,
        },
    )
    if "accum_steps" in cfg.get("training", {}):
        raise ValueError("Step2 forbids gradient accumulation; remove training.accum_steps")
    _require_values(
        "loss",
        cfg.get("loss", {}),
        {
            "teacher_temp_start": TEACHER_TEMP_START,
            "teacher_temp_end": TEACHER_TEMP_END,
            "teacher_temp_warmup_epochs": TEACHER_TEMP_WARMUP_EPOCHS,
            "dino_local_weight_start": DINO_LOCAL_WEIGHT_START,
            "dino_local_weight_end": DINO_LOCAL_WEIGHT_END,
            "student_temp": 0.1,
            "sk_n_iters": 3,
        },
    )
    _require_values(
        "gram",
        cfg.get("gram", {}),
        {
            "loss_weight_start": GRAM_WEIGHT_START,
            "loss_weight_end": GRAM_WEIGHT_END,
            "ramp_fraction_of_epoch_251": GRAM_RAMP_FRACTION,
        },
    )
    _require_values(
        "checkpoint", cfg.get("checkpoint", {}), {"save_freq": 100}
    )

    metadata = cfg.get("recipe_adaptation", {})
    _require_values(
        "recipe_adaptation.released_recipe",
        metadata.get("released_recipe", {}),
        {
            "core_epochs": 1000,
            "gram_total_epochs": 1200,
            "teacher_temp_warmup_epochs": 100,
            "core_teacher_momentum": CORE_TEACHER_MOMENTUM,
            "gram_teacher_momentum": GRAM_TEACHER_MOMENTUM,
            "gram_teacher_refresh_epochs": list(OFFICIAL_GRAM_UPDATE_EPOCHS),
        },
    )
    _require_values(
        "recipe_adaptation.compressed_step2",
        metadata.get("compressed_step2", {}),
        {
            "schedule_clock": "optimizer_step",
            "core_epochs": CORE_EPOCHS,
            "gram_epochs": GRAM_STAGE_EPOCHS,
            "teacher_temp_warmup_epochs": TEACHER_TEMP_WARMUP_EPOCHS,
            "gram_teacher_snapshot_epoch": CORE_EPOCHS,
            "gram_teacher_refresh_epochs": list(GRAM_TEACHER_UPDATE_EPOCHS),
            "gram_ramp": "cosine_over_first_quarter_of_epoch_251",
        },
    )
    _require_values(
        "recipe_adaptation.unified_deviations",
        metadata.get("unified_deviations", {}),
        {
            "backbone": "vit_base_patch16",
            "physical_global_batch": PHYSICAL_GLOBAL_BATCH,
            "batch_layout": "8_ranks_x_128_no_accumulation",
            "sinkhorn_scope": "physical_global_batch_1024",
            "koleo_scope": "released_non_distributed_per_rank_physical_batch_128",
            "global_crop_size": 224,
            "local_crop_size": 96,
            "gram_teacher_crop_size": 224,
            "dino_prototypes": 65536,
            "ibot_prototypes": 65536,
            "high_resolution_adaptation": "omitted_to_preserve_strict_224_input",
        },
    )


def validate_physical_batch(
    world_size: int,
    batch_size_per_gpu: int,
    total_batch: int,
) -> None:
    if total_batch != PHYSICAL_GLOBAL_BATCH:
        raise ValueError(
            f"TOTAL_BATCH must be exactly {PHYSICAL_GLOBAL_BATCH}, got {total_batch}"
        )
    if world_size != WORLD_SIZE:
        raise ValueError(f"Step2 requires exactly {WORLD_SIZE} ranks, got {world_size}")
    if batch_size_per_gpu != BATCH_SIZE_PER_GPU:
        raise ValueError(
            f"Step2 requires {BATCH_SIZE_PER_GPU} samples per GPU, "
            f"got {batch_size_per_gpu}"
        )
    if world_size * batch_size_per_gpu != total_batch:
        raise ValueError("Step2 physical global batch must be 8 x 128 = 1024")
