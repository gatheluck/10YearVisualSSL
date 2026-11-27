"""Atomic checkpoint publication and distributed training-state validation."""

import os
import uuid

import torch
import torch.distributed as dist


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def parse_stop_after_epochs(value, schedule_epochs=300):
    if value in (None, ""):
        return None
    stop_after = int(value)
    if stop_after <= 0 or stop_after > schedule_epochs:
        raise ValueError(f"STOP_AFTER_EPOCHS must be in [1, {schedule_epochs}]")
    return stop_after


def checkpoint_contract(pilot_mode):
    return {
        "canonical": not pilot_mode,
        "resumable": not pilot_mode,
        "is_pilot": bool(pilot_mode),
        "artifact_kind": (
            "noncanonical_nonresumable_pilot" if pilot_mode else "canonical"
        ),
    }


def per_rank_batch_size(global_batch, world_size, expected_global_batch=1024):
    global_batch = int(global_batch)
    world_size = int(world_size)
    if global_batch != expected_global_batch:
        raise ValueError(
            f"Step 2 requires global batch {expected_global_batch}, got {global_batch}"
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if global_batch % world_size != 0:
        raise ValueError(
            f"global batch {global_batch} is not divisible by world_size {world_size}"
        )
    per_rank = global_batch // world_size
    if per_rank * world_size != expected_global_batch:
        raise ValueError(
            f"per-rank batch {per_rank} x world_size {world_size} must equal "
            f"{expected_global_batch}"
        )
    return per_rank


def require_resume_contract(checkpoint, expected_protocol, schedule_epochs, path):
    protocol = checkpoint.get("step2_protocol")
    horizon = checkpoint.get("schedule_epochs")
    if protocol != expected_protocol or horizon != schedule_epochs:
        raise RuntimeError(
            f"Refusing incompatible checkpoint {path!r}: expected "
            f"step2_protocol={expected_protocol!r} and schedule_epochs={schedule_epochs}, "
            f"found protocol={protocol!r}, schedule_epochs={horizon!r}."
        )
    if checkpoint.get("canonical") is not True or checkpoint.get("resumable") is not True:
        raise RuntimeError(
            f"Refusing noncanonical or nonresumable checkpoint {path!r}: "
            f"canonical={checkpoint.get('canonical')!r}, "
            f"resumable={checkpoint.get('resumable')!r}."
        )


def require_finite_loss(loss, context="loss"):
    finite = torch.isfinite(loss.detach()).all().to(dtype=torch.int32)
    if _distributed():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() != 1:
        raise FloatingPointError(f"Non-finite {context} detected on at least one rank")


@torch.no_grad()
def validate_epoch_state(model, epoch_loss, sync_atol=1e-6):
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    require_finite_loss(torch.as_tensor(epoch_loss, device=device), "epoch loss")

    finite = torch.ones((), dtype=torch.int32, device=device)
    for parameter in parameters:
        finite.mul_(torch.isfinite(parameter).all().to(dtype=torch.int32))
    if _distributed():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item() != 1:
        raise FloatingPointError("Non-finite model parameter detected on at least one rank")

    max_difference = torch.zeros((), dtype=torch.float32, device=device)
    if _distributed():
        for parameter in parameters:
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            if parameter.numel():
                difference = (parameter.detach() - reference).abs().max().float()
                max_difference = torch.maximum(max_difference, difference)
        dist.all_reduce(max_difference, op=dist.ReduceOp.MAX)
        if max_difference.item() > sync_atol:
            raise RuntimeError(
                f"Rank parameter synchronization check failed: "
                f"max_abs_difference={max_difference.item():.3e} > {sync_atol:.3e}"
            )
    return max_difference.item()


def _atomic_path(path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f".{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def atomic_torch_save(state, path):
    path = os.path.abspath(path)
    temporary = _atomic_path(path)
    try:
        torch.save(state, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _atomic_write_text(text, path):
    path = os.path.abspath(path)
    temporary = _atomic_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_latest_checkpoint(state, save_dir):
    if state.get("canonical") is not True or state.get("resumable") is not True:
        raise ValueError("canonical latest publication requires a resumable canonical state")
    latest = atomic_torch_save(state, os.path.join(save_dir, "checkpoint_latest.pth"))
    _atomic_write_text(latest + "\n", os.path.join(save_dir, "latest_valid_checkpoint.txt"))
    return latest


def publish_pilot_checkpoint(state, save_dir):
    if state.get("canonical") is not False or state.get("resumable") is not False:
        raise ValueError("pilot publication requires an explicit noncanonical state")
    return atomic_torch_save(
        state, os.path.join(save_dir, "pilot_checkpoint_nonresumable.pth")
    )
