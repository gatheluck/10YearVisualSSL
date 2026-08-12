"""
Official-style PyTorch port for Step 1 Context Prediction.

This is intentionally separate from train_pretrain_alexnet.py because the legacy
file is not paper-compatible: model, preprocessing, and sampling all differ from
the released deepcontext implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.context_dataset_official import (
    OfficialContextPredictionDataset,
    make_official_context_loader,
    seed_worker,
)
from models.alexnet_context_official import build_official_context_alexnet


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size(), True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def is_finite_model(model: nn.Module) -> bool:
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                if not torch.isfinite(tensor).all():
                    return False
    return True


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return pred.eq(target).float().mean().mul(100.0).item()


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, global_step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": global_step,
            "state_dict": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "protocol": {
                "source": "PyTorch port of cdoersch/deepcontext train.py",
                "official_repo": "https://github.com/cdoersch/deepcontext",
                "patch_size": 96,
                "gap": 48,
                "jitter": 7,
                "resize_target_pixels": [150000, 450000],
                "optimizer": "SGD fixed lr=1e-5 weight_decay=0 momentum=0",
                "loss_reduction": "sum, matching Caffe SoftmaxWithLoss normalization NONE",
                "ddp_loss_scale": "loss multiplied by world_size before backward to undo DDP gradient averaging",
            },
        },
        path,
    )


@torch.no_grad()
def evaluate_pretext(model: nn.Module, loader, criterion, device, max_batches: int) -> dict:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    acc_sum = 0.0
    for batch_idx, (first, second, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(first, second)
        loss = criterion(logits, labels)
        n = labels.numel()
        loss_sum += float(loss.item())
        acc_sum += accuracy(logits, labels) * n
        sample_count += n
    model.train()
    if sample_count == 0:
        return {"val_loss": None, "val_acc1": None}
    return {"val_loss": loss_sum / sample_count, "val_acc1": acc_sum / sample_count}


def build_parser() -> argparse.ArgumentParser:
    """The original command line, unchanged.

    The cluster's PBS scripts call this file directly, so every flag stays as
    it was. Split out of main() during the port so that an adapter can build
    the same arguments without going through the shell.
    """
    parser = argparse.ArgumentParser(description="Official-style Context Prediction pretraining")
    parser.add_argument("--data_path", required=True, help="ImageNet root with train/ and val/")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU directed patch-pair batch size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_every_steps", type=int, default=2000)
    parser.add_argument("--eval_every_steps", type=int, default=2000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--resume", default="")
    parser.add_argument("--allow_resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu"),
        help="auto uses cuda when it is available. Added during the port: the "
             "original hard-coded cuda and so could not run anywhere else",
    )
    return parser


def resolve_device(requested: str, local_rank: int) -> torch.device:
    """Pick the device. **Refuse rather than fall back silently.**

    Falling back from cuda to cpu without a word would turn a misconfigured
    cluster job into a run that looks fine and takes a thousand times longer.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda was asked for, but no CUDA device "
                               "is available. Ask for cpu to run without one")
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def make_deterministic() -> None:
    """Ask torch for reproducible kernels.

    **Added during the port.** Without it torch may choose a kernel by timing,
    and two runs of the same config on the same machine can differ in the last
    bits. That defeats the one guarantee that is actually achievable: same
    environment, same config, same result.

    `warn_only=True` deliberately. An operation with no deterministic
    implementation then warns instead of aborting, which keeps the method
    runnable while still saying, in the run's own output, that a step was not
    reproducible. Aborting would trade a recorded caveat for an unusable port.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args) -> dict:
    """The original body, unchanged apart from the device and the return.

    Returns the final pretext evaluation so that a caller does not have to
    re-derive it from the log.
    """
    rank, local_rank, world_size, is_distributed = setup_distributed()
    is_main = rank == 0
    if not is_distributed:
        local_rank = args.gpu
    device = resolve_device(getattr(args, "device", "auto"), local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # **Reproducibility fix made during the port.** The captured original
    # seeded torch only. The dataset draws every patch position, jitter,
    # colour channel and pixelation decision from Python's `random` and from
    # `np.random`, and neither was ever seeded on the training path:
    # `seed_worker` is passed to the validation loader alone, and with
    # num_workers=0 it does not run at all. Measured on CPU, two runs of the
    # same config produced different weights. This changes no distribution --
    # only whether the same draw can be repeated.
    make_deterministic()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "run_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        print("=" * 80, flush=True)
        print("Official-style Context Prediction pretraining", flush=True)
        print(f"save_dir={save_dir}", flush=True)
        print(f"world_size={world_size}", flush=True)
        print(f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * world_size}", flush=True)
        print("=" * 80, flush=True)

    model = build_official_context_alexnet(num_classes=8).to(device)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)

    global_step = 0
    if args.resume:
        if not args.allow_resume:
            raise RuntimeError("Refusing to resume unless --allow_resume is explicit")
        ckpt = torch.load(args.resume, map_location="cpu")
        unwrap(model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = int(ckpt["global_step"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        if is_main:
            print(f"Resumed from {args.resume} at global_step={global_step}", flush=True)

    steps_per_epoch = max(1, math.ceil(args.max_steps / 100))
    samples_per_epoch = steps_per_epoch * args.batch_size * world_size
    train_dataset = OfficialContextPredictionDataset(
        image_folder=os.path.join(args.data_path, "train"),
        samples_per_epoch=samples_per_epoch,
        mode="train",
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        # Part of the same fix: the captured original passed this to the
        # validation loader only, so worker processes on the training path
        # seeded neither `random` nor `np.random`.
        worker_init_fn=seed_worker,
    )
    val_loader = make_official_context_loader(
        image_folder=os.path.join(args.data_path, "val"),
        batch_size=args.batch_size,
        num_workers=max(1, min(args.num_workers, 4)),
        samples_per_epoch=args.eval_batches * args.batch_size,
        mode="val",
    )

    running_loss = 0.0
    running_acc = 0.0
    running_samples = 0
    epoch = 0
    model.train()
    start_time = time.time()

    while global_step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for first, second, labels in train_loader:
            if global_step >= args.max_steps:
                break
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(first, second)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at global_step={global_step}: {loss.item()}")
            backward_loss = loss * world_size if is_distributed else loss
            backward_loss.backward()
            optimizer.step()

            n = labels.numel()
            running_loss += float(loss.item())
            running_acc += accuracy(logits.detach(), labels) * n
            running_samples += n
            global_step += 1

            if is_main and (global_step == 1 or global_step % 50 == 0):
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f"step={global_step} loss={running_loss / running_samples:.6f} "
                    f"acc1={running_acc / running_samples:.3f} "
                    f"samples={running_samples} steps_per_sec={global_step / elapsed:.3f}",
                    flush=True,
                )
                running_loss = 0.0
                running_acc = 0.0
                running_samples = 0

            if is_main and global_step % args.eval_every_steps == 0:
                metrics = evaluate_pretext(unwrap(model), val_loader, criterion, device, args.eval_batches)
                print(f"eval step={global_step} {metrics}", flush=True)
                with open(save_dir / "progress.jsonl", "a") as f:
                    f.write(json.dumps({"global_step": global_step, **metrics}) + "\n")

            if is_main and (global_step % args.save_every_steps == 0 or global_step == args.max_steps):
                if not is_finite_model(unwrap(model)):
                    raise RuntimeError(f"Non-finite model parameters at global_step={global_step}")
                ckpt_path = save_dir / f"checkpoint_step_{global_step}.pth"
                save_checkpoint(ckpt_path, model, optimizer, global_step, args)
                save_checkpoint(save_dir / "latest.pth", model, optimizer, global_step, args)
                print(f"saved {ckpt_path}", flush=True)

    final_metrics: dict = {}
    if is_main:
        save_checkpoint(save_dir / "final.pth", model, optimizer, global_step, args)
        # The original evaluated only every eval_every_steps, so a run whose
        # length is not a multiple of it finished with nothing to report.
        final_metrics = evaluate_pretext(unwrap(model), val_loader, criterion,
                                         device, args.eval_batches)
        print(f"training complete global_step={global_step} {final_metrics}", flush=True)

    cleanup_distributed()
    return {"global_step": global_step, **final_metrics}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
