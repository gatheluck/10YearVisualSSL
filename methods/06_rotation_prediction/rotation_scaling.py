"""Explicit 90-epoch indexed ImageNet scaling; component runs are not paper scores."""

import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from downstream import continuation_validation
from downstream.contract import write_metrics
from downstream.extended_distributed import cli, session
from downstream.extended_resume import _cpu, _restore_rng, _rng
from downstream.frozen_vit import validate_state
from downstream.indexed_tar import IndexedTarDataset, freeze, validate_output

PRESETS = {
    "vit_base_patch16_224": (768, 12, 12),
    "vit_large_patch16_224": (1024, 24, 16),
}


def schedule(step, total_steps):
    if step < 10000:
        return max(0.0006 * (step + 1) / 10000, 1e-8)
    progress = min(max((step - 10000) / max(total_steps - 10000, 1), 0), 1)
    return 0.000001 + 0.5 * (0.0006 - 0.000001) * (1 + math.cos(math.pi * progress))


def build_model(arch):
    from timm.models.vision_transformer import VisionTransformer

    width, depth, heads = PRESETS[arch]
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=width,
        depth=depth,
        num_heads=heads,
        num_classes=4,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    )


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ]
    )


def rotations(image):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return torch.stack(
        [normalize(torch.rot90(image, k, (1, 2))) for k in range(4)]
    ), torch.arange(4)


class RotationInputs:
    def __init__(self):
        self.transform = train_transform()

    def __call__(self, image):
        return rotations(self.transform(image))


def require_finite(value):
    ok = torch.isfinite(value).all().to(dtype=torch.int32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    if not ok.item():
        raise ValueError("nonfinite rotation loss or training state")


def train_step(model, optimizer, batches, precision, accumulation):
    if (
        precision not in ("float32", "bfloat16")
        or not batches
        or not 0 < len(batches) <= accumulation
    ):
        raise ValueError("invalid rotation accumulation or precision")
    if precision == "bfloat16" and batches[0][0].device.type != "cuda":
        raise ValueError("bfloat16 rotation training requires CUDA")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, (images, labels) in enumerate(batches):
        sync = (
            model.no_sync()
            if hasattr(model, "no_sync") and index + 1 < len(batches)
            else nullcontext()
        )
        with sync:
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bfloat16"
                else nullcontext()
            ):
                loss = nn.functional.cross_entropy(model(images), labels)
            require_finite(loss)
            (loss / accumulation).backward()
        losses.append(loss.detach())
    if len(batches) == accumulation:
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        require_finite(norm)
        optimizer.step()
    else:
        # Native tail still consumes augmentation/dropout RNG, but never updates.
        optimizer.zero_grad(set_to_none=True)
    return torch.stack(losses).mean().item()


def validate(cfg):
    keys = {
        "device",
        "seed",
        "model",
        "precision",
        "prepared_root",
        "validation_root",
        "epochs",
        "microbatch",
        "accum",
        "num_workers",
        "resume",
    }
    if set(cfg) != keys:
        raise ValueError("rotation scaling requires exactly the documented keys")
    for key, minimum in [
        ("seed", 0),
        ("epochs", 1),
        ("microbatch", 1),
        ("accum", 1),
        ("num_workers", 0),
    ]:
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError("invalid rotation scaling " + key)
    if cfg["epochs"] > 90 or cfg["model"] not in PRESETS:
        raise ValueError(
            "rotation scaling requires a supported preset and at most 90 epochs"
        )
    if cfg["device"] not in ("cpu", "cuda") or cfg["precision"] not in (
        "float32",
        "bfloat16",
    ):
        raise ValueError("explicit rotation device and precision required")
    if cfg["precision"] == "bfloat16" and cfg["device"] != "cuda":
        raise ValueError("bfloat16 rotation requires CUDA")
    for key in ("prepared_root", "validation_root"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError("invalid input root")
    if cfg["resume"] is not None and (
        not isinstance(cfg["resume"], str) or not cfg["resume"].strip()
    ):
        raise ValueError("resume must be an explicit checkpoint path or null")


def restore(path, identity, model, optimizer, loader, context, steps, target):
    if path is None:
        return 0, 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {
        "format",
        "identity",
        "epoch",
        "global_step",
        "model",
        "optimizer",
        "rng",
    }:
        raise ValueError("unsupported rotation continuation state")
    epoch = state["epoch"]
    updates = state["global_step"]
    if (
        state["format"] != "rotation_scaling_epoch_v1"
        or state["identity"] != identity
        or type(epoch) is not int
        or not 1 <= epoch < target
        or type(updates) is not int
        or updates != epoch * steps
    ):
        raise ValueError("rotation continuation identity or counters differ")
    if not isinstance(state["rng"], list) or len(state["rng"]) != context.world:
        raise ValueError("rotation continuation rank count differs")
    saved = state["optimizer"]
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("invalid optimizer state")
    continuation_validation.optimizer_state(optimizer, saved, updates)
    if len(saved["state"]) != len(list(model.parameters())):
        raise ValueError("incomplete optimizer state")
    for item in saved["state"].values():
        if item["step"].item() != updates or any(
            not torch.isfinite(v).all() for v in item.values()
        ):
            raise ValueError("nonfinite or inconsistent optimizer state")
    if any(
        group["lr"] != schedule(updates, 90 * steps) for group in saved["param_groups"]
    ):
        raise ValueError("rotation continuation learning rate differs")
    continuation_validation.rng_states(state["rng"], loader, context.device)
    validate_state(model, state["model"])
    optimizer.load_state_dict(saved)
    _restore_rng(state["rng"][context.rank], loader, context.device)
    return epoch, updates


def run(cfg, out):
    out = Path(out)
    with session(cfg["device"]) as context:
        context.call(lambda: validate(cfg))
        context.agree(cfg)
        meta = context.call(
            lambda: freeze(
                cfg["prepared_root"], cfg["validation_root"], out / "membership"
            ),
            leader=True,
        )
        seed = cfg["seed"] + context.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        def setup():
            data = IndexedTarDataset(
                out / "membership/TRAIN_MANIFEST.jsonl",
                cfg["prepared_root"],
                RotationInputs(),
            )
            generator = torch.Generator().manual_seed(seed)
            sampler = (
                torch.utils.data.DistributedSampler(
                    data,
                    num_replicas=context.world,
                    rank=context.rank,
                    shuffle=True,
                    seed=cfg["seed"],
                    drop_last=True,
                )
                if context.world > 1
                else None
            )
            loader = torch.utils.data.DataLoader(
                data,
                batch_size=cfg["microbatch"],
                sampler=sampler,
                shuffle=sampler is None,
                num_workers=cfg["num_workers"],
                drop_last=True,
                generator=generator,
            )
            steps = len(loader) // cfg["accum"]
            if not steps:
                raise ValueError("population too small for a complete update")
            model = build_model(cfg["model"]).to(context.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=schedule(0, 90 * steps),
                betas=(0.9, 0.999),
                weight_decay=0.05,
            )
            return model, optimizer, loader, sampler, steps

        raw, optimizer, loader, sampler, steps = context.call(setup)
        identity = {k: v for k, v in cfg.items() if k not in ("epochs", "resume")}
        identity.update(
            manifest=meta,
            world=context.world,
            steps=steps,
            preset=PRESETS[cfg["model"]],
            torch=str(torch.__version__),
        )
        identity = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        context.agree(identity)
        epoch_start, updates = context.call(
            lambda: restore(
                cfg["resume"],
                identity,
                raw,
                optimizer,
                loader,
                context,
                steps,
                cfg["epochs"],
            )
        )
        model = (
            torch.nn.parallel.DistributedDataParallel(
                raw,
                device_ids=[context.device.index]
                if context.device.type == "cuda"
                else None,
            )
            if context.world > 1
            else raw
        )
        model.train()
        for epoch in range(epoch_start, cfg["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            losses = []
            consumed = 0
            while consumed < len(loader):
                count = min(cfg["accum"], len(loader) - consumed)

                def read(count=count, iterator=iterator):
                    result = []
                    for _ in range(count):
                        (images, labels), _, _ = next(iterator)
                        result.append(
                            (
                                images.flatten(0, 1).to(context.device),
                                labels.flatten().to(context.device),
                            )
                        )
                    return result

                batches = context.call(read)
                losses.append(
                    (
                        train_step(
                            model, optimizer, batches, cfg["precision"], cfg["accum"]
                        ),
                        count,
                    )
                )
                consumed += count
                if count == cfg["accum"]:
                    updates += 1
                    for group in optimizer.param_groups:
                        group["lr"] = schedule(updates, 90 * steps)
            require_finite(
                torch.tensor(
                    0.0
                    if all(torch.isfinite(v).all() for v in raw.state_dict().values())
                    else float("nan"),
                    device=context.device,
                )
            )
            loss = torch.tensor(
                sum(value * count for value, count in losses) / len(loader),
                device=context.device,
                dtype=torch.float64,
            )
            states = [_rng(loader, context.device)]
            if context.world > 1:
                torch.distributed.all_reduce(loss)
                loss /= context.world
                states = [None] * context.world
                torch.distributed.all_gather_object(
                    states, _rng(loader, context.device)
                )

            def save(epoch=epoch, updates=updates, states=states):
                blob = {
                    "format": "rotation_scaling_epoch_v1",
                    "identity": identity,
                    "epoch": epoch + 1,
                    "global_step": updates,
                    "model": _cpu(raw.state_dict()),
                    "optimizer": _cpu(optimizer.state_dict()),
                    "rng": states,
                }
                temp = out / "checkpoint_latest.tmp"
                torch.save(blob, temp)
                temp.replace(out / "checkpoint_latest.pth")
                export = {
                    "frozen_encoder_version": 1,
                    "kind": "rotation_scaling",
                    "arch": cfg["model"],
                    "state_dict": {
                        k: v
                        for k, v in _cpu(raw.state_dict()).items()
                        if not k.startswith("head.")
                    },
                }
                torch.save(export, out / "encoder.pt")
                if epoch + 1 in (30, 60, 90):
                    torch.save(blob, out / f"checkpoint_epoch_{epoch + 1}.pth")
                    torch.save(export, out / f"encoder_epoch{epoch + 1}.pt")

            context.call(save, leader=True)

        def deliver():
            metrics = {
                "epochs_completed": cfg["epochs"],
                "global_step": updates,
                "final_loss": loss.item(),
            }
            write_metrics(out, metrics, {k: None for k in metrics})
            (out / "results.json").write_text(
                json.dumps(
                    dict(
                        metrics,
                        canonical_eligible=False,
                        global_source_batch_size=context.world
                        * cfg["microbatch"]
                        * cfg["accum"],
                        global_rotation_batch_size=4
                        * context.world
                        * cfg["microbatch"]
                        * cfg["accum"],
                        dropped_microbatches_per_rank=len(loader) % cfg["accum"],
                        schedule_epochs=90,
                        optimizer_steps_per_epoch=steps,
                        precision=cfg["precision"],
                        manifest_sha256=meta["manifest_sha256"],
                    ),
                    indent=2,
                )
                + "\n"
            )

        context.call(deliver, leader=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    data = Path(args.config).read_bytes()
    cfg = json.loads(data)
    validate(cfg)
    validate_output(args.out, cfg["prepared_root"], cfg["validation_root"])
    if cfg["resume"] and Path(cfg["resume"]).resolve().is_relative_to(
        Path(args.out).resolve()
    ):
        raise ValueError("output must not contain resume source")
    return cli(data, args.out, run, "rotation_scaling", method_ref="rotation_scaling")


if __name__ == "__main__":
    raise SystemExit(main())
