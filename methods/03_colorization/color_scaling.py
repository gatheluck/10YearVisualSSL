"""Fresh indexed colorization scaling runs; original continuation is not inferred."""

import argparse
import importlib
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from downstream.contract import write_metrics
from downstream.extended_distributed import cli, session
from downstream.extended_resume import _cpu, _rng
from downstream.indexed_tar import IndexedTarDataset, freeze, validate_output

_model = importlib.import_module("methods.03_colorization.models.vit_colorization")
_data = importlib.import_module("methods.03_colorization.data.colorization_dataset")
_quant = importlib.import_module("methods.03_colorization.data.ab_quantization")
PRESETS = {
    "vit_base_patch16_224": (768, 12, 12),
    "vit_large_patch16_224": (1024, 24, 16),
}
CROP, RESIZE = 224, 256


def build_model(arch):
    width, depth, heads = PRESETS[arch]
    return _model.ColorizationViT(
        img_size=CROP,
        embed_dim=width,
        depth=depth,
        num_heads=heads,
        native_initialization=True,
    )


def schedule(epoch, batch):
    peak, floor = 0.00015 * batch / 256, 1e-6 * batch / 256
    factor = (
        epoch / 10
        if epoch < 10
        else 0.5 * (1 + float(torch.cos(torch.tensor((epoch - 10) / 80 * 3.14159265))))
    )
    return floor + (peak - floor) * factor


class Inputs:
    def __init__(self, training=True):
        steps = [
            transforms.Resize(RESIZE),
            transforms.RandomCrop(CROP) if training else transforms.CenterCrop(CROP),
        ]
        if training:
            steps.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(steps)

    def __call__(self, image):
        light, ab = _data._lab_components(self.transform(image), native_eval=True)
        if not np.isfinite(light).all() or not np.isfinite(ab).all():
            raise ValueError("nonfinite Lab input")
        return (
            torch.from_numpy((light / 100).astype(np.float32)).unsqueeze(0),
            torch.from_numpy(_quant.quantize_ab_fast(ab)).long(),
        )


def class_weights(dataset):
    indices = np.random.RandomState(42).choice(
        len(dataset), min(10000, len(dataset)), replace=False
    )
    counts = np.zeros(313, dtype=np.float64)
    for index in indices:
        (_, target), _, _ = dataset[int(index)]
        bins, values = np.unique(target.numpy(), return_counts=True)
        counts[bins] += values
    probs = counts / (counts.sum() + 1e-8)
    weights = 1 / (0.5 * probs + 0.5 / 313 + 1e-8)
    return torch.from_numpy((weights / weights.mean()).astype(np.float32))


def train_step(model, optimizer, images, targets, weights, scaler):
    # Shared rank error propagation happens before any gradient collective.
    from downstream.extended_distributed import Session

    context = Session(images.device)

    def finite(value):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite color training state")

    optimizer.zero_grad(set_to_none=True)
    with (
        torch.autocast("cuda", dtype=torch.float16)
        if scaler is not None
        else nullcontext()
    ):
        loss = nn.functional.cross_entropy(model(images), targets, weight=weights)
    context.call(lambda: finite(loss))
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    context.call(lambda: finite(norm))
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return loss.item()


def validate(cfg):
    if set(cfg) != {
        "device",
        "seed",
        "model",
        "precision",
        "prepared_root",
        "validation_root",
        "epochs",
        "microbatch",
        "num_workers",
    }:
        raise ValueError("color scaling requires exactly the documented fresh-run keys")
    for key, minimum in (
        ("seed", 0),
        ("epochs", 1),
        ("microbatch", 1),
        ("num_workers", 0),
    ):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError("invalid color scaling " + key)
    if cfg["epochs"] > 90 or cfg["model"] not in PRESETS:
        raise ValueError("supported preset and at most 90 epochs required")
    if cfg["device"] not in ("cpu", "cuda") or cfg["precision"] not in (
        "float32",
        "float16",
    ):
        raise ValueError("explicit color device and precision required")
    if cfg["precision"] == "float16" and cfg["device"] != "cuda":
        raise ValueError("float16 requires CUDA")
    for key in ("prepared_root", "validation_root"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError("explicit input root required")


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
                out / "membership/TRAIN_MANIFEST.jsonl", cfg["prepared_root"], Inputs()
            )
            sampler = (
                torch.utils.data.DistributedSampler(
                    data,
                    num_replicas=context.world,
                    rank=context.rank,
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
                drop_last=True,
                num_workers=cfg["num_workers"],
                generator=torch.Generator().manual_seed(seed),
            )
            if not len(loader):
                raise ValueError("population smaller than physical batch")
            return build_model(cfg["model"]).to(context.device), loader, sampler

        raw, loader, sampler = context.call(setup)
        weights = context.call(
            lambda: class_weights(
                IndexedTarDataset(
                    out / "membership/TRAIN_MANIFEST.jsonl",
                    cfg["prepared_root"],
                    Inputs(False),
                )
            ),
            leader=True,
        )
        context.call(lambda: torch.save(weights, out / "class_weights.pt"), leader=True)
        weights = weights.to(context.device)
        model = (
            nn.parallel.DistributedDataParallel(
                raw,
                device_ids=[context.device.index]
                if context.device.type == "cuda"
                else None,
            )
            if context.world > 1
            else raw
        )
        batch = cfg["microbatch"] * context.world
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=schedule(0, batch),
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )
        scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "float16" else None
        updates = 0
        model.train()
        for epoch in range(cfg["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for group in optimizer.param_groups:
                group["lr"] = schedule(epoch, batch)
            iterator = iter(loader)
            total = 0.0
            for _ in range(len(loader)):

                def read(iterator=iterator):
                    (images, targets), _, _ = next(iterator)
                    return images.to(context.device), targets.to(context.device)

                images, targets = context.call(read)
                total += train_step(model, optimizer, images, targets, weights, scaler)
                updates += 1
            loss = torch.tensor(
                total / len(loader), dtype=torch.float64, device=context.device
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
                    "format": "color_scaling_training_v1",
                    "epoch": epoch + 1,
                    "global_step": updates,
                    "model": _cpu(raw.state_dict()),
                    "optimizer": _cpu(optimizer.state_dict()),
                    "scaler": scaler.state_dict() if scaler else None,
                    "rng": states,
                    "config": cfg,
                    "membership": meta,
                    "class_weights": weights.cpu(),
                }
                temporary = out / "checkpoint_latest.tmp"
                torch.save(blob, temporary)
                temporary.replace(out / "checkpoint_latest.pth")
                export = {
                    "frozen_encoder_version": 1,
                    "kind": "colorization_scaling",
                    "arch": cfg["model"],
                    "state_dict": _cpu(raw.encoder.state_dict()),
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
                        global_batch_size=batch,
                        schedule_epochs=90,
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
    return cli(
        data, args.out, run, "colorization_scaling", method_ref="colorization_scaling"
    )


if __name__ == "__main__":
    raise SystemExit(main())
