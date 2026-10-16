"""Indexed SplitBrain scaling with explicit Lab backend and source update rules."""

import argparse
import importlib
import json
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from downstream.contract import write_metrics
from downstream.extended_distributed import Session, cli, session
from downstream.extended_resume import _cpu, _rng
from downstream.indexed_tar import IndexedTarDataset, freeze, validate_output

_data = importlib.import_module("methods.08_split_brain.data.split_brain_dataset")
_model = importlib.import_module("methods.08_split_brain.models.vit_split_brain")
# Both source trainers use the same fixed update-clock schedule.
schedule = importlib.import_module(
    "methods.06_rotation_prediction.rotation_scaling"
).schedule
PRESETS = {"vit_base_patch16_224": (384, 12, 6), "vit_large_patch16_224": (512, 24, 8)}
CROP = 224


def build_model(arch):
    width, depth, heads = PRESETS[arch]
    return _model.SplitBrainViTModel(
        img_size=CROP,
        embed_dim=width,
        depth=depth,
        num_heads=heads,
        native_scaling=True,
    )


class Inputs:
    def __init__(self, backend):
        self.backend = backend
        self.transform = transforms.Compose(
            [transforms.RandomResizedCrop(CROP), transforms.RandomHorizontalFlip()]
        )
        if backend == "skimage":
            self.convert = importlib.import_module("skimage.color").rgb2lab
        elif backend == "fallback":
            self.convert = partial(_data.rgb2lab, native_eval=True)
        else:
            raise ValueError("explicit Lab backend required")

    def __call__(self, image):
        lab = self.convert(np.array(self.transform(image)))
        if not np.isfinite(lab).all():
            raise ValueError("nonfinite Lab input")
        return (
            torch.from_numpy(np.ascontiguousarray((lab[..., 0] - 50) / 50))
            .float()
            .unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(lab[..., 1:] / 128))
            .permute(2, 0, 1)
            .float(),
        )


def targets(light, ab):
    if not torch.isfinite(light).all() or not torch.isfinite(ab).all():
        raise ValueError("nonfinite SplitBrain input")
    code = torch.tensor(
        _data.load_ab_codebook(), dtype=torch.float32, device=light.device
    )
    l = (light[:, 0].float() * 50 + 50).div(2).floor().clamp(0, 49).long()
    values = ab.permute(0, 2, 3, 1).float() * 128
    a = torch.empty_like(l)
    for start in range(0, values.shape[1], 16):
        chunk = values[:, start : start + 16]
        a[:, start : start + 16] = (
            (chunk.unsqueeze(-2) - code).square().sum(-1).argmin(-1)
        )
    return l, a


def resize_target(target, size):
    return (
        nn.functional.interpolate(
            target.unsqueeze(1).float(), size=size, mode="nearest"
        )
        .long()
        .squeeze(1)
    )


def train_step(model, optimizer, batches, precision, accumulation):
    if precision not in ("float32", "bfloat16") or not 0 < len(batches) <= accumulation:
        raise ValueError("invalid SplitBrain update configuration")
    context = Session(batches[0][0].device)
    if precision == "bfloat16" and context.device.type != "cuda":
        raise ValueError("BF16 requires CUDA")

    def finite(value):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite SplitBrain training state")

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for light, ab in batches:
        lt, at = context.call(lambda light=light, ab=ab: targets(light, ab))
        with (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bfloat16"
            else nullcontext()
        ):
            ap, lp = model(light, ab)
            loss = nn.functional.cross_entropy(
                ap, resize_target(at, ap.shape[2:])
            ) + nn.functional.cross_entropy(lp, resize_target(lt, lp.shape[2:]))
        context.call(lambda loss=loss: finite(loss))
        (loss / accumulation).backward()
        losses.append(loss.detach().item())
    if len(batches) == accumulation:
        # Fail closed on invalid gradients; do not clip the source gradients.
        context.call(
            lambda: [finite(p.grad) for p in model.parameters() if p.grad is not None]
        )
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses)


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
        "accum",
        "lab_backend",
    }:
        raise ValueError("SplitBrain scaling requires exactly the documented keys")
    for key, minimum in (
        ("seed", 0),
        ("epochs", 1),
        ("microbatch", 1),
        ("num_workers", 0),
        ("accum", 1),
    ):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError("invalid SplitBrain " + key)
    if cfg["epochs"] > 90 or cfg["model"] not in PRESETS:
        raise ValueError("supported preset and at most 90 epochs required")
    if cfg["device"] not in ("cpu", "cuda") or cfg["precision"] not in (
        "float32",
        "bfloat16",
    ):
        raise ValueError("explicit device and precision required")
    if cfg["precision"] == "bfloat16" and cfg["device"] != "cuda":
        raise ValueError("BF16 requires CUDA")
    if cfg["lab_backend"] not in ("fallback", "skimage"):
        raise ValueError("explicit Lab backend required")
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
                out / "membership/TRAIN_MANIFEST.jsonl",
                cfg["prepared_root"],
                Inputs(cfg["lab_backend"]),
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
            if len(loader) < cfg["accum"]:
                raise ValueError("population smaller than effective batch")
            return build_model(cfg["model"]).to(context.device), loader, sampler

        raw, loader, sampler = context.call(setup)
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
        steps = len(loader) // cfg["accum"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=schedule(0, 90 * steps),
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )
        updates = 0
        model.train()
        for epoch in range(cfg["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            total, consumed = 0.0, 0
            while consumed < len(loader):
                count = min(cfg["accum"], len(loader) - consumed)

                def read(count=count, iterator=iterator):
                    result = []
                    for _ in range(count):
                        pair, _, _ = next(iterator)
                        result.append(tuple(value.to(context.device) for value in pair))
                    return result

                batches = context.call(read)
                total += (
                    train_step(
                        model, optimizer, batches, cfg["precision"], cfg["accum"]
                    )
                    * count
                )
                consumed += count
                if count == cfg["accum"]:
                    updates += 1
                    for group in optimizer.param_groups:
                        group["lr"] = schedule(updates, 90 * steps)
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
                    "format": "splitbrain_scaling_training_v1",
                    "epoch": epoch + 1,
                    "global_step": updates,
                    "model": _cpu(raw.state_dict()),
                    "optimizer": _cpu(optimizer.state_dict()),
                    "rng": states,
                    "config": cfg,
                    "membership": meta,
                }
                temp = out / "checkpoint_latest.tmp"
                torch.save(blob, temp)
                temp.replace(out / "checkpoint_latest.pth")
                export = {
                    "frozen_encoder_version": 1,
                    "kind": "split_brain_scaling",
                    "arch": cfg["model"],
                    "state_dict": _cpu(
                        nn.ModuleList([raw.net1.encoder, raw.net2.encoder]).state_dict()
                    ),
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
                        global_batch_size=cfg["microbatch"]
                        * context.world
                        * cfg["accum"],
                        schedule_epochs=90,
                        precision=cfg["precision"],
                        lab_backend=cfg["lab_backend"],
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
        data, args.out, run, "split_brain_scaling", method_ref="split_brain_scaling"
    )


if __name__ == "__main__":
    raise SystemExit(main())
