"""Explicit single-process FP32 IDv2 components; not canonical score runs."""
import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nce import NCELoss


class Normalize(torch.nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)


def make_bank(n, dim, temperature, momentum, negatives, profile):
    expected = .99 if profile == "ema_bank" else .5
    if momentum != expected:
        raise ValueError(f"{profile} requires bank momentum {expected}")
    if profile == "multi_prototype":
        from nce.multi_prototype import MultiPrototypeBank
        return MultiPrototypeBank(n, dim, temperature, momentum, negatives)
    return NCELoss(n, dim, temperature, momentum, negatives)


def shared_rows(bank, indices):
    negatives = torch.randint(0, bank.num_samples,
                              (len(indices), bank.num_negatives), device=indices.device)
    ids = torch.cat((indices[:, None], negatives), dim=1)
    return bank.memory[ids].detach()


def rows_loss(bank, features, rows, update_z=True):
    out = torch.exp(torch.bmm(rows, features.unsqueeze(2)).squeeze(2) / bank.temperature)
    return bank.loss_from_exp(out, update_z=update_z)


def dense_forward(model, view):
    tokens = model.encoder.forward_features(view)
    patches = tokens[:, 1:]
    if patches.size(1) < 4:
        raise ValueError("dense ID requires at least four patch tokens")
    noise = torch.rand(patches.shape[:2], device=patches.device, dtype=patches.dtype)
    ids = noise.topk(4, dim=1).indices
    selected = patches.gather(1, ids[..., None].expand(-1, -1, patches.size(2)))
    return F.normalize(model.fc(tokens[:, 0]), dim=1), F.normalize(model.fc(selected), dim=-1)


def dense_loss(bank, globals_, patches, indices):
    rows, losses = [], []
    for z in globals_:
        row = shared_rows(bank, indices)
        rows.append(row)
        losses.append(rows_loss(bank, z, row))
    patch_losses = [rows_loss(bank, p, row, update_z=False)
                   for ps, row in zip(patches, rows) for p in ps.unbind(1)]
    return .5 * sum(losses) + .1 * torch.stack(patch_losses).mean()


def view_loss(bank, features, indices, profile, epoch):
    if profile != "false_negative" or epoch < 20:
        return bank(features, indices)
    if bank.num_samples < 2:
        raise ValueError("false-negative sampling requires at least two instances")
    candidates = torch.randint(0, bank.num_samples - 1,
                                (len(indices), bank.num_negatives + 5), device=indices.device)
    candidates = candidates + (candidates >= indices[:, None]).long()
    sim = torch.bmm(bank.memory[candidates].detach(), features.unsqueeze(2)).squeeze(2)
    keep = torch.ones_like(sim, dtype=torch.bool)
    keep.scatter_(1, sim.topk(5, dim=1).indices, False)
    positive = (bank.memory[indices].detach() * features).sum(1, keepdim=True)
    retained = sim[keep].view(len(indices), bank.num_negatives)
    return bank.loss_from_exp(torch.exp(torch.cat((positive, retained), dim=1) / bank.temperature))


def koleo(features):
    features = F.normalize(features.float(), dim=-1, eps=1e-8)
    if len(features) < 2:
        return features.new_zeros(())
    with torch.no_grad():
        sim = features @ features.T
        sim.fill_diagonal_(float("-inf"))
        nearest = sim.argmax(1)
    distances = F.pairwise_distance(features, features[nearest], eps=1e-8)
    return -(distances + 1e-8).log().mean()


def train_batch(model, bank, optimizer, views, indices, profile, epoch):
    required = 6 if profile == "multicrop" else 2
    if len(views) != required:
        raise ValueError(f"{profile} requires {required} views")
    optimizer.zero_grad()
    if profile == "multicrop":
        rows = shared_rows(bank, indices)
        globals_, losses = [], []
        for i, view in enumerate(views):
            embedding = model(view)
            loss = rows_loss(bank, embedding, rows)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite IDv2 loss")
            (loss / 6).backward()
            losses.append(loss.detach())
            if i < 2:
                globals_.append(embedding.detach())
        total = torch.stack(losses).mean()
    elif profile == "dense_id":
        results = [dense_forward(model, view) for view in views]
        embeddings, patches = zip(*results)
        total = dense_loss(bank, embeddings, patches, indices)
        if not torch.isfinite(total):
            raise ValueError("nonfinite IDv2 loss")
        total.backward()
        globals_ = [z.detach() for z in embeddings]
    else:
        embeddings = [model(view) for view in views]
        total = .5 * sum(view_loss(bank, z, indices, profile, epoch) for z in embeddings)
        if profile == "koleo":
            total = total + .1 * .5 * sum(koleo(z) for z in embeddings)
        if not torch.isfinite(total):
            raise ValueError("nonfinite IDv2 loss")
        total.backward()
        globals_ = [z.detach() for z in embeddings]
    optimizer.step()
    if profile == "multi_prototype":
        bank.update_memory(globals_[0], globals_[1], indices)
    else:
        bank.update_memory(F.normalize(.5 * (globals_[0] + globals_[1]), dim=1), indices)
    return total.item()


class Views:
    def __init__(self, image_size, local_size=None):
        from data import get_instdisc_transforms
        from torchvision import transforms
        self.global_transform = get_instdisc_transforms("train", image_size)
        self.local_transform = None
        if local_size is not None:
            self.local_transform = get_instdisc_transforms("train", local_size)
            self.local_transform.transforms[0] = transforms.RandomResizedCrop(
                local_size, scale=(.05, .32))

    def __call__(self, image):
        views = [self.global_transform(image) for _ in range(2)]
        if self.local_transform:
            views.extend(self.local_transform(image) for _ in range(4))
        return views


def resume_contract(cfg):
    contract = copy.deepcopy(cfg)
    contract.pop("output")
    contract["training"].pop("epochs")
    contract["training"].pop("save_at_epochs")
    return contract


def rng_state(generator, device):
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate(), "loader": generator.get_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, generator, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["loader"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def run_components(args, cfg):
    from train_pretrain_vit_instdisc import model_kwargs, build_vit_instdisc
    from train_pretrain_instdisc import make_deterministic, resolve_device
    from data import ImageFolderWithIndex
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or (
            torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1):
        raise ValueError("IDv2 components support single-process execution only")
    profile = cfg["profile"].removeprefix("idv2_").removesuffix("_components")
    if profile not in ("twoview", "false_negative", "ema_bank", "koleo", "multicrop",
                        "multi_prototype", "dense_id"):
        raise ValueError("unsupported IDv2 profile")
    out = Path(cfg["output"]["checkpoint_dir"]).resolve()
    source = Path(args.resume).resolve() if args.resume else None
    if source and (source == out.parent or out.parent in source.parents):
        raise ValueError("output would overwrite the source checkpoint")
    if source and not source.is_file():
        raise ValueError("resume source does not exist")
    device = resolve_device(args.device)
    make_deterministic(cfg["seed"])
    tr, m = cfg["training"], cfg["model"]
    model = build_vit_instdisc(**model_kwargs(m)).to(device).train()
    if profile == "multicrop":
        from timm.layers.format import Format
        model.encoder.dynamic_img_size = True
        model.encoder.patch_embed.strict_img_size = False
        model.encoder.patch_embed.flatten = False
        model.encoder.patch_embed.output_fmt = Format.NHWC
    dataset = ImageFolderWithIndex(cfg["data"]["data_root"], transform=Views(
        cfg["data"]["img_size"], cfg["data"].get("local_size")))
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=tr["batch_size"],
        shuffle=True, num_workers=tr["num_workers"], drop_last=True, generator=generator)
    if not len(loader):
        raise ValueError("IDv2 requires at least one complete batch")
    bank = make_bank(len(dataset), m["feature_dim"], cfg["nce"]["temperature"],
                     cfg["nce"]["momentum"], cfg["nce"]["num_negatives"], profile).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    start = 0
    if source:
        state = torch.load(source, map_location="cpu", weights_only=False)
        if (state.get("component_version") != 1 or state.get("contract") != resume_contract(cfg)
                or state.get("steps_per_epoch") != len(loader) or state.get("device") != device.type):
            raise ValueError("incompatible IDv2 component checkpoint")
        start = state["epoch"] + 1
        if not 0 < start < tr["epochs"]:
            raise ValueError("resume must advance beyond the saved epoch")
        adam = state.get("optimizer_state_dict", {})
        if len(adam.get("state", {})) != len(list(model.parameters())) or any(
                not {"step", "exp_avg", "exp_avg_sq"} <= set(value)
                for value in adam.get("state", {}).values()):
            raise ValueError("incomplete IDv2 AdamW optimizer state")
        model.load_state_dict(state["model_state_dict"])
        bank.load_state_dict(state["nce_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        restore_rng(state["rng"], generator, device)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, tr["epochs"]):
        warmup = tr["warmup_epochs"]
        if epoch < warmup:
            lr = max(tr["lr"] * (epoch + 1) / max(warmup, 1), 1e-6)
        else:
            lr = tr["min_lr"] + .5 * (tr["lr"] - tr["min_lr"]) * (
                1 + math.cos(math.pi * (epoch - warmup) / max(300 - warmup, 1)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = []
        for views, idx, _ in loader:
            losses.append(train_batch(model, bank, optimizer,
                [v.to(device) for v in views], idx.to(device), profile, epoch))
        final_loss = sum(losses) / len(losses)
        state = {"component_version": 1, "contract": resume_contract(cfg),
                 "epoch": epoch, "steps_per_epoch": len(loader), "device": device.type,
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "nce_state_dict": bank.state_dict(), "rng": rng_state(generator, device),
                 "loss": final_loss, "canonical_eligible": False}
        torch.save(state, out / "checkpoint_latest.pth")
        if epoch + 1 in tr["save_at_epochs"]:
            torch.save(state, out / f"checkpoint_epoch_{epoch + 1}.pth")
    return {"epochs": tr["epochs"] - start, "final_loss": final_loss}
