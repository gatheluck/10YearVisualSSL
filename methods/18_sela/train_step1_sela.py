"""SeLa step 1 (Asano et al., ICLR 2020), the ResNet path.

A self-contained re-implementation, ported from the lab's own SeLa code. A ResNet
backbone with `num_heads` linear prototype heads is trained to predict pseudo-
labels; the labels start balanced (a shuffled equipartition) and are recomputed
with Sinkhorn-Knopp optimal transport at the official `nopts` scheduled points
inside the batch loop; training is cross-entropy on the hard targets, averaged
over the heads. The heads are never reset (unlike DeepCluster).

The lab wrapper trains under DataParallel/DistributedDataParallel with AMP and
logs to TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, and
TensorBoard/tqdm are dropped. `encoder.pt` is the backbone; the prototype heads
are excluded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import create_resnet_sela               # noqa: E402
from data import create_indexed_train_loader        # noqa: E402
from utils import compute_hard_sinkhorn_assignments  # noqa: E402


def model_config(model: dict) -> dict:
    """The kwargs needed to rebuild the model for loading. Only arch shapes the
    backbone (all that encoder.pt carries); k / num_heads shape the excluded
    prototype heads, so load_encoder can rebuild with any."""
    return {"arch": str(model["arch"])}


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; getting a CPU silently would misreport "
                "what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def get_epoch_lr(epoch: int, config: dict) -> float:
    """Official self-label ResNet uses step LR drops every 150 epochs."""
    training = config["training"]
    base_lr = training["learning_rate"]
    schedule = training.get("lr_schedule", "step")
    if schedule == "step":
        step_size = training.get("lr_step_size", 150)
        gamma = training.get("lr_gamma", 0.1)
        return base_lr * (gamma ** (epoch // step_size))
    if schedule == "cosine":
        total_epochs = training["epochs"]
        min_lr = training.get("min_lr", 1e-4)
        progress = epoch / total_epochs
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    raise ValueError(f"Unsupported lr_schedule: {schedule}")


def initialize_balanced_labels(num_heads, num_samples, num_clusters, seed=31):
    """SeLa starts from balanced shuffled labels for each head."""
    rng = np.random.default_rng(seed)
    labels = torch.empty(num_samples, num_heads, dtype=torch.long)
    base = np.arange(num_samples, dtype=np.int64) % int(num_clusters)
    for head_idx in range(num_heads):
        labels[:, head_idx] = torch.from_numpy(rng.permutation(base))
    return labels


def build_optimize_times(num_epochs, num_samples, nopts):
    """Sample-count thresholds at which Sinkhorn re-runs, consumed from the end
    during the batch loop (mirrors yukimasano/self-label)."""
    times = [(num_epochs + 2) * num_samples]
    times += ((num_epochs + 1.01) * num_samples *
              (np.linspace(0, 1, nopts) ** 2)[::-1]).tolist()
    return times


def hard_pseudo_label_loss(logits, targets, temperature=1.0):
    """Cross-entropy against hard Sinkhorn targets, averaged over heads.

    logits:  (B, K) for one head, or (B, H, K) for multiple heads.
    targets: (B,) for one head, or (B, H) for multiple heads.
    """
    logits = logits / temperature
    if logits.dim() == 2:
        return F.cross_entropy(logits, targets.view(-1).long())
    if logits.dim() != 3:
        raise ValueError(f"Expected logits (B,K) or (B,H,K), got "
                         f"{tuple(logits.shape)}")
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
    if targets.size(1) != logits.size(1):
        raise ValueError(
            f"Target heads {targets.size(1)} != logit heads {logits.size(1)}")
    losses = [F.cross_entropy(logits[:, h, :], targets[:, h].long())
              for h in range(logits.size(1))]
    return torch.stack(losses).mean()


def train_one_epoch(model, train_loader, optimizer, epoch, config, device,
                    labels, optimize_times):
    model.train()
    tau = config["training"]["temperature"]
    cl = config["clustering"]
    num_heads = cl.get("num_heads", 1)
    nopts = cl.get("nopts", 100)
    max_sk_iters = cl.get("sinkhorn_max_iters", 1000)
    sk_tol = cl.get("sinkhorn_tol", 1e-1)
    sinkhorn_lamb = cl.get("lambda", cl.get("lamb", 25))
    batch_size = config["training"]["batch_size"]
    n_samples = len(train_loader.dataset)

    running, count = 0.0, 0
    for i, (images, _, indices) in enumerate(train_loader):
        niter = epoch * len(train_loader) + i
        sample_progress = niter * batch_size
        if optimize_times and sample_progress >= optimize_times[-1]:
            optimize_times.pop()
            labels = compute_hard_sinkhorn_assignments(
                model, train_loader, device, num_heads=num_heads,
                n_iters=max_sk_iters, temperature=tau,
                epsilon=cl.get("epsilon", 0.04), lamb=sinkhorn_lamb,
                tol=sk_tol, verbose=True)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            if labels.size(0) != n_samples:
                raise RuntimeError(
                    f"SeLa assignment size mismatch: {labels.size(0)} != "
                    f"{n_samples}")
            labels = labels.cpu().long()

        images = images.to(device, non_blocking=True)
        hard_labels = labels[indices].to(device, non_blocking=True)
        logits = model(images)
        loss = hard_pseudo_label_loss(logits, hard_labels, tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        count += images.size(0)

    avg = running / count if count else None
    return avg, labels, optimize_times


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SeLa step 1 (ResNet)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the lab wrapper assumed CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 31))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    cl = cfg["clustering"]
    t = cfg["training"]
    d = cfg["data"]

    model = create_resnet_sela(cfg).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(t["learning_rate"]), momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]))

    train_loader, train_dataset = create_indexed_train_loader(
        d["data_root"], image_size=int(d["image_size"]),
        batch_size=int(t["batch_size"]), num_workers=int(t["num_workers"]),
        seed=seed)

    num_heads = int(cl.get("num_heads", 1))
    labels = initialize_balanced_labels(
        num_heads, len(train_dataset), int(cl["k"]), seed=seed)
    total_epochs = int(t["epochs"])
    optimize_times = build_optimize_times(
        total_epochs, len(train_dataset), int(cl.get("nopts", 100)))

    print("=" * 70)
    print("SeLa  Step 1: ResNet + Sinkhorn-Knopp self-labelling  "
          "(arXiv:1911.05371)")
    print(f"  device={device}  epochs={total_epochs}  images={len(train_dataset)}"
          f"  K={cl['k']}  heads={num_heads}  arch={cfg['model']['arch']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(total_epochs):
        lr = get_epoch_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        final_loss, labels, optimize_times = train_one_epoch(
            model, train_loader, optimizer, epoch, cfg, device, labels,
            optimize_times)
        print(f"  [{epoch}] lr={lr:.6f} sela_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "label_assignments": labels,
                    "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nSeLa Step 1 training complete!")
    ran = total_epochs > 0 and final_loss is not None
    return {"epochs": total_epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
