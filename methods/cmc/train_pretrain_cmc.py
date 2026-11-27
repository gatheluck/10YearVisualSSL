"""Visual CMC step 1 (Tian et al., 2019), the paper-faithful AlexNet path.

A self-contained re-implementation, ported from the lab's own CMC code. An RGB
image is converted to CIE Lab and split into its L and ab views; a two-branch
half-size AlexNet maps each view to an L2-normalised embedding; an NCE loss over
two momentum memory banks (one per view, cross-view scored) pulls an image's two
views together and apart from K negatives.

The lab wrapper trains under DistributedDataParallel with AMP and logs to
TensorBoard; none is needed for a single-process run, so the loop here is
single-process fp32, the device is resolved rather than assumed CUDA, TensorBoard
is dropped, and NCE negatives come from within the batch. `encoder.pt` is the
two-branch encoder; the memory banks (in the NCEAverage module) are excluded.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_cmc_from_config          # noqa: E402
from data import CMCDataset                        # noqa: E402
from nce import NCEAverage, NCECriterion           # noqa: E402

MODEL_KEYS = ("feat_dim",)


def model_config(model: dict) -> dict:
    """The model sub-config build_cmc_from_config reads. load_encoder rebuilds
    with the same set, so it lives here once."""
    return {"model": {k: model[k] for k in MODEL_KEYS}}


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


def adjust_lr(optimizer, epoch: int, base_lr: float, decay_epochs, decay_rate):
    steps = sum(epoch >= m for m in decay_epochs)
    lr = base_lr * (decay_rate ** steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual CMC step 1 (AlexNet)")
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
    seed = int(cfg.get("seed", 42))
    make_deterministic(seed)

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m = cfg["model"]
    n = cfg["nce"]
    d = cfg["data"]
    t = cfg["training"]

    model = build_cmc_from_config(model_config(m)).to(device)
    model.train()

    dataset = CMCDataset(d["data_root"], mode="train",
                         image_size=int(d["img_size"]),
                         crop_low=float(d["crop_low"]))
    n_data = len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(t["batch_size"]), shuffle=True,
        num_workers=int(t["num_workers"]), drop_last=True,
        generator=torch.Generator().manual_seed(seed))

    contrast = NCEAverage(
        feat_dim=int(m["feat_dim"]), n_data=n_data,
        K=int(n["num_negatives"]), T=float(n["temperature"]),
        momentum=float(n["nce_momentum"])).to(device)
    contrast.multinomial.to(device)
    criterion_l = NCECriterion(n_data).to(device)
    criterion_ab = NCECriterion(n_data).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(t["lr"]), momentum=float(t["momentum"]),
        weight_decay=float(t["weight_decay"]))
    decay_epochs = [int(e) for e in t["lr_decay_epochs"]]
    decay_rate = float(t["lr_decay_rate"])

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        contrast.load_state_dict(state["contrast_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(t["epochs"])
    print("=" * 70)
    print("Visual CMC  Step 1: two-branch AlexNet + NCE memory bank  "
          "(arXiv:1906.05849)")
    print(f"  device={device}  epochs={total_epochs}  images={n_data}  "
          f"feat_dim={m['feat_dim']}  K={n['num_negatives']}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        adjust_lr(optimizer, epoch, float(t["lr"]), decay_epochs, decay_rate)
        running, count = 0.0, 0
        for imgs, _, idx in loader:
            imgs = imgs.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            feat_l, feat_ab = model(imgs)
            out_l, out_ab = contrast(feat_l, feat_ab, idx)
            loss = criterion_l(out_l) + criterion_ab(out_ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] nce_loss={final_loss}")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "contrast_state_dict": contrast.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": final_loss, "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nVisual CMC Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
