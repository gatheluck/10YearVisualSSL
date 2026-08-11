"""MSN step 1 (Assran et al., 2022; arXiv:2204.07141), Masked Siamese Networks.

A masked-siamese port that reuses the official facebookresearch/msn model and loss
(imported from the pinned submodule third_party/msn, never copied): an anchor ViT
sees several augmented, patch-dropped views (`imgs[1:]`) and an EMA target ViT sees
one un-dropped random view (`imgs[0]`); both are compared to a set of learnable
prototypes via a soft-nearest-neighbour classifier, and the loss is the MSN
cross-entropy plus a me-max entropy regulariser:

    loss = ploss + memax_weight * me_max + ent_weight * ent

AdamW under a warmup+cosine LR schedule; the target encoder is an EMA of the
anchor. `encoder.pt` is the anchor ViT trunk (the projection head `fc` excluded).

The lab wrapper runs the official main.py under DistributedDataParallel + submitit
and evaluates with cyanure; none is needed here. This single-process port imports
the official model (src.deit), MSN loss (src.losses.init_msn_loss) and optimiser
(src.msn_train.init_opt), and owns a thin single-process loop -- the device is
resolved rather than assumed CUDA, DDP/submitit/cyanure are dropped, and the
multi-view augmentation is reimplemented (the upstream one trips the pinned Pillow;
see data/msn_data.py). AllReduce calls are inert single-process. The number of
anchor views passed to the loss is len(imgs[1:]) = rand_views + focal_views - 1.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_MSN_SUBMODULE = ROOT.parent.parent / "third_party" / "msn"
if _MSN_SUBMODULE.is_dir() and str(_MSN_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_MSN_SUBMODULE))

from models import build_msn_model                 # noqa: E402
from data import get_msn_dataloader                # noqa: E402
from src.losses import init_msn_loss               # noqa: E402
from src.msn_train import init_opt                 # noqa: E402
from src.utils import AllReduceSum                 # noqa: E402

MODEL_ARGS = ("img_size", "patch_size", "embed_dim", "depth", "num_heads",
              "mlp_ratio", "use_bn", "hidden_dim", "output_dim", "drop_path_rate")


def _model_kwargs(m: dict) -> dict:
    return {"img_size": int(m["img_size"]), "patch_size": int(m["patch_size"]),
            "embed_dim": int(m["embed_dim"]), "depth": int(m["depth"]),
            "num_heads": int(m["num_heads"]), "mlp_ratio": float(m["mlp_ratio"]),
            "use_bn": bool(m["use_bn"]), "hidden_dim": int(m["hidden_dim"]),
            "output_dim": int(m["output_dim"]),
            "drop_path_rate": float(m["drop_path_rate"])}


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


def one_hot(targets, num_classes, smoothing, device):
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    t = targets.long().view(-1, 1).to(device)
    return torch.full((len(t), num_classes), off, device=device).scatter_(1, t, on)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSN step 1")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    return parser


def run(args, config: "dict | None" = None) -> dict:
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    if getattr(args, "data_path", None):
        cfg["data"]["data_root"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    seed = int(cfg.get("seed", 0))
    make_deterministic(seed)
    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    m, d, c, t = cfg["model"], cfg["data"], cfg["criterion"], cfg["training"]

    encoder = build_msn_model(**_model_kwargs(m)).to(device)
    encoder.train()
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.eval()

    num_proto = int(c["num_proto"])
    output_dim = int(m["output_dim"])
    # Build the prototypes as a leaf Parameter already on the device: wrapping a
    # tensor in Parameter and then .to(device) makes a non-leaf on CUDA, so the
    # requires_grad assignment would fail (it is a no-op on CPU, which is why only
    # the real-CUDA path caught it).
    _sqrt_k = (1.0 / output_dim) ** 0.5
    proto_init = torch.empty(num_proto, output_dim, device=device)
    torch.nn.init.uniform_(proto_init, -_sqrt_k, _sqrt_k)
    prototypes = torch.nn.parameter.Parameter(proto_init)
    proto_labels = one_hot(torch.arange(num_proto), num_proto,
                           float(d["label_smoothing"]), device)

    rand_views = int(d["rand_views"])
    focal_views = int(d["focal_views"])
    num_anchor_views = rand_views + focal_views - 1
    msn = init_msn_loss(num_views=num_anchor_views, tau=float(c["temperature"]),
                        me_max=bool(c["me_max"]), return_preds=True)

    loader, dataset = get_msn_dataloader(
        d["data_root"], batch_size=int(t["batch_size"]),
        rand_size=int(m["img_size"]), focal_size=int(d["focal_size"]),
        rand_crop_scale=tuple(d["rand_crop_scale"]),
        focal_crop_scale=tuple(d["focal_crop_scale"]),
        color_jitter=float(d["color_jitter"]), rand_views=rand_views,
        focal_views=focal_views, num_workers=int(d["num_workers"]), seed=seed)

    epochs = int(t["epochs"])
    ipe = max(1, len(loader))
    encoder, optimizer, scheduler, wd_scheduler = init_opt(
        encoder=encoder, iterations_per_epoch=ipe, start_lr=float(t["start_lr"]),
        ref_lr=float(t["lr"]), warmup=int(t["warmup"]), num_epochs=epochs,
        prototypes=prototypes, wd=float(t["weight_decay"]),
        final_wd=float(t["final_weight_decay"]), final_lr=float(t["final_lr"]))

    _start_m, _final_m = float(t["ema_start"]), float(t["ema_final"])
    _total = int(ipe * epochs * 1.25)
    _incr_m = (_final_m - _start_m) / max(1, ipe * epochs * 1.25)
    momentum_scheduler = (_start_m + _incr_m * i for i in range(_total + 1))
    _start_T, _final_T = float(c["start_sharpen"]), float(c["final_sharpen"])
    _incr_T = (_final_T - _start_T) / max(1, ipe * epochs * 1.25)
    sharpen_scheduler = (_start_T + _incr_T * i for i in range(_total + 1))

    patch_drop = float(d["patch_drop"])
    memax_weight = float(c["memax_weight"])
    ent_weight = float(c["ent_weight"])
    use_ent = bool(c["use_ent"])
    use_sinkhorn = bool(c["use_sinkhorn"])
    clip_grad = float(t["clip_grad"])

    print("=" * 72)
    print("MSN  Step 1: masked siamese networks + prototypes  (arXiv:2204.07141)")
    print(f"  device={device}  epochs={epochs}  images={len(dataset)}  "
          f"embed_dim={m['embed_dim']}  num_proto={num_proto}  "
          f"anchor_views={num_anchor_views}")
    print("=" * 72)

    final_loss = None
    for epoch in range(epochs):
        running, count = 0.0, 0
        for udata, _labels in loader:
            imgs = [u.to(device, non_blocking=True) for u in udata]
            optimizer.zero_grad(set_to_none=True)
            h, z = encoder(imgs[1:], return_before_head=True, patch_drop=patch_drop)
            with torch.no_grad():
                h_t, _ = target_encoder(imgs[0], return_before_head=True)
            anchor_views, target_views = z.float(), h_t.detach().float()
            T = next(sharpen_scheduler)
            ploss, me_max, ent, _logs, _ = msn(
                T=T, use_sinkhorn=use_sinkhorn, use_entropy=use_ent,
                anchor_views=anchor_views, target_views=target_views,
                proto_labels=proto_labels, prototypes=prototypes)
            loss = ploss + memax_weight * me_max + ent_weight * ent
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"MSN loss became non-finite: {loss.item()}")
            scheduler.step()
            wd_scheduler.step()
            loss.backward()
            if prototypes.grad is not None:
                prototypes.grad.data = AllReduceSum.apply(prototypes.grad.data)
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
            optimizer.step()
            with torch.no_grad():
                mm = next(momentum_scheduler)
                for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                    pk.data.mul_(mm).add_((1.0 - mm) * pq.detach().data)
            running += loss.item() * imgs[0].size(0)
            count += imgs[0].size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] msn_loss={final_loss}  ploss={ploss.item():.4f}"
              f"  me_max={me_max.item():.4f}")
        torch.save({"epoch": epoch, "model_state_dict": encoder.state_dict(),
                    "prototypes": prototypes.detach().cpu(), "loss": final_loss,
                    "config": cfg},
                   os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nMSN Step 1 training complete!")
    ran = epochs > 0 and final_loss is not None
    return {"epochs": epochs, "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
