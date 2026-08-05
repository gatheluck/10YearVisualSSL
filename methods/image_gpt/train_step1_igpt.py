"""image_gpt step 1: generative pretraining from pixels (Chen et al., 2020;
arXiv:2006.14671).

A self-contained re-implementation, ported from the lab's ARSSL inline model
(`src/models/train_igpt_scratch.py`). An image is quantised to colour-cluster
tokens and a causal transformer is trained to predict the next token
(cross-entropy over the colour vocabulary).

What the port owns, and what it drops. The lab wrapper trains under
`init_distributed_mode`/`DistributedDataParallel` and a `torch.cuda.amp` context;
none of that is needed for a single-process run, so the loop here is
single-process and full-precision, and the device is **resolved** rather than
assumed CUDA -- so the same step runs on a CPU or a GPU unchanged.

The colour clusters are computed once, here, from the training images and saved
next to the checkpoint, because the linear probe must quantise with the *same*
clusters the model was trained on.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_igpt                       # noqa: E402
from quantize import kmeans, quantize_images        # noqa: E402

# How many images and pixels the clusters are fit on. Bounded so a real run does
# not read all of ImageNet into memory; the whole tiny dataset fits under these
# in the smoke.
CLUSTER_IMAGES = 512
CLUSTER_PIXELS = 100_000


def model_kwargs(model: dict) -> dict:
    """The architecture arguments `build_igpt` takes. `load_encoder` builds with
    the same set, so they live here once."""
    return {
        "vocab_size": int(model["vocab_size"]),
        "img_size": int(model["img_size"]),
        "n_layer": int(model["n_layer"]),
        "n_head": int(model["n_head"]),
        "n_embd": int(model["n_embd"]),
    }


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed. `"cuda"` is honoured
    only when a GPU is visible; falling back quietly would let a run asked for a
    GPU report success from a CPU."""
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


def _transform(img_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),                       # [0, 1], the range the clusters use
    ])


def _dataset(data_root: str, img_size: int):
    from torchvision.datasets import ImageFolder
    return ImageFolder(data_root, transform=_transform(img_size))


def fit_clusters(dataset, n_clusters: int, seed: int) -> np.ndarray:
    """Colour clusters from the training images, deterministically. Images are
    read in order (bounded), their pixels pooled, subsampled with a seeded RNG,
    and k-means'd."""
    rng = np.random.RandomState(seed)
    pixels = []
    for i in range(min(len(dataset), CLUSTER_IMAGES)):
        img, _ = dataset[i]                 # [3, H, W] in [0, 1]
        pixels.append(img.permute(1, 2, 0).reshape(-1, 3).numpy())
    pixels = np.concatenate(pixels, axis=0)
    if len(pixels) > CLUSTER_PIXELS:
        pixels = pixels[rng.choice(len(pixels), size=CLUSTER_PIXELS,
                                   replace=False)]
    return kmeans(pixels, n_clusters=n_clusters, seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="image_gpt step 1 (pixel GPT)")
    parser.add_argument("--config", default="configs/step1.yaml")
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

    mk = model_kwargs(cfg["model"])
    dataset = _dataset(cfg["data"]["data_root"], mk["img_size"])
    clusters = fit_clusters(dataset, n_clusters=mk["vocab_size"], seed=seed)
    np.save(os.path.join(save_dir, "clusters.npy"), clusters)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True, num_workers=int(cfg["training"]["num_workers"]),
        drop_last=True, generator=torch.Generator().manual_seed(seed))

    model = build_igpt(**mk).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(cfg["training"]["lr"]),
                                 betas=(0.9, 0.95))
    grad_clip = float(cfg["training"]["grad_clip"])

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("image_gpt  Step 1: generative pretraining from pixels  "
          "(arXiv:2006.14671)")
    print(f"  device={device}  epochs={total_epochs}  "
          f"images={len(dataset)}  clusters={mk['vocab_size']}  "
          f"seq={mk['img_size'] ** 2}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for imgs, _labels in loader:
            tokens = quantize_images(imgs, clusters).to(device)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, mk["vocab_size"]),
                tokens[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * imgs.size(0)
            count += imgs.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] ce_loss={final_loss}")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nimage_gpt Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
