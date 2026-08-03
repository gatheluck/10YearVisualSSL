"""MAR step 1: masked autoregressive pretraining in VAE-latent space
(Li et al., 2024; arXiv:2406.11838).

The model is the **pinned upstream**, imported from `third_party/mar` -- it is
never copied here (DESIGN section 1). What lives in this file is the thin
training loop the port owns, for two measured reasons the upstream engine
cannot be called directly:

  - `third_party/mar/engine_mar.py` imports `torch_fidelity` and `cv2` at module
    load and calls `torch.cuda.synchronize()` in its inner loop, so it neither
    imports nor runs without a GPU. The loop here is the cached-latent path of
    its `train_one_epoch`, with the DDP / AMP / EMA / FID machinery removed, so
    the same step runs on a CPU or a GPU unchanged.
  - the upstream model's own `forward` created tensors with a hard-coded
    `.cuda()` (`sample_orders`, `forward_mae_encoder`), so it raised on a
    machine with no GPU. That is fixed in the pinned **fork** (submodule+patch,
    DESIGN section 2.8); the patched commit is recorded in `provenance.json`.

Trained in **VAE-latent space**: a step reads pre-encoded latent *moments* (the
upstream `CachedFolder` cached-latent format), samples a latent from them, and
asks the model for its masked-autoregressive loss. No VAE is loaded, so no
weights are downloaded -- which is what lets a hermetic smoke run offline.

Changed during the port, and recorded in `provenance.json`:

  - **the device is resolved instead of assumed** (`resolve_device`); asking for
    `cuda` where there is none is refused rather than served a CPU in silence.
  - **`main()` splits into `build_parser()` and `run(args, config)`,** which
    returns the epoch's pretext loss and the number of epochs.
  - **the run is seeded through `make_deterministic`.**
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
REPO_ROOT = ROOT.parent.parent
UPSTREAM = REPO_ROOT / "third_party" / "mar"

# engine_mar normalises the latent std to 1 for this tokenizer; the same
# constant has to scale the sampled latent here (engine_mar.train_one_epoch).
LATENT_SCALE = 0.2325

# Top-level package names the pinned upstream defines. Other methods define
# their own `models` (and could define `util`), and `sys.modules` keeps only
# the first -- so an in-process import of the upstream must drop any cached
# module of these names that is not the upstream's before importing.
_UPSTREAM_TOP = ("models", "util", "diffusion")


def _from_upstream(name: str) -> bool:
    """Whether a cached module of an upstream name is the upstream's own."""
    mod = sys.modules.get(name)
    if mod is None:
        return False
    up = str(UPSTREAM)
    origin = getattr(mod, "__file__", None) or ""
    search = list(getattr(mod, "__path__", None) or [])
    return origin.startswith(up) or any(str(p).startswith(up) for p in search)


def _load_upstream():
    """Import the pinned upstream model, collision-safe.

    The adapter runs in its own process where only the upstream is on the path,
    but the test suite holds several methods at once. Two things make the
    upstream win regardless:

      - any cached module of an upstream top-level name that is **not** the
        upstream's is dropped, so a stale `models` from another method cannot be
        returned from `sys.modules`;
      - the upstream's `models` and `util` are **namespace** packages (no
        `__init__.py`), and a regular-package `models` from another method still
        on `sys.path` would win the import whatever the path order. So they are
        bound explicitly to the upstream directory rather than left to the path
        search. `diffusion` is a regular package and resolves from the path.
    """
    import importlib.machinery
    import importlib.util

    up = str(UPSTREAM)
    if not (UPSTREAM / "models" / "mar.py").is_file():
        raise RuntimeError(
            f"the pinned upstream is not present at {UPSTREAM}. The submodule "
            "is not checked out: run `git submodule update --init`")
    for name in list(sys.modules):
        if name.split(".", 1)[0] in _UPSTREAM_TOP and not _from_upstream(name):
            del sys.modules[name]
    while up in sys.path:
        sys.path.remove(up)
    sys.path.insert(0, up)
    for pkg in ("models", "util"):
        d = UPSTREAM / pkg
        if pkg not in sys.modules and d.is_dir():
            spec = importlib.machinery.ModuleSpec(pkg, None, is_package=True)
            spec.submodule_search_locations = [str(d)]
            sys.modules[pkg] = importlib.util.module_from_spec(spec)
    from models.mar import mar_base
    from models.vae import DiagonalGaussianDistribution
    from util.loader import CachedFolder
    return mar_base, DiagonalGaussianDistribution, CachedFolder


def model_kwargs(train: dict) -> dict:
    """The arguments that build the MAR model, drawn from the resolved config.

    These fix the architecture, so `load_encoder` must build with the same set
    or `load_state_dict` reports a wall of size mismatches -- the model is not
    self-describing. Kept in one place so the trainer and the loader cannot
    drift apart.
    """
    return {
        "img_size": int(train["img_size"]),
        "vae_stride": int(train["vae_stride"]),
        "patch_size": int(train["patch_size"]),
        "vae_embed_dim": int(train["vae_embed_dim"]),
        "class_num": int(train["class_num"]),
        "buffer_size": int(train["buffer_size"]),
        "diffloss_d": int(train["diffloss_d"]),
        "diffloss_w": int(train["diffloss_w"]),
        "mask_ratio_min": float(train["mask_ratio_min"]),
        "label_drop_prob": float(train["label_drop_prob"]),
    }


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured recipe ran on CUDA unconditionally. `"cuda"` is honoured only
    when there is a GPU: falling back quietly would let a run asked for a GPU
    report success from a CPU, and the two are not the same run.
    """
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device is 'cuda' but no CUDA device is visible. Ask for "
                "'auto' to accept a CPU; asking for a GPU and getting a CPU "
                "silently would misreport what ran")
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device(f"cuda:{local_rank}"
                            if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device {spec!r}; expected auto, cuda or cpu")


def make_deterministic(seed: int) -> None:
    """Seed everything the run draws from -- python, numpy, torch.

    MAR draws from all three: `sample_orders` shuffles with numpy, the mask
    ratio is a scipy truncated normal over numpy's global state, and the
    diffusion loss samples timesteps and noise with torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAR step 1 (cached latents)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the cached-latents root (class subdirs "
                             "of .npz files with 'moments'/'moments_flip')")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured recipe assumed "
                             "CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured recipe's training, callable in process and returning its
    pretext loss and epoch count."""
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        cfg["data"]["cached_path"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    make_deterministic(int(cfg.get("seed", 42)))

    save_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    mar_base, DiagonalGaussianDistribution, CachedFolder = _load_upstream()

    model = mar_base(**cfg["model"]).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]))

    dataset = CachedFolder(cfg["data"]["cached_path"])
    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["data"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]), drop_last=True,
        generator=generator)

    grad_clip = float(cfg["training"]["grad_clip"])

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from epoch {ckpt['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("MAR  Step 1: masked autoregressive pretraining  (arXiv:2406.11838)")
    print(f"  device={device}  epochs={total_epochs}  "
          f"batch={cfg['data']['batch_size']}  latents={len(dataset)}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for moments, labels in loader:
            moments = moments.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                x = DiagonalGaussianDistribution(moments).sample().mul_(
                    LATENT_SCALE)
            loss = model(x, labels)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * labels.size(0)
            count += labels.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] pretext_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        # Only the latest is kept (overwritten each epoch): mar_base is ~700 MB
        # a copy, and a numbered checkpoint per epoch over a 400-epoch recipe is
        # a great deal of disk for no reproducibility gain -- the encoder is
        # extracted from the latest, and `--resume` reads it.
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nMAR Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
