"""VAR step 1: next-scale autoregressive pretraining
(Tian, Jiang, Yuan, Peng & Wang, *Visual Autoregressive Modeling*, NeurIPS 2024;
arXiv:2404.02905).

The model is the **pinned upstream** under `third_party/var`, imported and never
copied (DESIGN section 1). It runs on a CPU or a GPU unmodified -- its model has
no hardcoded device -- so this is a `submodule+adapter` port with no fork.

What lives here is the thin training loop the port owns. The upstream
`trainer.py` is a DDP trainer wired to `dist.py` and a bfloat16 AMP context; none
of that is needed for a single-process, device-resolved run, so the loop is
rebuilt from the model's own forward.

VAR predicts image tokens **scale by scale**. A VQVAE tokenises an image into a
multi-scale pyramid of code indices; VAR is trained, class-conditioned, to
predict each scale from the coarser ones, with a cross-entropy loss over the
codebook. Training tokenises images inline with the VQVAE, so a real run needs
the pretrained VQVAE (`vqvae_ckpt`); the hermetic smoke builds a **tiny random
VQVAE** instead, so nothing is downloaded and it runs offline on a CPU.

Changed during the port, and recorded in `provenance.json`:

  - **the device is resolved instead of assumed** (`resolve_device`); the
    upstream reached CUDA through `dist.py`/AMP unconditionally.
  - **`main()` splits into `build_parser()` and `run(args, config)`,** returning
    the epoch's cross-entropy loss and the number of epochs.
  - **the run is seeded through `make_deterministic`.**
  - **single process, full precision**; the DDP/AMP machinery is not brought
    across, so the same step runs on a CPU or a GPU unchanged.
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
REPO_ROOT = ROOT.parent.parent
UPSTREAM = REPO_ROOT / "third_party" / "var"

# The VQVAE's fixed spatial downsample: the image side is this times the finest
# token-grid side (`patch_nums[-1]`), so the two are kept consistent here rather
# than asked for twice.
VAE_DOWNSAMPLE = 16

# Top-level module names the pinned upstream defines. Several methods define a
# `models`, and `sys.modules` keeps only the first, so an in-process import of
# the upstream must drop any cached module of these names that is not the
# upstream's before importing.
_UPSTREAM_TOP = ("models", "dist", "utils", "trainer", "train")


def _from_upstream(name: str) -> bool:
    mod = sys.modules.get(name)
    if mod is None:
        return False
    up = str(UPSTREAM)
    origin = getattr(mod, "__file__", None) or ""
    search = list(getattr(mod, "__path__", None) or [])
    return origin.startswith(up) or any(str(p).startswith(up) for p in search)


def _load_upstream():
    """Import the pinned upstream builder, collision-safe.

    The adapter runs in its own process where only the upstream is on the path,
    but the test suite holds several methods at once. Any cached module of an
    upstream top-level name that is not the upstream's is dropped, and the
    upstream is put first on `sys.path`, so `from models import build_vae_var`
    (which itself does `import dist`) resolves to `third_party/var`.
    """
    up = str(UPSTREAM)
    if not (UPSTREAM / "models" / "var.py").is_file():
        raise RuntimeError(
            f"the pinned upstream is not present at {UPSTREAM}. The submodule "
            "is not checked out: run `git submodule update --init`")
    for name in list(sys.modules):
        if name.split(".", 1)[0] in _UPSTREAM_TOP and not _from_upstream(name):
            del sys.modules[name]
    while up in sys.path:
        sys.path.remove(up)
    sys.path.insert(0, up)
    from models import build_vae_var
    return build_vae_var


def model_kwargs(train: dict) -> dict:
    """The architecture arguments `build_vae_var` takes, from the resolved
    config. `load_encoder` must build with the same set, so they live here once.

    `flash`/`fused` attention are deliberately excluded and forced off: they are
    a GPU speed path with the same result, and leaving them on the automatic
    default would make a run depend on whether flash-attention happened to be
    installed."""
    return {
        "patch_nums": tuple(int(x) for x in train["patch_nums"]),
        "V": int(train["vocab_size"]),
        "Cvae": int(train["Cvae"]),
        "ch": int(train["ch"]),
        "num_classes": int(train["num_classes"]),
        "depth": int(train["depth"]),
        "shared_aln": bool(train["shared_aln"]),
        "attn_l2_norm": bool(train["attn_l2_norm"]),
    }


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The upstream reached CUDA through `dist.py` and a `torch.cuda.amp` context.
    `"cuda"` is honoured only when there is a GPU: falling back quietly would let
    a run asked for a GPU report success from a CPU, and the two are not the same
    run."""
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
    """Seed python, numpy and torch, and ask torch for reproducible kernels.

    Seeding alone was not enough for VAR. Its attention runs through
    `scaled_dot_product_attention`, whose CPU backend and reduction order vary
    with the environment: a run that was bitwise-reproducible on one machine
    diverged -- even to NaN -- on another (the container image, with a different
    core count), so two runs of one config produced different encoders. Asking
    for deterministic algorithms pins the stable backend, and a single training
    thread removes the parallel-reduction nondeterminism that a multi-core host
    otherwise introduces. Both are no-ops for the numbers on a machine that was
    already reproducible; they make the ones that were not."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def _build_loader(data_root: str, img_size: int, batch_size: int,
                  num_workers: int, seed: int):
    """An ImageFolder over `data_root`, normalised to [-1, 1] as the VQVAE
    expects. The finest scale fixes the image size, so it is derived, not
    configured."""
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(data_root, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, generator=generator)
    return dataset, loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAR step 1 (next-scale AR)")
    parser.add_argument("--config", default="configs/step1.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override the ImageFolder root of training images")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the upstream assumed CUDA")
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

    build_vae_var = _load_upstream()
    mk = model_kwargs(cfg["model"])
    vae, var = build_vae_var(device=device, flash_if_available=False,
                             fused_if_available=False, **mk)

    ckpt = cfg["data"].get("vqvae_ckpt")
    if ckpt:
        vae.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    else:
        # build_vae_var disables reset_parameters and initialises only VAR, so a
        # VQVAE built without a checkpoint is left uninitialised (torch.empty).
        # Its contents are environment-dependent -- finite on one host, NaN on
        # another -- which made tokenisation, and the whole run, reproducible on
        # one machine and divergent on the next. Give it finite, seeded weights
        # so the smoke's tokenisation is well-defined. A real run loads a
        # pretrained VQVAE (vqvae_ckpt) and never takes this path.
        for p in vae.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    vae.eval()
    var.train()

    optimizer = torch.optim.AdamW(
        var.parameters(), lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]))

    img_size = mk["patch_nums"][-1] * VAE_DOWNSAMPLE
    dataset, loader = _build_loader(
        cfg["data"]["data_root"], img_size,
        int(cfg["data"]["batch_size"]), int(cfg["data"]["num_workers"]), seed)

    grad_clip = float(cfg["training"]["grad_clip"])
    vocab = mk["V"]

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = state["epoch"] + 1
        var.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"Resumed from epoch {state['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    print("=" * 70)
    print("VAR  Step 1: next-scale autoregressive pretraining  (arXiv:2404.02905)")
    print(f"  device={device}  epochs={total_epochs}  "
          f"batch={cfg['data']['batch_size']}  images={len(dataset)}  "
          f"vqvae={'pretrained' if ckpt else 'random (smoke)'}")
    print("=" * 70)

    final_loss = None
    for epoch in range(start_epoch, total_epochs):
        running, count = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                gt_idx_Bl = vae.img_to_idxBl(imgs)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            logits = var(labels, x_BLCv)
            loss = F.cross_entropy(logits.reshape(-1, vocab), gt_BL.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(var.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * labels.size(0)
            count += labels.size(0)
        final_loss = running / count if count else None
        print(f"  [{epoch}] ce_loss={final_loss}")

        state = {"epoch": epoch, "model_state_dict": var.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "loss": final_loss, "config": cfg}
        torch.save(state, os.path.join(save_dir, "checkpoint_latest.pth"))

    print("\nVAR Step 1 training complete!")
    ran = total_epochs > start_epoch and final_loss is not None
    return {"epochs": total_epochs - start_epoch,
            "final_loss": final_loss if ran else None}


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
