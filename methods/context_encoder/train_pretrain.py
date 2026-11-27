"""
Context Encoder step 1 training: AlexNet-based inpainting (Pathak et al., 2016).

    Encoder    : AlexNet-style conv stack + 4096-d bottleneck (the representation)
    Decoder    : transposed-conv head that reconstructs the 128x128 centre hole
    Loss       : reconstruction (L1/L2/smooth L1) + optional adversarial (a
                 centre-hole discriminator, BCE-with-logits, weight 0.001)
    Optimiser  : SGD for the generator; Adam(betas=0.5,0.999) for the
                 discriminator when adversarial training is on

This is the AlexNet path of the captured `train.py`, extracted as a
self-contained pretrain trainer. It is what the capture calls Step 1; the
ViT-based Step 2 (and its two-AdamW, bfloat16, adversarial-always protocol) was
not brought across.

Changed during the port, and recorded in provenance.json:

  - **the device is resolved instead of assumed.** The captured trainer sent
    the model and every batch to CUDA with `.cuda(args.gpu)`, so it could not
    start without a GPU. `resolve_device` picks one; asking for `cuda` where
    there is none is refused rather than served a CPU in silence
  - **`main()` is split into `build_parser()` and `run(args, config)`,** which
    returns the epoch's reconstruction, adversarial and total losses. The
    captured `main()` returned only the total and dropped the components
  - **the run is seeded through `make_deterministic`.**
  - **single process, full precision.** The captured AMP/DDP/step-2 machinery is
    not brought across; the pretrain loop's plain fp32 path is used, which runs on
    a CPU or a GPU unchanged
"""

import os
import random
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.context_encoder import create_model, Discriminator
from datasets import create_dataloader


class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def resolve_device(spec: str, local_rank: int = 0) -> "torch.device":
    """Which device to run on, decided rather than assumed.

    The captured trainer called `.cuda(args.gpu)` on the model and every batch,
    so it raised on a machine with no GPU. `"cuda"` is honoured only when there
    is one: falling back quietly would let a run asked for a GPU report success
    from a CPU, and the two are not the same run.
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
    """Seed everything the run draws from, and pin the sources of run-to-run
    nondeterminism so two runs of one config produce bit-identical weights.

    Seeding alone is not enough here: this is the one GAN (two models, two
    optimisers), and its multi-threaded floating-point reductions gave a
    different reduction order -- and so slightly different weights -- from one
    run to the next in multi-core CI containers, which is why its
    same-config-twice determinism test flaked there. Single-threading and
    deterministic algorithms remove that source; every other method already
    pins these, and this brings 04 in line."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def compute_reconstruction_loss(pred, target, loss_type="l2"):
    """Reconstruction loss, exactly as the captured trainer computes it."""
    if loss_type == "l1":
        return nn.L1Loss()(pred, target)
    if loss_type == "l2":
        return nn.MSELoss()(pred, target)
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()(pred, target)
    raise ValueError(f"Unknown loss type: {loss_type}")


def adjust_learning_rate(optimizer, epoch, base_lr, total_epochs,
                         warmup_epochs=10):
    """Cosine learning-rate schedule with warmup (the captured schedule)."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        cosine_epochs = total_epochs - warmup_epochs
        progress = (epoch - warmup_epochs) / max(cosine_epochs - 1, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_epoch(model, discriminator, train_loader, criterion, optimizer,
                d_optimizer, epoch, cfg, device):
    """One epoch of the AlexNet inpainting loop, full precision.

    Mirrors the captured `train_epoch_alexnet` non-AMP branch, with `.cuda()`
    replaced by `.to(device)`.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()

    use_adversarial = bool(cfg["training"]["use_adversarial"])
    adversarial_weight = float(cfg["training"]["adversarial_weight"])
    print_freq = int(cfg["training"]["print_freq"])

    losses = {"recon": AverageMeter("Recon"), "adv": AverageMeter("Adv"),
              "total": AverageMeter("Total")}
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        masked_images = batch["masked_image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Train the discriminator on detached generator output.
        if discriminator is not None and use_adversarial:
            with torch.no_grad():
                pred_detached, _ = model(masked_images)
                pred_detached = pred_detached.detach()
            real_validity = discriminator(targets)
            fake_validity = discriminator(pred_detached)
            d_real_loss = nn.BCEWithLogitsLoss()(
                real_validity, torch.ones_like(real_validity))
            d_fake_loss = nn.BCEWithLogitsLoss()(
                fake_validity, torch.zeros_like(fake_validity))
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Train the generator.
        pred, _features = model(masked_images)
        recon_loss = criterion(pred, targets)
        if discriminator is not None and use_adversarial:
            fake_validity = discriminator(pred)
            adv_loss = nn.BCEWithLogitsLoss()(
                fake_validity, torch.ones_like(fake_validity))
            loss = recon_loss + adversarial_weight * adv_loss
        else:
            adv_loss = torch.tensor(0.0)
            loss = recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = targets.size(0)
        losses["recon"].update(recon_loss.item(), n)
        losses["adv"].update(
            adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            n)
        losses["total"].update(loss.item(), n)

        if batch_idx % print_freq == 0:
            print(f"  [{epoch}][{batch_idx:5d}/{len(train_loader)}]  "
                  f"loss={losses['total'].avg:.4f} "
                  f"(recon={losses['recon'].avg:.4f} "
                  f"adv={losses['adv'].avg:.4f})  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"t={time.time() - t0:.1f}s")

    return losses["recon"].avg, losses["adv"].avg, losses["total"].avg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context Encoder Step 1 (AlexNet)")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--data_path", default=None,
                        help="Override ImageNet root (parent of train/ and val/)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Added by the port; the captured trainer assumed "
                             "CUDA")
    return parser


def run(args, config: dict | None = None) -> dict:
    """The captured AlexNet `main()`, callable in process and returning its
    reconstruction/adversarial/total losses."""
    if config is not None:
        cfg = config
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    if getattr(args, "data_path", None):
        # InpaintingDataset joins root + split itself, so this is the parent of
        # train/ and val/, not the train/ directory.
        cfg["data"]["train_path"] = args.data_path

    device = resolve_device(getattr(args, "device", "auto"))
    make_deterministic(int(cfg.get("seed", 42)))

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    model = create_model("alexnet", channels=3).to(device)

    use_adversarial = bool(cfg["training"]["use_adversarial"])
    discriminator = None
    d_optimizer = None
    if use_adversarial:
        discriminator = Discriminator(
            channels=3, img_size=int(cfg["data"]["mask_size"])).to(device)
        d_optimizer = optim.Adam(
            discriminator.parameters(), lr=float(cfg["training"]["lr"]),
            betas=(0.5, 0.999))

    optimizer = optim.SGD(
        model.parameters(), lr=float(cfg["training"]["lr"]),
        momentum=float(cfg["training"]["momentum"]),
        weight_decay=float(cfg["training"]["weight_decay"]))

    criterion = lambda pred, target: compute_reconstruction_loss(
        pred, target, cfg["training"]["loss_type"])

    train_loader = create_dataloader(
        "inpainting", cfg["data"]["train_path"], split="train",
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        model_type="alexnet", img_size=int(cfg["data"]["img_size"]),
        mask_size=int(cfg["data"]["mask_size"]))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from epoch {ckpt['epoch']}")

    total_epochs = int(cfg["training"]["epochs"])
    warmup_epochs = int(cfg["training"]["warmup_epochs"])
    save_freq = int(cfg["training"]["save_freq"])

    print("=" * 70)
    print("Context Encoder  Step 1: AlexNet inpainting  (arXiv:1604.07379)")
    print(f"  epochs={total_epochs}  batch={cfg['training']['batch_size']}  "
          f"adversarial={use_adversarial}  loss={cfg['training']['loss_type']}")
    print("=" * 70)

    recon = adv = total = None
    for epoch in range(start_epoch, total_epochs):
        adjust_learning_rate(optimizer, epoch, float(cfg["training"]["lr"]),
                             total_epochs, warmup_epochs)
        recon, adv, total = train_epoch(
            model, discriminator, train_loader, criterion, optimizer,
            d_optimizer, epoch, cfg, device)

        if (epoch + 1) % save_freq == 0 or epoch == total_epochs - 1:
            state = {"epoch": epoch,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "loss": total, "config": cfg}
            if discriminator is not None:
                state["discriminator_state_dict"] = discriminator.state_dict()
            path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(state, path)
            print(f"  [ckpt] Saved: {path}")

    print("\nContext Encoder Step 1 training complete!")

    ran = total_epochs > start_epoch
    return {
        "epochs": total_epochs - start_epoch,
        "final_loss": total if ran else None,
        "final_recon_loss": recon if ran else None,
        "final_adv_loss": adv if ran else None,
    }


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
