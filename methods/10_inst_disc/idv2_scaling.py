"""Separate 90-epoch IDv2 multicrop scaling execution, not a score assertion."""
import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from idv2_components import Views, restore_rng, rng_state, shared_rows
from idv2_distributed import require_finite, view_context
from nce import NCELoss

from downstream.contract import write_metrics
from downstream.extended_distributed import cli, session


def schedule(step, total_steps):
    if step < 10000:
        return max(.0006 * (step + 1) / 10000, 1e-8)
    progress = min(max((step - 10000) / max(total_steps - 10000, 1), 0), 1)
    return .0003 * (1 + math.cos(math.pi * progress))


class ScalingBank(NCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_cap = self.num_samples * math.exp(1 / self.temperature)


def train_step(model, bank, optimizer, batches, precision):
    if not batches or precision not in ("float32", "bfloat16"):
        raise ValueError("nonempty accumulated batches and explicit precision required")
    optimizer.zero_grad(set_to_none=True)
    features, indices, losses = [], [], []
    for micro, (views, idx, _) in enumerate(batches):
        if len(views) != 6:
            raise ValueError("scaling requires six views")
        if precision == "bfloat16" and views[0].device.type != "cuda":
            raise ValueError("bfloat16 scaling requires CUDA")
        rows = shared_rows(bank, idx)
        with view_context(model, micro == len(batches) - 1):
            with torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext():
                globals_ = model(torch.cat(views[:2])).float().chunk(2)
                locals_ = model(torch.cat(views[2:])).float().chunk(4)
            terms = [bank.loss_from_exp(torch.exp(torch.bmm(rows, z.unsqueeze(2)).squeeze(2) / bank.temperature))
                     for z in (*globals_, *locals_)]
            loss = torch.stack(terms).mean()
            require_finite(loss, bank.distributed)
            (loss / len(batches)).backward()
        features.append(F.normalize(.5 * (globals_[0].detach() + globals_[1].detach()), dim=1))
        indices.append(idx); losses.append(loss.detach())
    optimizer.step()
    bank.update_memory(torch.cat(features), torch.cat(indices))
    return torch.stack(losses).mean().item()


class ScalingModel(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        import timm
        from timm.models.vision_transformer import VisionTransformer
        encoder = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="", dynamic_img_size=True)
        if not isinstance(encoder, VisionTransformer):
            raise TypeError("scaling preset must construct a VisionTransformer")
        self.encoder = encoder
        self.fc = torch.nn.Linear(self.encoder.embed_dim, 128)
        self.encoder.set_grad_checkpointing(True)

    def forward(self, x):
        return F.normalize(self.fc(self.encoder.forward_features(x)[:, 0]), dim=1)


def build_model(arch):
    return ScalingModel(arch)


def load_encoder(path):
    """Restore the native preset and return raw CLS features, not projections."""
    from models.vit_instdisc import _ClsFeature
    export = torch.load(path, map_location="cpu", weights_only=True)
    if (export.get("scaling_encoder_version") != 1 or export.get("readout") != "cls"
            or export.get("model") not in ("vit_base_patch16_224", "vit_large_patch16_224")):
        raise ValueError("incompatible scaling encoder identity")
    model = build_model(export["model"])
    model.encoder.load_state_dict(export["state_dict"], strict=True)
    return _ClsFeature(model.encoder)


def validate(config):
    keys = {"device", "seed", "model", "precision", "prepared_root", "validation_root",
            "epochs", "microbatch", "accum", "num_workers", "resume"}
    if set(config) != keys:
        raise ValueError("scaling requires exactly the documented configuration keys")
    for key, minimum in (("seed", 0), ("epochs", 1), ("microbatch", 1), ("accum", 1), ("num_workers", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"invalid scaling {key}")
    if config["epochs"] > 90:
        raise ValueError("scaling endpoint must not exceed 90 epochs")
    if config["model"] not in ("vit_base_patch16_224", "vit_large_patch16_224"):
        raise ValueError("scaling requires the ViT-B/16 or ViT-L/16 preset")
    if config["device"] not in ("cpu", "cuda") or config["precision"] not in ("float32", "bfloat16"):
        raise ValueError("explicit scaling device and precision required")
    if config["precision"] == "bfloat16" and config["device"] != "cuda":
        raise ValueError("bfloat16 scaling requires CUDA")


def _setup(config, out, context):
    from data.indexed_tar import IndexedTarDataset
    seed = config["seed"] + context.rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    dataset = IndexedTarDataset(out / "membership/TRAIN_MANIFEST.jsonl", config["prepared_root"], Views(224, 96))
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=context.world,
        rank=context.rank, shuffle=True, seed=config["seed"], drop_last=True) if context.world > 1 else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["microbatch"], sampler=sampler,
        shuffle=sampler is None, drop_last=True, num_workers=config["num_workers"], generator=generator)
    steps = len(loader) // config["accum"]
    if not steps:
        raise ValueError("population too small for one complete accumulated update")
    model = build_model(config["model"]).to(context.device)
    bank = ScalingBank(len(dataset), 128, .07, .5, 4096).to(context.device)
    bank.distributed = context.world > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0006, weight_decay=.05)
    return model, bank, optimizer, loader, sampler, generator, steps


def _restore(source, contract, model, bank, optimizer, generator, context, epochs, steps):
    if not source:
        return 0, 0
    state = torch.load(source, map_location="cpu", weights_only=False)
    if (state.get("scaling_version") != 1 or state.get("contract") != contract
            or type(state.get("epoch")) is not int or not 0 <= state["epoch"] < epochs - 1
            or state.get("global_step") != (state["epoch"] + 1) * steps
            or len(state.get("rank_rng", [])) != context.world):
        raise ValueError("incompatible scaling checkpoint")
    adam = state.get("optimizer", {})
    if len(adam.get("state", {})) != len(list(model.parameters())) or any(
            not {"step", "exp_avg", "exp_avg_sq"} <= set(v) for v in adam.get("state", {}).values()):
        raise ValueError("incompatible incomplete AdamW state")
    model.load_state_dict(state["model"]); bank.load_state_dict(state["bank"])
    optimizer.load_state_dict(adam)
    restore_rng(state["rank_rng"][context.rank], generator, context.device)
    return state["epoch"] + 1, state["global_step"]


def run(config, out):
    from data.indexed_tar import freeze, validate_output
    out = Path(out).resolve()
    with session(config["device"]) as context:
        context.agree(config)
        context.call(lambda: validate(config))
        context.call(lambda: validate_output(out, config["prepared_root"], config["validation_root"]))
        source = Path(config["resume"]).resolve() if config["resume"] else None
        if source and source.is_relative_to(out):
            raise ValueError("output must not contain resume source")
        meta = context.call(lambda: freeze(config["prepared_root"], config["validation_root"], out / "membership"), leader=True)
        model, bank, optimizer, loader, sampler, generator, steps = context.call(lambda: _setup(config, out, context))
        contract = {k: v for k, v in config.items() if k not in ("epochs", "resume")}
        contract.update(manifest_sha256=meta["manifest_sha256"], world_size=context.world,
                        steps_per_epoch=steps, schedule_epochs=90)
        context.agree(contract)
        start, global_step = context.call(lambda: _restore(source, contract, model, bank, optimizer,
                                                          generator, context, config["epochs"], steps))
        raw = model
        if context.world > 1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                device_ids=[context.device.index] if context.device.type == "cuda" else None)
            bank.synchronize_state()
        model.train()
        for epoch in range(start, config["epochs"]):
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader); losses = []
            for _ in range(steps):
                def read(iterator=iterator):
                    batches = []
                    for _ in range(config["accum"]):
                        views, idx, label = next(iterator)
                        batches.append(([v.to(context.device) for v in views], idx.to(context.device), label))
                    return batches
                batches = context.call(read)
                for group in optimizer.param_groups:
                    group["lr"] = schedule(global_step, 90 * steps)
                losses.append(train_step(model, bank, optimizer, batches, config["precision"]))
                global_step += 1
            healthy = all(bool(torch.isfinite(value).all()) for module in (raw, bank)
                          for value in module.state_dict().values())
            require_finite(torch.tensor(0.0 if healthy else float("nan"), device=context.device), bank.distributed)
            loss = torch.tensor(sum(losses) / steps, dtype=torch.float64, device=context.device)
            rng = [rng_state(generator, context.device)]
            if context.world > 1:
                torch.distributed.all_reduce(loss); loss /= context.world
                rng = [None] * context.world
                torch.distributed.all_gather_object(rng, rng_state(generator, context.device))
            def save(epoch=epoch, global_step=global_step, rng=rng):
                state = {"scaling_version": 1, "contract": contract, "epoch": epoch, "global_step": global_step,
                         "model": raw.state_dict(), "bank": bank.state_dict(), "optimizer": optimizer.state_dict(),
                         "rank_rng": rng, "canonical_eligible": False}
                target = out / "checkpoint_latest.pth"
                temporary = target.with_suffix(".tmp")
                torch.save(state, temporary); temporary.replace(target)
                encoder = {k.removeprefix("encoder."): v for k, v in raw.state_dict().items() if k.startswith("encoder.")}
                if not encoder:
                    raise ValueError("empty scaling encoder export")
                export = {"scaling_encoder_version": 1, "model": config["model"],
                          "readout": "cls", "state_dict": encoder, "canonical_eligible": False}
                torch.save(export, out / "encoder.pt")
                if epoch + 1 in (30, 60, 90):
                    torch.save(state, out / f"checkpoint_epoch_{epoch + 1}.pth")
                    torch.save(export, out / f"encoder_epoch{epoch + 1}.pt")
            context.call(save, leader=True)
        def deliver():
            metrics = {"epochs_completed": config["epochs"], "final_loss": loss.item(), "global_step": global_step}
            write_metrics(out, metrics, {k: None for k in metrics})
            (out / "results.json").write_text(json.dumps({**metrics, "canonical_eligible": False,
                "global_batch_size": context.world * config["microbatch"] * config["accum"],
                "optimizer_steps_per_epoch": steps, "dropped_microbatches_per_rank": len(loader) % config["accum"],
                "precision": config["precision"], "manifest_sha256": meta["manifest_sha256"]}, indent=2) + "\n")
        context.call(deliver, leader=True)


def main(argv=None):
    from data.indexed_tar import validate_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    data = Path(args.config).read_bytes()
    config = json.loads(data)
    validate(config)
    validate_output(args.out, config["prepared_root"], config["validation_root"])
    return cli(data, args.out, run, "idv2_scaling", method_ref="idv2_scaling")


if __name__ == "__main__":
    raise SystemExit(main())
