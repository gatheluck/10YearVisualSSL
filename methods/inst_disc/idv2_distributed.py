"""Explicit FP32 torchrun entry for the seven IDv2 component profiles."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from downstream.contract import write_metrics
from downstream.extended_distributed import cli, session


def require_finite(loss, distributed=False):
    flag = torch.isfinite(loss.detach()).to(torch.int32)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise ValueError("nonfinite IDv2 loss on a training rank")


def view_context(model, last):
    return model.no_sync() if isinstance(model, DDP) and not last else nullcontext()


class DenseForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, view):
        from idv2_components import dense_forward
        return dense_forward(self.model, view)


def call(context, fn, leader=False):
    return fn() if context is None else context.call(fn, leader=leader)


def run(config, out):
    from adapter import extract_encoder, to_args, to_run_config
    from idv2_components import run_components
    with session(config["device"]) as context:
        args, cfg = context.call(lambda: (to_args(config, out), to_run_config(config, out)))
        if not cfg.get("profile"):
            raise ValueError("distributed entry requires an IDv2 component profile")
        args.distributed = True
        metrics = run_components(args, cfg)

        def deliver():
            work = Path(out) / "work"
            for epoch in [None, *cfg["training"]["save_at_epochs"]]:
                name = "checkpoint_latest.pth" if epoch is None else f"checkpoint_epoch_{epoch}.pth"
                source = work / name
                if epoch is not None and not source.is_file():
                    continue
                state = torch.load(source, map_location="cpu", weights_only=False)
                name = "encoder.pt" if epoch is None else f"encoder_epoch{epoch}.pt"
                torch.save(extract_encoder(state["model_state_dict"]), Path(out) / name)
            write_metrics(out, metrics, {"epochs": None, "final_loss": None})
            (Path(out) / "results.json").write_text(json.dumps({**metrics,
                "world_size": context.world, "global_batch_size": cfg["training"]["batch_size"],
                "precision": "float32", "canonical_eligible": False}, indent=2) + "\n")
        context.call(deliver, leader=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return cli(Path(args.config).read_bytes(), args.out, run, "idv2_components",
               method_ref="idv2_components")


if __name__ == "__main__":
    raise SystemExit(main())
