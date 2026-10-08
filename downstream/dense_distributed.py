"""Shared optional DDP lifecycle for the existing BasicFive dense runners."""

from contextlib import contextmanager
from functools import wraps

import torch
from torch.utils.data import DataLoader

from downstream import dense_execution, dense_resume
from downstream import extended_distributed as distributed
from downstream.spatial_backbones import dense_execution_policy


def requested(cfg):
    return "execution" in cfg and distributed.launch()[2] > 1


def call(context, fn, *, leader=False):
    return fn() if context is None else context.call(fn, leader=leader)


@contextmanager
def setup(context):
    """Finish local setup on every rank before any model collective is entered."""
    if context is None:
        yield
        return
    error = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - propagate any setup failure to all peers
        error = exc

    def check():
        if error is not None:
            raise error

    context.call(check)


def with_execution(validate):
    def decorate(fn):
        @wraps(fn)
        def run(cfg, out, device_override=None):
            if "execution" not in cfg:
                return fn(cfg, out, device_override, context=None)
            # Preserve single-process ConfigError behavior before device selection.
            if distributed.launch()[2] == 1:
                validate(cfg)
            with distributed.session(
                device_override or cfg.get("device", "cpu")
            ) as context:
                context.agree(cfg)
                context.call(lambda: validate(cfg))
                context.call(
                    lambda: dense_execution.autocast_context(
                        context.device, dense_execution.resolve(cfg)["precision"]
                    )
                )
                context.call(lambda: dense_resume.output_directory(out), leader=True)
                return fn(cfg, out, device_override, context=context)

        return run

    return decorate


def seed(cfg, context):
    rank = 0 if context is None else context.rank
    policy = dense_execution_policy(cfg["backbone"]["kind"]) if rank else None
    return int(cfg["seed"]) + (policy["rank_seed_stride"] * rank if policy else 0)


def loader(context, data, settings, seed_value, optimization, *, collate_fn=None):
    if context is not None and context.world > 1:
        return context.loader(data, settings, seed_value, collate_fn=collate_fn)
    return DataLoader(
        data,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings["num_workers"]),
        drop_last=optimization is not None,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed_value),
    )


def set_epoch(context, loader, epoch):
    if context is not None and context.world > 1:
        loader.sampler.set_epoch(epoch)


def wrap(context, model):
    return model if context is None else context.wrap(model)


def cli(data, out, run, task, device_override):
    return distributed.cli(
        data,
        out,
        lambda cfg, path: run(cfg, path, device_override=device_override),
        task,
        device_override=device_override,
    )
