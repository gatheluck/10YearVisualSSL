"""Preflight portable training state without mutating the live run."""

import random

import numpy as np
import torch


def optimizer_state(optimizer, saved, updates):
    """Validate the recipe and initialized slots; unused parameters may be lazy.

    These portable runners use SGD and AdamW. PyTorch's load_state_dict maps
    slots by position and replaces group options, so loading alone cannot
    verify that a checkpoint retains the configured training recipe.
    """
    current = optimizer.state_dict()
    groups = saved["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current["param_groups"]):
        raise ValueError("checkpoint optimizer group count differs")
    parameters = {}
    policies = {}
    for actual, expected, live in zip(
        groups, current["param_groups"], optimizer.param_groups
    ):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("checkpoint optimizer group schema differs")
        if (
            not isinstance(actual["params"], list)
            or any(type(index) is not int for index in actual["params"])
            or actual["params"] != expected["params"]
        ):
            raise ValueError("checkpoint optimizer parameter mapping differs")
        # The scheduler's exact current learning rates are checked by Continuation.
        if any(
            actual[key] != expected[key]
            for key in expected
            if key not in ("params", "lr")
        ):
            raise ValueError("checkpoint optimizer recipe differs")
        parameters.update(zip(expected["params"], live["params"]))
        policies.update((index, expected) for index in expected["params"])
    states = saved["state"]
    if not isinstance(states, dict) or any(
        type(index) is not int or index not in parameters for index in states
    ):
        raise ValueError("checkpoint optimizer contains unknown parameter state")
    if not isinstance(optimizer, (torch.optim.SGD, torch.optim.AdamW)):
        raise TypeError("checkpoint optimizer algorithm is not supported")
    for index, state in states.items():
        group = policies[index]
        if isinstance(optimizer, torch.optim.SGD):
            keys = {"momentum_buffer"} if group["momentum"] else set()
        else:
            keys = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                keys.add("max_exp_avg_sq")
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("checkpoint optimizer slots are incomplete or unexpected")
        parameter = parameters[index]
        for key, value in state.items():
            if key == "step":
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 0
                    or not value.is_floating_point()
                    or not torch.isfinite(value)
                    or not 1 <= value.item() <= updates
                    or value.item() != int(value.item())
                ):
                    raise ValueError("checkpoint optimizer step is invalid")
            else:
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                ):
                    raise ValueError(
                        "checkpoint optimizer moment shape or dtype differs"
                    )
                if key in ("exp_avg_sq", "max_exp_avg_sq") and (value < 0).any():
                    raise ValueError("checkpoint optimizer second moment is negative")


def rng_states(states, loader, device):
    """Validate every rank using isolated generators before restoring any rank."""
    keys = {"python", "numpy", "torch", "cuda", "loader"}
    for state in states:
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("checkpoint RNG schema differs")
        if (state["cuda"] is None) != (device.type != "cuda"):
            raise ValueError("checkpoint CUDA RNG identity mismatch")
        if (state["loader"] is None) != (loader.generator is None):
            raise ValueError("checkpoint loader RNG identity mismatch")
        try:
            random.Random().setstate(state["python"])
            value = state["numpy"]
            np.random.RandomState().set_state(
                (value[0], np.asarray(value[1], dtype=np.uint32), *value[2:])
            )
            torch.Generator().set_state(state["torch"])
            if device.type == "cuda":
                torch.Generator(device=device).set_state(state["cuda"])
            if loader.generator is not None:
                torch.Generator(device=loader.generator.device).set_state(
                    state["loader"]
                )
        except (TypeError, ValueError, RuntimeError, IndexError, OverflowError) as exc:
            raise ValueError("invalid checkpoint RNG state") from exc
