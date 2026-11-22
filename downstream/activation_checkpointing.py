"""Explicit recomputation policies; state dictionaries retain their original keys."""

from functools import wraps


def validate(spec, *, trainable):
    from downstream.spatial_backbones import _PROVIDERS, _load_provider

    enabled = spec.get("activation_checkpointing", False)
    if type(enabled) is not bool:
        raise ValueError("activation_checkpointing must be a boolean")
    if not enabled:
        return None
    kind = spec.get("kind")
    policy = (
        getattr(_load_provider(_PROVIDERS[kind]), "ACTIVATION_CHECKPOINTING", None)
        if kind in _PROVIDERS
        else None
    )
    if not trainable or policy is None:
        raise ValueError(
            "activation_checkpointing requires a supported finetune provider"
        )
    return policy


def enable(model, policy):
    """Apply only a provider's inspected backend; never silently ignore a request."""
    strategy, path = policy
    target = model.get_submodule(path)
    if strategy == "hf":
        # Non-reentrant supports task inputs without requires_grad and kwargs.
        target.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif strategy == "native_flag":
        if not hasattr(target, "use_activation_checkpointing"):
            raise ValueError("activation_checkpointing native flag is unavailable")
        target.use_activation_checkpointing = True
    elif strategy == "blocks":
        enable_blocks(target.blocks)
    elif strategy == "timm":
        target.set_grad_checkpointing(True)
    else:
        raise ValueError("unknown activation_checkpointing strategy")

def enable_blocks(blocks):
    """Checkpoint grad-enabled forwards while retaining module and parameter identities."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def wrap(forward):
        @wraps(forward)
        def recompute(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)
        return recompute

    if not len(blocks):
        raise ValueError("activation_checkpointing requires nonempty blocks")
    for block in blocks:
        if not getattr(block, "_portable_checkpointed", False):
            block.forward = wrap(block.forward)
            block._portable_checkpointed = True
