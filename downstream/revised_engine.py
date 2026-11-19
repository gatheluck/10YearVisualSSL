"""Explicit revised ImageNet/ADE updates; callers own loaders and artifacts."""
from contextlib import nullcontext

import torch
from torch import distributed as dist
from torch.nn import functional as F

from downstream.extended_execution import _require_all
from downstream.revised_recipe import set_learning_rates


def _world():
    return dist.get_world_size() if dist.is_initialized() else 1


def _task(task):
    if task not in ("imagenet", "ade20k"):
        raise ValueError("unsupported revised task")
    if task == "ade20k" and _world() > 1:
        raise ValueError("revised dense source uses a single device")


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _stage(function, device):
    """Reach one common status collective even when local work raises."""
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        if _world() == 1:
            raise
        error = f"{type(exc).__name__}: {exc}"
    _require_all(error is None, device, "revised stage failed: " + (error or "peer failure"))
    return result


def check_initial_state(model):
    """Reject different rank initialization without changing any parameters."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("no trainable parameters")
    if _world() == 1:
        return
    metadata = [(n, tuple(p.shape), str(p.dtype)) for n, p in named]
    gathered = [None] * _world()
    dist.all_gather_object(gathered, metadata)
    if any(value != metadata for value in gathered):
        raise ValueError("rank trainable parameter identity differs")
    flat = torch.cat([p.detach().reshape(-1) for _, p in named])
    reference = flat.clone()
    dist.broadcast(reference, 0)
    _require_all(torch.equal(flat, reference), flat.device, "rank initial parameters differ")


def mean_gradients(model):
    """Match the reference flat mean reduction, including zero missing entries."""
    params = [p for p in model.parameters() if p.requires_grad]
    if _world() == 1:
        return
    gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in gradients])
    dist.all_reduce(flat)
    flat /= _world()
    offset = 0
    for parameter in params:
        count = parameter.numel()
        parameter.grad = flat[offset:offset + count].view_as(parameter).clone()
        offset += count


def train_epoch(model, loader, optimizer, recipe, step, steps_per_epoch, device,
                max_steps=0, adapters=None):
    """One source-shaped epoch without accumulation; caps remain diagnostic.

    ImageNet callers provide equal physical batches and a drop-last distributed
    sampler. Dense execution is single-device as in the inspected source. The
    schedule always retains the recipe horizon, including diagnostic truncation.
    """
    _task(recipe.task)
    if (type(step) is not int or step < 0 or type(steps_per_epoch) is not int
            or steps_per_epoch < 1 or type(max_steps) is not int or max_steps < 0):
        raise ValueError("invalid revised update counters")
    if not len(loader) or len(loader) != steps_per_epoch:
        raise ValueError("epoch loader must match the nonempty schedule horizon")
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    losses, updates = 0., 0
    iterator = _stage(lambda: iter(loader), device)
    for _ in range(steps_per_epoch):
        if max_steps and step >= max_steps:
            break
        batch = _stage(lambda: next(iterator), device)

        def forward_loss(batch=batch, step=step):
            images, labels = (value.to(device) for value in batch)
            set_learning_rates(optimizer, recipe, step, steps_per_epoch)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(images) if recipe.task == "imagenet" else model(images, labels.shape[-2:])
            return F.cross_entropy(logits.float(), labels,
                                   ignore_index=255 if recipe.task == "ade20k" else -100)

        loss = _stage(forward_loss, device)
        _require_all(torch.isfinite(loss), device, "nonfinite revised training loss")
        _stage(loss.backward, device)
        _require_all(not any(p.grad is not None for p in model.parameters() if not p.requires_grad),
                     device, "frozen parameter received a gradient")
        _require_all(all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable),
                     device, "nonfinite revised gradient")
        if recipe.task == "ade20k":
            _require_all(not any(p.grad is None for n, p in model.named_parameters()
                                 if p.requires_grad and not n.endswith("lora_A")),
                         device, "trainable parameter missing gradient")
        mean_gradients(model)
        if adapters and not updates:
            zero = sum(layer.lora_B.grad is None or layer.lora_B.grad.abs().sum().item() == 0
                       for layer in adapters.values())
            _require_all(zero <= .1 * len(adapters), device, "LoRA B gradient is absent or zero")
        torch.nn.utils.clip_grad_norm_(trainable, recipe.clip_norm)
        optimizer.step()
        step += 1
        updates += 1
        losses += loss.item()
    if not updates:
        raise ValueError("epoch produced no optimizer updates")
    return step, losses / updates, updates


@torch.no_grad()
def evaluate(model, loader, task, device):
    """Source metrics; distributed image loaders must be rank-strided, unpadded."""
    _task(task)
    model.eval()
    counts = torch.zeros(3, dtype=torch.float64, device=device)
    intersection = torch.zeros(150, device=device)
    union = torch.zeros_like(intersection)
    images_seen = 0
    def local_evaluation():
        nonlocal images_seen, intersection, union
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with _autocast(device):
                output = model(images) if task == "imagenet" else model(images, labels.shape[-2:])
            # No collectives inside uneven validation loops.
            if not torch.isfinite(output).all():
                counts[0] = float("nan")
                continue
            images_seen += len(labels)
            if task == "imagenet":
                top = output.float().topk(5, dim=1).indices
                counts[0] += (top[:, 0] == labels).sum()
                counts[1] += (top == labels[:, None]).any(1).sum()
                counts[2] += labels.numel()
            else:
                prediction = output.float().argmax(1)
                valid = labels != 255
                predicted, target = prediction[valid], labels[valid]
                counts[0] += (predicted == target).sum()
                counts[2] += valid.sum()
                matched = target[predicted == target]
                intersection += torch.bincount(matched, minlength=150)
                union += (torch.bincount(predicted, minlength=150)
                          + torch.bincount(target, minlength=150)
                          - torch.bincount(matched, minlength=150))

    _stage(local_evaluation, device)
    if _world() > 1:
        dist.all_reduce(counts)
    if not torch.isfinite(counts).all() or counts[2] <= 0:
        raise ValueError("nonfinite or empty revised evaluation")
    if task == "imagenet":
        return {"top1": 100 * counts[0].item() / counts[2].item(),
                "top5": 100 * counts[1].item() / counts[2].item(),
                "n": int(counts[2].item())}
    present = union > 0
    return {"miou": float((intersection[present] / union[present]).double().mean().item() * 100),
            "pixel_acc": 100 * counts[0].item() / counts[2].item(),
            "n_pixels": int(counts[2].item()), "n_classes_present": int(present.sum().item()),
            "n": images_seen}
