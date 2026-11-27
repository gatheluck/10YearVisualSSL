"""One weighted teacher transport for CLS and selected masked patches."""
import torch
import torch.distributed as dist


@torch.no_grad()
def joint_teacher_assignments(cls_logits, patch_logits, teacher_temp, n_iters=3):
    if (cls_logits.ndim != 2 or patch_logits.ndim != 2
            or cls_logits.shape[1] != patch_logits.shape[1]
            or cls_logits.shape[1] == 0):
        raise ValueError('joint logits must share a nonempty prototype dimension')
    if teacher_temp <= 0 or n_iters < 1:
        raise ValueError('positive temperature and iteration count required')
    distributed = dist.is_available() and dist.is_initialized()

    def reduce(tensor):
        if distributed:
            dist.all_reduce(tensor)

    nc, np = cls_logits.shape[0], patch_logits.shape[0]
    counts = torch.tensor([nc, np], device=cls_logits.device, dtype=torch.int64)
    reduce(counts)
    gc, gp = counts.tolist()
    cm, pm = (.5, .5) if gc and gp else (float(bool(gc)), float(bool(gp)))
    mass = torch.cat((cls_logits.new_full((nc,), cm / max(gc, 1), dtype=torch.float32),
                      patch_logits.new_full((np,), pm / max(gp, 1), dtype=torch.float32)))
    logits = torch.cat((cls_logits.detach().float(), patch_logits.detach().float()))
    q = torch.exp(logits / teacher_temp).t()
    total = q.sum()
    reduce(total)
    if total <= 0:
        empty = torch.zeros_like(logits)
        return empty[:nc], empty[nc:]
    q = q / total
    prototype_mass = q.new_full((q.shape[0], 1), 1.0 / q.shape[0])
    for _ in range(n_iters):
        row_sum = q.sum(1, keepdim=True)
        reduce(row_sum)
        q = q * (prototype_mass / row_sum.clamp_min(1e-12))
        q = q * (mass.unsqueeze(0) / q.sum(0, keepdim=True).clamp_min(1e-12))
    probs = q.t().contiguous() / mass.clamp_min(1e-12).unsqueeze(1)
    return probs[:nc], probs[nc:]
