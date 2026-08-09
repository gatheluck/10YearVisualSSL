"""Sinkhorn-Knopp optimal transport for SeLa (Asano et al., 2020), ported from
the lab's own implementation.

The label assignment is an optimal-transport problem: given the network's cluster
logits, find the assignment closest to the network's prediction subject to an
**equipartition** constraint (every cluster gets the same mass). Sinkhorn-Knopp
solves it; the argmax of the transported matrix gives balanced hard pseudo-labels.
The internal iteration runs in float64 (the official code raises softmax to a
large power `lambda`, which underflows float32). torch only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores, n_iters=1000, temperature=1.0, epsilon=0.05,
                   lamb=None, tol=1e-1):
    """Sinkhorn-Knopp optimal transport.

    Returns ``(Q, info)``. Q has row sums ~= 1/N and column sums ~= 1/K (doubly
    stochastic / N); callers row-normalise (or argmax) before use.
    """
    N, K = scores.shape
    work_dtype = torch.float64
    P = F.softmax(scores.to(work_dtype) / temperature, dim=1)
    power = float(lamb) if lamb is not None else 1.0 / float(epsilon)
    Q = P.clamp_min(torch.finfo(work_dtype).tiny).pow(power).t().contiguous()

    inv_k = 1.0 / K
    inv_n = 1.0 / N
    r = torch.full((K, 1), inv_k, dtype=work_dtype, device=Q.device)
    c = torch.full((N, 1), inv_n, dtype=work_dtype, device=Q.device)
    err = float("inf")
    counter = 0
    tiny = torch.finfo(work_dtype).tiny

    while err > tol and counter < n_iters:
        r = inv_k / (Q @ c).clamp_min(tiny)
        c_new = inv_n / (r.t() @ Q).t().clamp_min(tiny)
        if counter % 10 == 0:
            err = torch.nansum(torch.abs(c / c_new - 1.0)).item()
        c = c_new
        counter += 1

    Q = Q * c.squeeze(1).unsqueeze(0)
    Q = Q * r.squeeze(1).unsqueeze(1)
    return Q.t().to(scores.dtype), {"iterations": counter, "error": float(err)}


@torch.no_grad()
def compute_hard_sinkhorn_assignments(model, loader, device, num_heads=1,
                                      n_iters=1000, temperature=1.0,
                                      epsilon=0.05, lamb=25, tol=1e-1,
                                      verbose=True):
    """Hard SeLa pseudo-labels for each prototype head.

    Extracts each head's logits over the whole dataset, runs Sinkhorn once per
    head, and stores the argmax target per image/head. Returns a LongTensor of
    shape ``(N,)`` for one head, or ``(N, H)`` for multiple heads. The loader
    must yield ``(images, labels, indices)`` so assignments map back to images.
    """
    was_training = model.training
    model.eval()

    n_total = len(loader.dataset)
    hard_targets = torch.empty(n_total, num_heads, dtype=torch.long)

    all_logits: "torch.Tensor | None" = None
    all_indices = torch.empty(n_total, dtype=torch.long)
    ptr = 0

    for batch in loader:
        if len(batch) != 3:
            raise ValueError("DataLoader must return (images, labels, indices).")
        images, _, indices = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)

        if logits.dim() == 2:
            if num_heads != 1:
                raise ValueError(
                    f"Model returned single-head logits but num_heads={num_heads}")
            batch_heads, K = 1, logits.size(1)
        elif logits.dim() == 3:
            if logits.size(1) != num_heads:
                raise ValueError(
                    f"Model returned {logits.size(1)} heads but "
                    f"num_heads={num_heads}")
            batch_heads, K = logits.size(1), logits.size(2)
        else:
            raise ValueError(
                f"Expected logits (B,K) or (B,H,K), got {tuple(logits.shape)}")

        bsz = logits.size(0)
        if all_logits is None:
            shape = (n_total, K) if batch_heads == 1 else (n_total, batch_heads, K)
            all_logits = torch.empty(*shape, dtype=torch.float16)
        all_logits[ptr:ptr + bsz] = logits.float().cpu().half()
        all_indices[ptr:ptr + bsz] = indices.cpu()
        ptr += bsz

    if all_logits is None:
        raise RuntimeError("Sinkhorn loader produced no batches.")
    if ptr != n_total:
        raise RuntimeError(f"Collected {ptr} logits for {n_total} samples.")

    inv_idx = torch.empty_like(all_indices)
    inv_idx[all_indices] = torch.arange(n_total, dtype=all_indices.dtype)

    for head_idx in range(num_heads):
        head_logits = all_logits if all_logits.dim() == 2 \
            else all_logits[:, head_idx, :]
        N, K = head_logits.shape
        Q, info = sinkhorn_knopp(head_logits.float().to(device), n_iters=n_iters,
                                 temperature=temperature, epsilon=epsilon,
                                 lamb=lamb, tol=tol)
        targets = Q.cpu().argmax(dim=1).long()
        if verbose:
            counts = torch.bincount(targets, minlength=K).float()
            print(f"  [Sinkhorn] head {head_idx + 1}/{num_heads}: N={N} K={K} "
                  f"expected={N / K:.2f} mean={counts.mean():.3f} "
                  f"empty={(counts == 0).sum().item()} "
                  f"iters={info['iterations']} err={info['error']:.4f}",
                  flush=True)
        hard_targets[:, head_idx] = targets[inv_idx]

    if was_training:
        model.train()
    if num_heads == 1:
        return hard_targets[:, 0]
    return hard_targets
