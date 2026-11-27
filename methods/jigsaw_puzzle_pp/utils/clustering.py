"""Faiss k-means for the Jigsaw++ knowledge-transfer stage (Noroozi et al., 2018).

Extract the VGG16 pretext encoder's conv4 features for the whole dataset,
L2-normalise them, and k-means them into k clusters; the assignments are the
pseudo-labels a standard AlexNet is then trained to predict.

**faiss is the clustering backend**, as in the capture and the original repo:
the capture's `cluster_and_pseudolabels.py` fails fast rather than falling back
to CPU/sklearn, "because fallback cluster assignments are inconsistent". This
port keeps that -- a missing faiss errors loudly. faiss-gpu ships only a
linux-x86_64 wheel, so the knowledge-transfer stage is GPU / x86_64-linux only
(faiss lives in the CUDA lock, marked `# gpu-only` in requirements).
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _require_faiss():
    if not HAS_FAISS:
        raise ImportError(
            "the Jigsaw++ knowledge-transfer clustering requires faiss (the "
            "paper-target backend); install faiss-gpu. This stage is GPU / "
            "x86_64-linux only. The CPU/sklearn fallback is not ported because "
            "the capture rejects it as inconsistent.")


@torch.no_grad()
def extract_conv4_features(encoder, loader, device) -> np.ndarray:
    """VGG16 conv4 features (8192-d) for every image, as float32 (N, 8192)."""
    encoder.eval()
    parts = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        parts.append(encoder.get_conv4_features(imgs).cpu())
    return torch.cat(parts, dim=0).numpy().astype(np.float32)


def run_kmeans(features: np.ndarray, num_clusters: int, seed: int = 42,
               use_gpu: bool = True, verbose: bool = True):
    """L2-normalise the features and k-means them into num_clusters (faiss).

    Returns ``(assignments (N,) int64, centroids (k, D) float32)``.
    """
    _require_faiss()
    features = np.ascontiguousarray(features.astype(np.float32))
    faiss.normalize_L2(features)
    d = features.shape[1]

    clus = faiss.Clustering(d, num_clusters)
    clus.seed = int(seed)
    clus.niter = 20
    clus.max_points_per_centroid = 10_000_000

    if use_gpu and torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)
    else:
        index = faiss.IndexFlatL2(d)

    clus.train(features, index)
    _, assign = index.search(features, 1)
    assignments = assign.reshape(-1).astype(np.int64)
    centroids = faiss.vector_to_array(clus.centroids).reshape(num_clusters, d)
    if verbose:
        n_empty = num_clusters - len(np.unique(assignments))
        print(f"[KT-Clustering] k={num_clusters} D={d}  empty clusters: "
              f"{n_empty}", flush=True)
    return assignments, centroids
