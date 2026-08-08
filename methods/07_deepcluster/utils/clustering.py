"""K-means clustering for DeepCluster (Caron et al., 2018), ported from the lab's
own faiss implementation.

Pipeline (the paper's, unchanged): extract fc7 features -> L2-normalise ->
PCA + whitening (pca_dim) -> L2-normalise -> k-means (k clusters) -> assignments.

**faiss is the clustering backend**, as in the capture and the original
DeepCluster repo. The capture ships faiss as the required paper-target path and
marks its sklearn fallback "not the official DeepCluster protocol"; this port
commits to faiss and drops the fallback, so a missing faiss errors loudly rather
than silently switching to a different numerical path. faiss-gpu has a
linux-x86_64-only wheel, which is why this method is GPU / x86_64-linux only.

The lab wrapper gathers features across DDP ranks before clustering; the
single-process port clusters the whole (local) feature matrix directly.
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
            "DeepCluster's clustering requires faiss (the paper-target backend "
            "the capture and the original repo use). Install faiss-gpu; this "
            "method is GPU / x86_64-linux only. The sklearn fallback is not "
            "ported because the capture marks it 'not the official protocol'.")


@torch.no_grad()
def extract_features_for_clustering(model, loader, device) -> np.ndarray:
    """Forward the dataset through the backbone (no grad) and return the fc7
    feature matrix (N, 4096) as float32."""
    model.eval()
    parts = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        parts.append(model.get_features(imgs, before_final_relu=True).cpu())
    return torch.cat(parts, dim=0).numpy().astype(np.float32)


def run_kmeans(features: np.ndarray, k: int, pca_dim: int = 256,
               use_gpu: bool = True, seed: int = 42, verbose: bool = True):
    """PCA-whiten the features and k-means them into k clusters (faiss).

    Returns:
        assignments : np.ndarray (N,) int64 cluster index per sample
        centroids   : np.ndarray (k, pca_dim) float32
    """
    _require_faiss()
    features = np.ascontiguousarray(features.astype(np.float32))
    faiss.normalize_L2(features)

    pca = faiss.PCAMatrix(features.shape[1], pca_dim, -0.5)  # -0.5 = whitening
    pca.train(features)
    assert pca.is_trained
    features_pca = pca.apply_py(features)
    faiss.normalize_L2(features_pca)

    clus = faiss.Clustering(pca_dim, k)
    clus.seed = int(seed)
    clus.niter = 20
    clus.max_points_per_centroid = 10_000_000

    if use_gpu and torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, pca_dim, cfg)
    else:
        index = faiss.IndexFlatL2(pca_dim)

    clus.train(features_pca, index)
    _, assign = index.search(features_pca, 1)
    assignments = assign.reshape(-1).astype(np.int64)
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, pca_dim)
    if verbose:
        n_empty = k - len(np.unique(assignments))
        print(f"[Clustering] k={k} pca_dim={pca_dim}  empty clusters: {n_empty}",
              flush=True)
    return assignments, centroids
