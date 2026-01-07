import numpy as np
import torch
from hierarchy_transformers import HierarchyTransformer

from evaluation.config import PROJECT_DIR
from granuscore.utils import get_best_device


def init_radial_anchors(model, original_vectors, K: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # ---- 1) Compute dist0 once ----
    original_dist0 = (
        model.manifold
        .dist0(torch.from_numpy(original_vectors))
        .detach()
        .cpu()
        .numpy()
    )

    # ---- 2) Build radial bins via quantiles ----
    q1, q2 = np.quantile(original_dist0, [0.33, 0.66])

    bins = [
        np.where(original_dist0 <= q1)[0],
        np.where((original_dist0 > q1) & (original_dist0 <= q2))[0],
        np.where(original_dist0 > q2)[0],
    ]

    # ---- 3) Sample fixed number per bin ----
    n_bins = len(bins)
    per_bin = K // n_bins

    anchor_indices = []

    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue

        size = min(per_bin, len(bin_indices))
        sampled = rng.choice(bin_indices, size=size, replace=False)
        anchor_indices.append(sampled)

    anchor_indices = np.concatenate(anchor_indices)

    # ---- 4) Fill remainder if K not exactly divisible ----
    if len(anchor_indices) < K:
        remaining = np.setdiff1d(
            np.arange(original_vectors.shape[0]),
            anchor_indices,
            assume_unique=False,
        )

        extra = rng.choice(remaining, size=K - len(anchor_indices), replace=False)
        anchor_indices = np.concatenate([anchor_indices, extra])

    anchor_embeddings = original_vectors[anchor_indices]
    np.save("50k-hit-anchor_embeddings.npy", anchor_embeddings.astype(np.float32))

if __name__ == '__main__':
    seed = 42
    K = 999  # should be dividable by 3, otherwise last positions filled up by random
    model = HierarchyTransformer.from_pretrained('Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun').to(
            get_best_device()
        )
    original_vectors = np.load(f"{PROJECT_DIR}/src/granuscore/assets/faiss/50k-hit-original-vectors.npy")

    init_radial_anchors(model, original_vectors, K, seed)