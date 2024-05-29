from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors


def apply_fn(fn: str | Callable[[np.ndarray], np.ndarray], d: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    if fn == "inv":
        return 1 / (d + epsilon)
    if fn == "exp":
        return np.exp(-d) + epsilon
    return fn(d)


def approximate_embedding(
        X: np.ndarray,
        knn: NearestNeighbors,
        embedding: np.ndarray,
        k: float,
        fn: str | Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    assert knn.n_samples_fit_ == embedding.shape[0]
    n_neighbors = min(knn.n_neighbors, embedding.shape[0])
    neigh_dist, neigh_ind = knn.kneighbors(
        X, n_neighbors=n_neighbors, return_distance=True)
    neigh_emb = embedding[neigh_ind]
    neigh_sim = apply_fn(fn, neigh_dist * k)
    emb = np.sum(neigh_sim[:, :, None] / neigh_sim.sum(axis=1)[:, None, None] * neigh_emb,
                 axis=1)
    return emb
