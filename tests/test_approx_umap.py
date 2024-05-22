import pytest
import numpy as np

from approx_umap import ApproxUMAP


@pytest.mark.parametrize("k", [0.5, 1, 2])
def test_transform_same(k):
    X = np.random.rand(100, 10)
    aumap = ApproxUMAP(n_neighbors=5, k=k)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    assert np.allclose(emb, emb2)


def test_not_enough_neigh():
    X = np.random.rand(5, 10)
    aumap = ApproxUMAP(n_neighbors=10)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    assert emb.shape == (5, 2)
    assert emb2.shape == (5, 2)
