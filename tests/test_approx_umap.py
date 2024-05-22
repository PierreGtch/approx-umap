import pytest
import numpy as np

from approx_umap import ApproxUMAP


@pytest.mark.parametrize("k,fn", [
    (0.5, 'inv'),
    (1, 'inv'),
    (2, 'inv'),
    (100, 'exp'),
    (1, lambda d: 1 / (d + 1e-8)),
])
def test_transform_same(k, fn):
    X = np.random.rand(100, 10)
    aumap = ApproxUMAP(n_neighbors=5, k=k, fn=fn)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    assert np.allclose(emb, emb2)


def test_exact_transform():
    X = np.random.rand(100, 10)
    aumap = ApproxUMAP(n_neighbors=5, k=1e-6)  # extreme k, approx should not work
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    emb3 = aumap.transform_exact(X)
    assert not np.allclose(emb, emb2)
    assert np.allclose(emb, emb3)


def test_not_enough_neigh():
    X = np.random.rand(5, 10)
    aumap = ApproxUMAP(n_neighbors=10)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    assert emb.shape == (5, 2)
    assert emb2.shape == (5, 2)
