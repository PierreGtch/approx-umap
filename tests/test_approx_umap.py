import numpy as np

from approx_umap import ApproxUMAP


def test_transform_same():
    X = np.random.rand(100, 10)
    aumap = ApproxUMAP(n_neighbors=5)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    assert np.allclose(emb, emb2)


def test_not_enough_neigh():
    X = np.random.rand(5, 10)
    aumap = ApproxUMAP(n_neighbors=10)
    emb = aumap.fit_transform(X)
    assert emb.shape == (5, 2)
