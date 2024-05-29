import pytest
import numpy as np

from approx_umap import ApproxAlignedUMAP


def test_transform_same():
    X = np.random.rand(100, 10)
    aumap = ApproxAlignedUMAP(n_neighbors=5, alignment_regularisation=10)
    emb = aumap.fit_transform(X)
    emb2 = aumap.transform(X)
    emb3 = aumap.update_transform(X)
    emb4 = aumap.transform(np.concatenate([X, X], axis=0))
    assert np.allclose(emb, emb2)
    assert np.allclose(emb, emb3[:len(X)], atol=1)
