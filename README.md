# Approximate UMAP

Modification of the UMAP algorithm to allow for fast approximate projections of
new data points.

## Description

This package provides the classes `ApproxUMAP` and `ApproxAlignedUMAP` that allow for fast approximate projections of
new data points in the target
space.

The `fit` and `fit_transform` methods of `ApproxUMAP` are nearly identical to those of `umap.UMAP`;
they simply fit an additional `sklearn.neighbors.NearestNeighbors` estimator.

Only the `transform` method significantly differs; it approximates the projection of new data points
in the embedding space to improve the projection speed.
The projections are approximated by finding the nearest neighbors in the
source space and computing their weighted average in the embedding space.
The weights are the inverse of the distances in the source space.

Formally, the projection of a new point $x$ is approximated as follows:
$$u=\sum_i^k\frac{f(k d_i)}{\sum_j^kf(k d_j)}u_i$$
with $x_1\dots x_k$ the $k$ nearest neighbours of $x$ in the source space
among the points used for training (i.e., passed to `fit` or `fit_transform`),
$d_i=distance(x, x_i)$, $u_1\dots u_i$ the exact UMAP projections of $x_1\dots x_k$, and $k$ the temperature parameter.
The function $f(\cdot)$ corresponds to $\frac{1}{\cdot}$ if `fn='inv'`, and to $\frac{1}{e^{\cdot}}$ if `fn='exp'`.

The original behavior of UMAP's `transform` method can be obtained using the `transform_exact` method.

## Installation

The package can be installed via pip:

```bash
pip install approx-umap
```

## Usage

The usage of `ApproxUMAP` is similar to that of any [scikit-learn](https://scikit-learn.org/stable/index.html)
transformer:

```python
import numpy as np
from approx_umap import ApproxUMAP

X = np.random.rand(100, 10)

emb_exact = ApproxUMAP(fn='exp', k=1).fit_transform(X)  # exact UMAP projections

projector = ApproxUMAP(fn='exp', k=1).fit(X)
emb_approx = projector.transform(X)  # approximate UMAP projection
emb_approx_exact = projector.transform_exact(X)  # exact UMAP projection
```

The class `ApproxAlignedUMAP` additionally implements the methods `update` and `update_transform`
to created aligned embeddings of new data points with respect to the training data.

```python
import numpy as np
from approx_umap import ApproxAlignedUMAP

X = np.random.rand(100, 10)
X_new = np.random.rand(10, 10)

emb_exact = ApproxAlignedUMAP(fn='exp', k=1).fit_transform(X)  # exact UMAP projections

projector = ApproxAlignedUMAP(fn='exp', k=1).fit(X)

emb_aligned = projector.update_transform(X_new)  # exact aligned UMAP projections
assert emb_aligned.shape[0] == X.shape[0] + X_new.shape[0]  # returns the aligned embeddings of the whole history

emb_approx_aligned = projector.transform(X_new)  # approximate aligned UMAP projections
```

## Citation

Please, cite this work as:

```bibtex
@inproceedings{approx-umap2024,
    title = {Approximate UMAP allows for high-rate online visualization of high-dimensional data streams},
    author = {Peter Wassenaar and Pierre Guetschel and Michael Tangermann},
    year = {2024},
    month = {September},
    booktitle = {9th Graz Brain-Computer Interface Conference},
    address = {Graz, Austria},
    url = {https://arxiv.org/abs/2404.04001},
}
```