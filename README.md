# Approximate UMAP

Modification of the UMAP algorithm to allow for fast approximate projections of
new data points.

## Description

This package provides a class `ApproxUMAP` that allows for fast approximate projections of new data points in the target
space.

The `fit` and `fit_transform` methods of `ApproxUMAP` are nearly identical to those of `umap.UMAP`;
they simply fit an additional `sklearn.neighbors.NearestNeighbors` estimator.

Only the `transform` method significantly differs; it approximates the projection of new data points
in the embedding space to improve the projection speed.
The projections are approximated by finding the nearest neighbors in the
source space and computing their weighted average in the embedding space.
The weights are the inverse of the distances in the source space.

Formally, the projection of a new point $x$ is approximated as follows:
$$u=\sum_i^k\frac{\frac{1}{d_i}}{\sum_j^k\frac{1}{d_j}}u_i$$
with $x_1\dots x_k$ the $k$ nearest neighbours of $x$ in the source space
among the points used for training (i.e., passed to `fit` or `fit_transform`),
$d_i=distance(x, x_i)$, and $u_1\dots u_i$ the exact UMAP projections of $x_1\dots x_k$.

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
emb_exact = ApproxUMAP().fit_transform(X)  # exact UMAP projections
emb_approx = ApproxUMAP().fit(X).transform(X)  # approximate UMAP projection
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