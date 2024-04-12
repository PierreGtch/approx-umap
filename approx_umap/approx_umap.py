# Authors: Pierre Guetschel
#          Peter Wassenaar
#
# License: BSD 3 clause
import numpy as np
from umap import UMAP
from sklearn.neighbors import NearestNeighbors


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class ApproxUMAP(UMAP, metaclass=PostInitCaller):

    def __post_init__(self):
        self._knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            # radius=1.0,
            # leaf_size=30,
            algorithm="auto",
            metric=self.metric,
            # p=2, Use  metric_params={'p': 3} instead
            metric_params=self.metric_kwds,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y=None, force_all_finite=True):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.

        force_all_finite : Whether to raise an error on np.inf, np.nan, pd.NA in array.
            The possibilities are: - True: Force all values of array to be finite.
                                   - False: accepts np.inf, np.nan, pd.NA in array.
                                   - 'allow-nan': accepts only np.nan and pd.NA values in array.
                                     Values cannot be infinite.
        """
        super().fit(X, y, force_all_finite)
        self._knn.fit(X)
        return self

    def fit_transform(self, X, y=None, force_all_finite=True):
        """Fit X into an embedded space and return that transformed
        output.

        The transformation is the exact UMAP transformation of the data
        not the approximate version returned by the transform method.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.

        force_all_finite : Whether to raise an error on np.inf, np.nan, pd.NA in array.
            The possibilities are: - True: Force all values of array to be finite.
                                   - False: accepts np.inf, np.nan, pd.NA in array.
                                   - 'allow-nan': accepts only np.nan and pd.NA values in array.
                                     Values cannot be infinite.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.

        or a tuple (X_new, r_orig, r_emb) if ``output_dens`` flag is set,
        which additionally includes:

        r_orig: array, shape (n_samples)
            Local radii of data points in the original data space (log-transformed).

        r_emb: array, shape (n_samples)
            Local radii of data points in the embedding (log-transformed).
        """
        emb = super().fit_transform(X, y, force_all_finite)
        self._knn.fit(X)
        return emb

    def transform(self, X):
        """Transform X into the existing embedded space using the approximate
        UMAP algorithm and return that transformed output.

        The projections are approximated by finding the nearest neighbors in the
        source space and computing their weighted average in the embedding space.
        The weights are the inverse of the distances in the source space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        force_all_finite : Whether to raise an error on np.inf, np.nan, pd.NA in array.
            The possibilities are: - True: Force all values of array to be finite.
                                   - False: accepts np.inf, np.nan, pd.NA in array.
                                   - 'allow-nan': accepts only np.nan and pd.NA values in array.
                                     Values cannot be infinite.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Approximate embedding of the new data in low-dimensional space.
        """
        n_neighbors = min(self._knn.n_neighbors, self.embedding_.shape[0])
        neigh_dist, neigh_ind = self._knn.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=True)
        neigh_emb = self.embedding_[neigh_ind]
        epsilon = 1e-8
        neigh_sim = 1 / (neigh_dist + epsilon)
        emb = np.sum(neigh_sim[:, :, None] / neigh_sim.sum(axis=1)[:, None, None] * neigh_emb,
                     axis=1)
        return emb
