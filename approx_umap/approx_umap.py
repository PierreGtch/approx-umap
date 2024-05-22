# Authors: Pierre Guetschel
#          Peter Wassenaar
#
# License: BSD 3 clause
import numpy as np
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from docstring_inheritance import NumpyDocstringInheritanceMeta


class ApproxUMAP(UMAP, metaclass=NumpyDocstringInheritanceMeta):
    """Approximate UMAP

    Parameters
    ----------
    k: float
        Temperature parameter.
    fn: str | Callable[[ndarray], ndarray]
        Function to apply to the distances before computing the weights.
        If a string, can be 'inv' for 1/d or 'exp'  for exp(-d).
    """

    def __init__(
            self,
            n_neighbors=15,
            n_components=2,
            metric="euclidean",
            metric_kwds=None,
            output_metric="euclidean",
            output_metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init="spectral",
            min_dist=0.1,
            spread=1.0,
            low_memory=True,
            n_jobs=-1,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            random_state=None,
            angular_rp_forest=False,
            target_n_neighbors=-1,
            target_metric="categorical",
            target_metric_kwds=None,
            target_weight=0.5,
            transform_seed=42,
            transform_mode="embedding",
            force_approximation_algorithm=False,
            verbose=False,
            tqdm_kwds=None,
            unique=False,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            output_dens=False,
            disconnection_distance=None,
            precomputed_knn=(None, None, None),
            k=1,
            fn="inv",
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            metric_kwds=metric_kwds,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            min_dist=min_dist,
            spread=spread,
            low_memory=low_memory,
            n_jobs=n_jobs,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            transform_queue_size=transform_queue_size,
            a=a,
            b=b,
            random_state=random_state,
            angular_rp_forest=angular_rp_forest,
            target_n_neighbors=target_n_neighbors,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            target_weight=target_weight,
            transform_seed=transform_seed,
            transform_mode=transform_mode,
            force_approximation_algorithm=force_approximation_algorithm,
            verbose=verbose,
            tqdm_kwds=tqdm_kwds,
            unique=unique,
            densmap=densmap,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
            output_dens=output_dens,
            disconnection_distance=disconnection_distance,
            precomputed_knn=precomputed_knn,
        )
        self.k = k
        self._fn = fn
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

    def fn(self, d):
        epsilon = 1e-8
        if self._fn == "inv":
            return 1 / (d + epsilon)
        if self._fn == "exp":
            return np.exp(-d)
        return self._fn(d)

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
        neigh_sim = self.fn(neigh_dist * self.k)
        emb = np.sum(neigh_sim[:, :, None] / neigh_sim.sum(axis=1)[:, None, None] * neigh_emb,
                     axis=1)
        return emb
