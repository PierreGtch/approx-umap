# Authors: Pierre Guetschel
#
# This code heavily relies on the original AlignUMAP implementation by Leland McInnes.
#
# License: BSD 3 clause
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import clone
from docstring_inheritance import NumpyDocstringInheritanceMeta
from umap import AlignedUMAP, UMAP
from umap.aligned_umap import get_nth_item_or_val

from approx_umap.functions import approximate_embedding


class ApproxAlignedUMAP(AlignedUMAP, metaclass=NumpyDocstringInheritanceMeta):
    """Approximate aligned UMAP

    Parameters
    ----------
    k: float
        Temperature parameter.
    fn: str | Callable[[ndarray], ndarray]
        Function to apply to the distances before computing the weights.
        If a string, can be 'inv' for 1/d or 'exp'  for exp(-d).
    copy_data: bool
        When fitting or updating ApproxAlignedUMAP, the training data is stored.
        By default it is copied. If set to False, the data is not copied.
    """

    def __init__(
            self,
            n_neighbors=15,
            n_components=2,
            metric="euclidean",
            metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init="spectral",
            alignment_regularisation=1.0e-2,
            alignment_window_size=3,
            min_dist=0.1,
            spread=1.0,
            low_memory=False,
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
            force_approximation_algorithm=False,
            verbose=False,
            unique=False,
            k=1,
            fn="inv",
            copy_data=True,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            metric_kwds=metric_kwds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            alignment_regularisation=alignment_regularisation,
            alignment_window_size=alignment_window_size,
            min_dist=min_dist,
            spread=spread,
            low_memory=low_memory,
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
            force_approximation_algorithm=force_approximation_algorithm,
            verbose=verbose,
            unique=unique,
        )
        self.k = k
        self.fn = fn

        self.copy_data = copy_data
        self._knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            # radius=1.0,
            # leaf_size=30,
            algorithm="auto",
            metric=self.metric,
            # p=2, Use  metric_params={'p': 3} instead
            metric_params=self.metric_kwds,
        )

    def fit(self, X, y=None, **fit_params):

        self.dict_relations_ = []
        assert type(X) == np.ndarray

        if y is not None:
            assert type(y) == np.ndarray

        # We need n_components to be constant or this won't work
        if type(self.n_components) in (list, tuple, np.ndarray):
            raise ValueError("n_components must be a single integer, and cannot vary")

        self.n_models_ = 1

        if self.n_epochs is None:
            self.n_epochs = 200

        n = 0
        self.mappers_ = [
            UMAP(
                n_neighbors=get_nth_item_or_val(self.n_neighbors, n),
                min_dist=get_nth_item_or_val(self.min_dist, n),
                n_epochs=get_nth_item_or_val(self.n_epochs, n),
                repulsion_strength=get_nth_item_or_val(self.repulsion_strength, n),
                learning_rate=get_nth_item_or_val(self.learning_rate, n),
                init=self.init,
                spread=get_nth_item_or_val(self.spread, n),
                negative_sample_rate=get_nth_item_or_val(self.negative_sample_rate, n),
                local_connectivity=get_nth_item_or_val(self.local_connectivity, n),
                set_op_mix_ratio=get_nth_item_or_val(self.set_op_mix_ratio, n),
                unique=get_nth_item_or_val(self.unique, n),
                n_components=self.n_components,
                metric=self.metric,
                metric_kwds=self.metric_kwds,
                low_memory=self.low_memory,
                random_state=self.random_state,
                angular_rp_forest=self.angular_rp_forest,
                transform_queue_size=self.transform_queue_size,
                target_n_neighbors=self.target_n_neighbors,
                target_metric=self.target_metric,
                target_metric_kwds=self.target_metric_kwds,
                target_weight=self.target_weight,
                transform_seed=self.transform_seed,
                force_approximation_algorithm=self.force_approximation_algorithm,
                verbose=self.verbose,
                a=self.a,
                b=self.b,
            ).fit(X, y),
        ]

        self.embeddings_ = [self.mappers_[0].embedding_]

        self._save_Xy(X, y)
        self._knn.fit(X)

        return self

    def _save_Xy(self, X, y):
        self.last_X_ = X.copy() if self.copy_data else X
        self.last_y_ = y.copy() if self.copy_data and y is not None else y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.embeddings_[-1]

    def aligned_fit(self, X, y=None, **fit_params):
        super().update(X, y, **fit_params)
        self._save_Xy(X, y)
        self._knn = clone(self._knn).fit(X)
        return self

    def update(self, X, y=None, **fit_params):
        """ Update the model with new data.

        A new ApproxUMAP model is fitted to the new data X plus all the pre-existing data.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The new data to be added to the model.
        """
        if self.last_X_ is None:
            raise ValueError("Must fit model before updating")
        full_X = np.concatenate([self.last_X_, X], axis=0)
        if y is not None:
            assert self.last_y_ is not None, "y was not provided during last fit"
            full_y = np.concatenate([self.last_y_, y], axis=0)
        else:
            full_y = None
        relations = {i: i for i in range(len(self.last_X_))}
        return self.aligned_fit(full_X, full_y, relations=relations, **fit_params)

    def update_transform(self, X, y=None, **fit_params):
        self.update(X, y, **fit_params)
        return self.embeddings_[-1]

    def transform(self, X):
        """Transform X into the las update of the embedded space using the approximate
        UMAP algorithm and return that transformed output.

        The projections are approximated by finding the nearest neighbors in the
        source space and computing their weighted average in the embedding space.
        The weights are the inverse of the distances in the source space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Approximate embedding of the new data in low-dimensional space.
        """
        return approximate_embedding(X, self._knn, self.embeddings_[-1], self.k, self.fn)
