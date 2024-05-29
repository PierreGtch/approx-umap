# Authors: Pierre Guetschel
#
# This code heavily relies on the original AlignUMAP implementation by Leland McInnes.
#
# License: BSD 3 clause
import numpy as np
import numba
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from umap.umap_ import UMAP, make_epochs_per_sample
from umap.spectral import spectral_layout
from umap.layouts import optimize_layout_aligned_euclidean
from umap.aligned_umap import (
    expand_relations,
    build_neighborhood_similarities,
    procrustes_align,
    init_from_existing,
    set_aligned_params,
    get_nth_item_or_val,
    invert_dict,
    INT32_MIN,
    INT32_MAX,
)
from approx_umap import ApproxUMAP


class AlignedUMAP(BaseEstimator):
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
    ):

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds

        self.n_epochs = n_epochs
        self.init = init
        self.n_components = n_components
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.alignment_regularisation = alignment_regularisation
        self.alignment_window_size = alignment_window_size

        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.angular_rp_forest = angular_rp_forest
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose
        self.unique = unique

        self.a = a
        self.b = b

    def fit(self, X, y=None, **fit_params):
        if "relations" not in fit_params:
            raise ValueError(
                "Aligned UMAP requires relations between data to be " "specified"
            )

        self.dict_relations_ = fit_params["relations"]
        assert type(self.dict_relations_) in (list, tuple)
        assert type(X) in (list, tuple, np.ndarray)
        assert (len(X) - 1) == (len(self.dict_relations_))

        if y is not None:
            assert type(y) in (list, tuple, np.ndarray)
            assert (len(y) - 1) == (len(self.dict_relations_))
        else:
            y = [None] * len(X)

        # We need n_components to be constant or this won't work
        if type(self.n_components) in (list, tuple, np.ndarray):
            raise ValueError("n_components must be a single integer, and cannot vary")

        self.n_models_ = len(X)

        if self.n_epochs is None:
            self.n_epochs = 200

        n_epochs = self.n_epochs

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
            ).fit(X[n], y[n])
            for n in range(self.n_models_)
        ]

        window_size = fit_params.get("window_size", self.alignment_window_size)
        relations = expand_relations(self.dict_relations_, window_size)

        indptr_list = numba.typed.List.empty_list(numba.types.int32[::1])
        indices_list = numba.typed.List.empty_list(numba.types.int32[::1])
        heads = numba.typed.List.empty_list(numba.types.int32[::1])
        tails = numba.typed.List.empty_list(numba.types.int32[::1])
        epochs_per_samples = numba.typed.List.empty_list(numba.types.float64[::1])

        for mapper in self.mappers_:
            indptr_list.append(mapper.graph_.indptr)
            indices_list.append(mapper.graph_.indices)
            heads.append(mapper.graph_.tocoo().row)
            tails.append(mapper.graph_.tocoo().col)
            epochs_per_samples.append(
                make_epochs_per_sample(mapper.graph_.tocoo().data, n_epochs)
            )

        rng_state_transform = np.random.RandomState(self.transform_seed)
        regularisation_weights = build_neighborhood_similarities(
            indptr_list,
            indices_list,
            relations,
        )
        first_init = spectral_layout(
            self.mappers_[0]._raw_data,
            self.mappers_[0].graph_,
            self.n_components,
            rng_state_transform,
        )
        expansion = 10.0 / np.abs(first_init).max()
        first_embedding = (first_init * expansion).astype(
            np.float32,
            order="C",
        )

        embeddings = numba.typed.List.empty_list(numba.types.float32[:, ::1])
        embeddings.append(first_embedding)
        for i in range(1, self.n_models_):
            next_init = spectral_layout(
                self.mappers_[i]._raw_data,
                self.mappers_[i].graph_,
                self.n_components,
                rng_state_transform,
            )
            expansion = 10.0 / np.abs(next_init).max()
            next_embedding = (next_init * expansion).astype(
                np.float32,
                order="C",
            )
            anchor_data = relations[i][window_size - 1]
            left_anchors = anchor_data[anchor_data >= 0]
            right_anchors = np.where(anchor_data >= 0)[0]
            embeddings.append(
                procrustes_align(
                    embeddings[-1],
                    next_embedding,
                    np.vstack([left_anchors, right_anchors]),
                )
            )

        seed_triplet = rng_state_transform.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        self.embeddings_ = optimize_layout_aligned_euclidean(
            embeddings,
            embeddings,
            heads,
            tails,
            n_epochs,
            epochs_per_samples,
            regularisation_weights,
            relations,
            seed_triplet,
            lambda_=self.alignment_regularisation,
            move_other=True,
        )

        for i, embedding in enumerate(self.embeddings_):
            disconnected_vertices = (
                    np.array(self.mappers_[i].graph_.sum(axis=1)).flatten() == 0
            )
            embedding[disconnected_vertices] = np.full(self.n_components, np.nan)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.embeddings_

    def update(self, X, y=None, **fit_params):
        if "relations" not in fit_params:
            raise ValueError(
                "Aligned UMAP requires relations between data to be " "specified"
            )

        new_dict_relations = fit_params["relations"]
        assert isinstance(new_dict_relations, dict)

        X = check_array(X)

        self.__dict__ = set_aligned_params(fit_params, self.__dict__, self.n_models_)

        # We need n_components to be constant or this won't work
        if type(self.n_components) in (list, tuple, np.ndarray):
            raise ValueError("n_components must be a single integer, and cannot vary")

        if self.n_epochs is None:
            self.n_epochs = 200

        n_epochs = self.n_epochs

        new_mapper = UMAP(
            n_neighbors=get_nth_item_or_val(self.n_neighbors, self.n_models_),
            min_dist=get_nth_item_or_val(self.min_dist, self.n_models_),
            n_epochs=get_nth_item_or_val(self.n_epochs, self.n_models_),
            repulsion_strength=get_nth_item_or_val(
                self.repulsion_strength, self.n_models_
            ),
            learning_rate=get_nth_item_or_val(self.learning_rate, self.n_models_),
            init=self.init,
            spread=get_nth_item_or_val(self.spread, self.n_models_),
            negative_sample_rate=get_nth_item_or_val(
                self.negative_sample_rate, self.n_models_
            ),
            local_connectivity=get_nth_item_or_val(
                self.local_connectivity, self.n_models_
            ),
            set_op_mix_ratio=get_nth_item_or_val(self.set_op_mix_ratio, self.n_models_),
            unique=get_nth_item_or_val(self.unique, self.n_models_),
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
        ).fit(X, y)

        self.n_models_ += 1
        self.mappers_ += [new_mapper]

        self.dict_relations_ += [new_dict_relations]

        window_size = fit_params.get("window_size", self.alignment_window_size)
        new_relations = expand_relations(self.dict_relations_, window_size)

        indptr_list = numba.typed.List.empty_list(numba.types.int32[::1])
        indices_list = numba.typed.List.empty_list(numba.types.int32[::1])
        heads = numba.typed.List.empty_list(numba.types.int32[::1])
        tails = numba.typed.List.empty_list(numba.types.int32[::1])
        epochs_per_samples = numba.typed.List.empty_list(numba.types.float64[::1])

        for i, mapper in enumerate(self.mappers_):
            indptr_list.append(mapper.graph_.indptr)
            indices_list.append(mapper.graph_.indices)
            heads.append(mapper.graph_.tocoo().row)
            tails.append(mapper.graph_.tocoo().col)
            if i == len(self.mappers_) - 1:
                epochs_per_samples.append(
                    make_epochs_per_sample(mapper.graph_.tocoo().data, n_epochs)
                )
            else:
                epochs_per_samples.append(
                    np.full(mapper.embedding_.shape[0], n_epochs + 1, dtype=np.float64)
                )

        new_regularisation_weights = build_neighborhood_similarities(
            indptr_list,
            indices_list,
            new_relations,
        )

        # TODO: We can likely make this more efficient and not recompute each time
        inv_dict_relations = invert_dict(new_dict_relations)

        new_embedding = init_from_existing(
            self.embeddings_[-1], new_mapper.graph_, inv_dict_relations
        )

        self.embeddings_.append(new_embedding)

        rng_state_transform = np.random.RandomState(self.transform_seed)
        seed_triplet = rng_state_transform.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        self.embeddings_ = optimize_layout_aligned_euclidean(
            self.embeddings_,
            self.embeddings_,
            heads,
            tails,
            n_epochs,
            epochs_per_samples,
            new_regularisation_weights,
            new_relations,
            seed_triplet,
            lambda_=self.alignment_regularisation,
        )
