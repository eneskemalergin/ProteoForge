"""Dendrogram cut strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.clustering._linkage import cut_maxclust, cut_maxclust_numba
from proteoforge.clustering._silhouette import silhouette_samples_precomputed

if TYPE_CHECKING:
    from proteoforge._config import Config


def _max_clusters_arg(max_clusters: int | None, n_samples: int) -> int:
    return -1 if max_clusters is None else min(max_clusters, n_samples - 1)


@dataclass(frozen=True)
class DynamicTreeCut:
    """Fast dendrogram cut based on merge-distance acceleration."""

    name: str = "dynamic_tree"

    def cut(
        self,
        linkage_matrix: npt.NDArray[np.float64],
        condensed_dist: npt.NDArray[np.float64],
        *,
        n_samples: int,
        config: Config,
    ) -> npt.NDArray[np.intp]:
        del condensed_dist
        return np.asarray(
            dynamic_tree_labels_numba(
                np.asarray(linkage_matrix, dtype=np.float64),
                n_samples,
                config.cluster_min_clusters,
                _max_clusters_arg(config.cluster_max_clusters, n_samples),
            ),
            dtype=np.intp,
        )


@dataclass(frozen=True)
class FixedHeightCut:
    """Fixed cluster count from ``config.fixed_n_clusters``."""

    name: str = "fixed_height"

    def cut(
        self,
        linkage_matrix: npt.NDArray[np.float64],
        condensed_dist: npt.NDArray[np.float64],
        *,
        n_samples: int,
        config: Config,
    ) -> npt.NDArray[np.intp]:
        del condensed_dist
        n_clusters = max(config.cluster_min_clusters, config.fixed_n_clusters)
        return cut_maxclust(
            linkage_matrix,
            n_samples=n_samples,
            n_clusters=min(n_clusters, n_samples),
        )


@dataclass(frozen=True)
class HybridOutlierCut:
    """
    Dendrogram cut followed by low-silhouette singleton isolation.

    Avoids a sklearn runtime dependency while matching reference hybrid behavior.
    """

    name: str = "hybrid_outlier"

    def cut(
        self,
        linkage_matrix: npt.NDArray[np.float64],
        condensed_dist: npt.NDArray[np.float64],
        *,
        n_samples: int,
        config: Config,
    ) -> npt.NDArray[np.intp]:
        initial = DynamicTreeCut().cut(
            linkage_matrix,
            condensed_dist,
            n_samples=n_samples,
            config=config,
        )
        if np.unique(initial).size < 2:
            return initial

        scores = silhouette_samples_precomputed(condensed_dist, initial)
        return relabel_hybrid_outliers(
            initial,
            scores,
            threshold=config.hybrid_outlier_threshold,
        )


def relabel_hybrid_outliers(
    labels: npt.NDArray[np.intp],
    scores: npt.NDArray[np.float64],
    *,
    threshold: float,
) -> npt.NDArray[np.intp]:
    """
    Isolate low-silhouette samples into singleton clusters.

    Parameters
    ----------
    labels
        1-based cluster labels, shape ``(n_samples,)``.
    scores
        Per-sample silhouette scores, shape ``(n_samples,)``.
    threshold
        Samples below this score are candidates for isolation.

    Returns
    -------
    ndarray of intp
        Relabeled cluster assignments. A sample is split out only when its
        current cluster has more than one member.
    """
    return np.asarray(
        relabel_hybrid_outliers_numba(
            np.asarray(labels, dtype=np.intp),
            np.asarray(scores, dtype=np.float64),
            threshold,
        ),
        dtype=np.intp,
    )


def select_cut_strategy(
    name: str,
) -> DynamicTreeCut | FixedHeightCut | HybridOutlierCut:
    """
    Return the cut strategy for a config ``cut`` name.

    Parameters
    ----------
    name
        One of ``dynamic_tree``, ``fixed_height``, or ``hybrid_outlier``.

    Returns
    -------
    DynamicTreeCut, FixedHeightCut, or HybridOutlierCut
        Matching cut implementation.

    Raises
    ------
    ProteoForgeValidationError
        If ``name`` is not a supported cut strategy.
    """
    if name == "hybrid_outlier":
        return HybridOutlierCut()
    if name == "dynamic_tree":
        return DynamicTreeCut()
    if name == "fixed_height":
        return FixedHeightCut()
    valid = ["dynamic_tree", "fixed_height", "hybrid_outlier"]
    msg = f"cut '{name}' is not supported. Valid: {valid}."
    raise ProteoForgeValidationError(msg)


@njit(cache=True)
def dendrogram_cluster_count_numba(
    linkage: npt.NDArray[np.float64],
    n_samples: int,
    min_clusters: int,
    max_clusters: int,
) -> int:
    """Merge-distance acceleration cluster count (``max_clusters < 0`` = no cap)."""
    if n_samples <= 1:
        return 1
    if max_clusters < 0:
        effective_max = n_samples - 1
    else:
        effective_max = min(max_clusters, n_samples - 1)

    n_merges = linkage.shape[0]
    if n_merges < 2:
        n_clusters = min_clusters
    else:
        accel_size = n_merges - 2
        if accel_size <= 0:
            n_clusters = min_clusters
        else:
            best_idx = 0
            best_val = -np.inf
            for i in range(accel_size):
                d0 = linkage[i + 2, 2] - 2.0 * linkage[i + 1, 2] + linkage[i, 2]
                if d0 > best_val:
                    best_val = d0
                    best_idx = i
            n_clusters = n_samples - (best_idx + 2)

    if n_clusters < min_clusters:
        n_clusters = min_clusters
    if n_clusters > effective_max:
        n_clusters = effective_max
    return n_clusters


@njit(cache=True)
def dynamic_tree_labels_numba(
    linkage: npt.NDArray[np.float64],
    n_samples: int,
    min_clusters: int,
    max_clusters: int,
) -> npt.NDArray[np.int64]:
    k = dendrogram_cluster_count_numba(
        linkage,
        n_samples,
        min_clusters,
        max_clusters,
    )
    return cut_maxclust_numba(linkage, n_samples, k)


@njit(cache=True)
def relabel_hybrid_outliers_numba(
    labels: npt.NDArray[np.intp],
    scores: npt.NDArray[np.float64],
    threshold: float,
) -> npt.NDArray[np.intp]:
    n = labels.shape[0]
    n_unique = 0
    seen = np.zeros(n + 1, dtype=np.int8)
    max_label = 0
    for i in range(n):
        lab = labels[i]
        if lab > max_label:
            max_label = lab
        if seen[lab] == 0:
            seen[lab] = 1
            n_unique += 1
    if n_unique < 2:
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = labels[i]
        return out

    has_outlier = False
    for i in range(n):
        if scores[i] < threshold:
            has_outlier = True
            break
    if not has_outlier:
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = labels[i]
        return out

    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = labels[i]

    sizes = np.zeros(max_label + n + 1, dtype=np.int64)
    for i in range(n):
        sizes[labels[i]] += 1

    next_cluster = max_label + 1
    for i in range(n):
        if scores[i] >= threshold:
            continue
        label = out[i]
        if sizes[label] > 1:
            out[i] = next_cluster
            sizes[label] -= 1
            sizes[next_cluster] = 1
            next_cluster += 1
    return out
