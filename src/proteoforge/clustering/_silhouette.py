"""Silhouette scoring for precomputed condensed distances."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


def silhouette_samples_precomputed(
    condensed: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """
    Compute per-sample silhouette scores from condensed distances.

    Parameters
    ----------
    condensed
        Condensed distance vector.
    labels
        Cluster labels, shape ``(n_samples,)``. Must be contiguous 1-based
        cluster ids (as produced by ``cut_maxclust``).

    Returns
    -------
    ndarray of float64
        Silhouette score per sample. Singleton clusters receive 0.0.
    """
    distances = np.asarray(condensed, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.intp)
    n_samples = int(label_array.size)
    if n_samples == 0:
        return np.empty(0, dtype=np.float64)
    if np.unique(label_array).size < 2:
        return np.zeros(n_samples, dtype=np.float64)
    return np.asarray(
        _silhouette_samples_numba(distances, label_array),
        dtype=np.float64,
    )


@njit(cache=True)
def _pair_dist_condensed(
    condensed: npt.NDArray[np.float64],
    i: int,
    j: int,
    n: int,
) -> float:
    if i == j:
        return 0.0
    if i < j:
        return condensed[n * i - (i * (i + 1)) // 2 + (j - i - 1)]
    return condensed[n * j - (j * (j + 1)) // 2 + (i - j - 1)]


@njit(cache=True)
def _silhouette_samples_numba(
    condensed: npt.NDArray[np.float64],
    labels: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    condensed = np.ascontiguousarray(condensed)
    labels = np.ascontiguousarray(labels)
    n = labels.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    if n == 0:
        return scores

    max_label = 0
    for i in range(n):
        lab = labels[i]
        if lab > max_label:
            max_label = lab
    if max_label < 2:
        return scores

    n_clusters = max_label
    for i in range(n):
        li = labels[i]
        own_idx = li - 1
        a_sum = 0.0
        a_count = 0
        other_sum = np.zeros(n_clusters, dtype=np.float64)
        other_count = np.zeros(n_clusters, dtype=np.int64)

        for j in range(n):
            if i == j:
                continue
            dist = _pair_dist_condensed(condensed, i, j, n)
            lj = labels[j]
            if lj == li:
                a_sum += dist
                a_count += 1
            else:
                idx = lj - 1
                other_sum[idx] += dist
                other_count[idx] += 1

        if a_count == 0:
            continue

        a = a_sum / a_count
        b = np.inf
        for k in range(n_clusters):
            if k == own_idx:
                continue
            if other_count[k] > 0:
                mean_other = other_sum[k] / other_count[k]
                if mean_other < b:
                    b = mean_other

        denom = a if a > b else b
        if denom == 0.0:
            scores[i] = 0.0
        else:
            scores[i] = (b - a) / denom

    return scores
