"""Ward linkage and maxclust cuts for per-protein peptide profile matrices."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from numba import njit


def ward_linkage(
    condensed: npt.NDArray[np.float64],
    *,
    n_samples: int,
) -> npt.NDArray[np.float64]:
    """
    Build a scipy-compatible Ward linkage matrix.

    Uses the nearest-neighbor-chain algorithm (Müllner 2011), matching
    ``scipy.cluster.hierarchy.linkage(..., method='ward')`` on condensed
    Euclidean distances.

    Parameters
    ----------
    condensed
        Condensed Euclidean distances.
    n_samples
        Number of peptides represented by ``condensed``.

    Returns
    -------
    ndarray of float64
        Linkage matrix with columns ``idx1, idx2, distance, count``.
    """
    distances = np.asarray(condensed, dtype=np.float64)
    expected = n_samples * (n_samples - 1) // 2
    if distances.size != expected:
        msg = (
            f"condensed distance length {distances.size} does not match "
            f"n_samples={n_samples}."
        )
        raise ValueError(msg)
    if n_samples < 1:
        msg = "n_samples must be at least 1."
        raise ValueError(msg)
    if n_samples == 1:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(
        _ward_linkage_numba_full(distances, n_samples),
        dtype=np.float64,
    )


def cut_maxclust(
    linkage_matrix: npt.NDArray[np.float64],
    *,
    n_samples: int,
    n_clusters: int,
) -> npt.NDArray[np.intp]:
    """
    Cut a linkage matrix to at most ``n_clusters`` labels.

    Parameters
    ----------
    linkage_matrix
        Linkage matrix, shape ``(n_samples - 1, 4)``.
    n_samples
        Number of original observations.
    n_clusters
        Desired cluster count.

    Returns
    -------
    ndarray of intp
        1-based cluster labels in original observation order.
    """
    if n_samples < 1:
        msg = "n_samples must be at least 1."
        raise ValueError(msg)
    target = max(1, min(int(n_clusters), n_samples))
    return np.asarray(
        cut_maxclust_numba(
            np.asarray(linkage_matrix, dtype=np.float64),
            n_samples,
            target,
        ),
        dtype=np.intp,
    )


@njit(cache=True)
def _condensed_index_sym(i: int, j: int, n: int) -> int:
    if i < j:
        return n * i - (i * (i + 1)) // 2 + (j - i - 1)
    return n * j - (j * (j + 1)) // 2 + (i - j - 1)


@njit(cache=True)
def _ward_distance_update(
    d_xi: float,
    d_yi: float,
    d_xy: float,
    size_x: int,
    size_y: int,
    size_i: int,
) -> float:
    total = float(size_x + size_y + size_i)
    updated_sq = (
        (size_i + size_x) / total * d_xi * d_xi
        + (size_i + size_y) / total * d_yi * d_yi
        - size_i / total * d_xy * d_xy
    )
    if updated_sq < 0.0:
        return 0.0
    return math.sqrt(updated_sq)


@njit(cache=True)
def _uf_find(parent: npt.NDArray[np.int64], x: int) -> int:
    p = x
    while parent[x] != x:
        x = parent[x]
    while parent[p] != x:
        nxt = parent[p]
        parent[p] = x
        p = nxt
    return x


@njit(cache=True)
def _label_linkage_matrix_numba(linkage: npt.NDArray[np.float64], n: int) -> None:
    parent = np.arange(2 * n - 1, dtype=np.int64)
    size = np.ones(2 * n - 1, dtype=np.int64)
    next_label = n

    for i in range(n - 1):
        left = int(linkage[i, 0])
        right = int(linkage[i, 1])
        left_root = _uf_find(parent, left)
        right_root = _uf_find(parent, right)
        if left_root < right_root:
            linkage[i, 0] = float(left_root)
            linkage[i, 1] = float(right_root)
            merge_left = left_root
            merge_right = right_root
        else:
            linkage[i, 0] = float(right_root)
            linkage[i, 1] = float(left_root)
            merge_left = right_root
            merge_right = left_root

        label = next_label
        parent[merge_left] = label
        parent[merge_right] = label
        linkage[i, 3] = float(size[merge_left] + size[merge_right])
        size[label] = int(size[merge_left] + size[merge_right])
        next_label += 1


@njit(cache=True)
def _mergesort_order(distances: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Stable mergesort order for linkage merge heights (matches numpy mergesort)."""
    n = distances.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    width = 1
    indices = np.arange(n, dtype=np.int64)
    buffer = np.empty(n, dtype=np.int64)

    while width < n:
        for start in range(0, n, 2 * width):
            mid = min(start + width, n)
            end = min(start + 2 * width, n)
            left = start
            right = mid
            cursor = start
            while left < mid and right < end:
                if distances[indices[left]] <= distances[indices[right]]:
                    buffer[cursor] = indices[left]
                    left += 1
                else:
                    buffer[cursor] = indices[right]
                    right += 1
                cursor += 1
            while left < mid:
                buffer[cursor] = indices[left]
                left += 1
                cursor += 1
            while right < end:
                buffer[cursor] = indices[right]
                right += 1
                cursor += 1
        indices, buffer = buffer, indices
        width *= 2

    return indices


@njit(cache=True)
def _sort_and_label_linkage(
    raw: npt.NDArray[np.float64],
    n: int,
) -> npt.NDArray[np.float64]:
    order = _mergesort_order(raw[:, 2])
    out = np.empty_like(raw)
    for i in range(n - 1):
        row = order[i]
        out[i, 0] = raw[row, 0]
        out[i, 1] = raw[row, 1]
        out[i, 2] = raw[row, 2]
        out[i, 3] = raw[row, 3]
    _label_linkage_matrix_numba(out, n)
    return out


@njit(cache=True)
def _ward_nn_chain_core(
    dists: npt.NDArray[np.float64],
    n: int,
) -> npt.NDArray[np.float64]:
    d = dists.copy()
    size = np.ones(n, dtype=np.int64)
    out = np.empty((n - 1, 4), dtype=np.float64)
    chain = np.empty(n, dtype=np.int64)
    chain_len = 0

    for step in range(n - 1):
        if chain_len == 0:
            chain_len = 1
            for slot in range(n):
                if size[slot] > 0:
                    chain[0] = slot
                    break

        while True:
            x = chain[chain_len - 1]
            if chain_len > 1:
                y = chain[chain_len - 2]
                current_min = d[_condensed_index_sym(x, y, n)]
            else:
                y = -1
                current_min = math.inf

            for slot in range(n):
                if size[slot] == 0 or slot == x:
                    continue
                dist = d[_condensed_index_sym(x, slot, n)]
                if dist < current_min:
                    current_min = dist
                    y = slot

            if chain_len > 1 and y == chain[chain_len - 2]:
                break

            chain[chain_len] = y
            chain_len += 1

        chain_len -= 2
        x = chain[chain_len]
        y = chain[chain_len + 1]
        if x > y:
            x, y = y, x

        nx = size[x]
        ny = size[y]
        out[step, 0] = x
        out[step, 1] = y
        out[step, 2] = current_min
        out[step, 3] = nx + ny
        size[x] = 0
        size[y] = nx + ny

        for slot in range(n):
            ni = size[slot]
            if ni == 0 or slot == y:
                continue
            idx = _condensed_index_sym(slot, y, n)
            d[idx] = _ward_distance_update(
                d[_condensed_index_sym(slot, x, n)],
                d[idx],
                current_min,
                nx,
                ny,
                ni,
            )
    return out


@njit(cache=True)
def _ward_linkage_numba_full(
    dists: npt.NDArray[np.float64],
    n: int,
) -> npt.NDArray[np.float64]:
    raw = _ward_nn_chain_core(dists, n)
    return _sort_and_label_linkage(raw, n)


@njit(cache=True)
def cut_maxclust_numba(
    linkage: npt.NDArray[np.float64],
    n_samples: int,
    n_clusters: int,
) -> npt.NDArray[np.int64]:
    target = n_clusters
    if target < 1:
        target = 1
    if target > n_samples:
        target = n_samples
    if target == n_samples:
        out = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            out[i] = i + 1
        return out
    if target == 1:
        return np.ones(n_samples, dtype=np.int64)

    parent = np.arange(2 * n_samples - 1, dtype=np.int64)
    merges = n_samples - target
    for step in range(merges):
        left = int(linkage[step, 0])
        right = int(linkage[step, 1])
        merged = n_samples + step
        left_root = _uf_find(parent, left)
        right_root = _uf_find(parent, right)
        parent[left_root] = merged
        parent[right_root] = merged
        parent[merged] = merged

    roots = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        roots[i] = _uf_find(parent, i)

    unique_roots = np.unique(roots)
    root_to_label = np.empty(2 * n_samples - 1, dtype=np.int64)
    for i in range(2 * n_samples - 1):
        root_to_label[i] = -1
    for idx in range(unique_roots.shape[0]):
        root_to_label[unique_roots[idx]] = idx + 1

    labels = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        labels[i] = root_to_label[roots[i]]
    return labels
