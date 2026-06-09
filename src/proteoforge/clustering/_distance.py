"""Distance helpers for peptide condition profiles."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from numba import njit


def euclidean_condensed(
    profiles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute condensed Euclidean distances.

    Parameters
    ----------
    profiles
        Profile matrix, shape ``(n_items, n_features)``.

    Returns
    -------
    ndarray of float64
        Condensed distance vector, length ``n_items * (n_items - 1) / 2``.
    """
    matrix = np.asarray(profiles, dtype=np.float64)
    if matrix.ndim != 2:
        msg = "profiles must be a 2-D array."
        raise ValueError(msg)
    n_items = matrix.shape[0]
    if n_items <= 1:
        return np.empty(0, dtype=np.float64)
    return np.asarray(_euclidean_condensed_numba_ij(matrix), dtype=np.float64)


def condensed_to_square(
    condensed: npt.NDArray[np.float64],
    *,
    n_samples: int,
) -> npt.NDArray[np.float64]:
    """
    Expand condensed distances to a square matrix.

    Parameters
    ----------
    condensed
        Condensed distance vector.
    n_samples
        Number of samples represented by the vector.

    Returns
    -------
    ndarray of float64
        Square symmetric matrix, shape ``(n_samples, n_samples)``.
    """
    expected = n_samples * (n_samples - 1) // 2
    if condensed.size != expected:
        msg = (
            f"condensed distance length {condensed.size} does not match "
            f"n_samples={n_samples} (expected {expected})."
        )
        raise ValueError(msg)
    square = np.zeros((n_samples, n_samples), dtype=np.float64)
    cursor = 0
    for i in range(n_samples - 1):
        span = n_samples - i - 1
        values = condensed[cursor : cursor + span]
        square[i, i + 1 :] = values
        square[i + 1 :, i] = values
        cursor += span
    return square


@njit(cache=True)
def _euclidean_condensed_numba_ij(
    profiles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    matrix = np.ascontiguousarray(profiles)
    n_items, n_features = matrix.shape
    if n_items <= 1:
        return np.empty(0, dtype=np.float64)
    out_len = n_items * (n_items - 1) // 2
    out = np.empty(out_len, dtype=np.float64)
    cursor = 0
    for i in range(n_items - 1):
        for j in range(i + 1, n_items):
            acc = 0.0
            for k in range(n_features):
                diff = matrix[i, k] - matrix[j, k]
                acc += diff * diff
            out[cursor] = math.sqrt(acc)
            cursor += 1
    return out
