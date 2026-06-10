"""Hommel closed-test kernel (R ``p.adjust`` parity, O(n))."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def _find_hull(m: int, p: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], int]:
    """
    Build the lower hull for Hommel alpha jump points.

    Parameters
    ----------
    m
        Number of sorted p-values.
    p
        Ascending sorted p-values (length ``m``).

    Returns
    -------
    hull
        Vertex indices using the 1-based convention of the reference
        implementation (index 0 is an unused sentinel).
    hlen
        Active length of ``hull``.
    """
    hull = np.zeros(m + 2, dtype=np.int64)
    hull[0] = 0
    hull[1] = 1
    hlen = 2
    for i in range(2, m + 1):
        if i == m or (m - 1) * (p[i - 1] - p[0]) < (i - 1) * (p[m - 1] - p[0]):
            while True:
                r = hlen - 1
                if r > 1:
                    not_convex = (i - hull[r - 1]) * (
                        p[hull[r] - 1] - p[hull[r - 1] - 1]
                    ) >= (hull[r] - hull[r - 1]) * (p[i - 1] - p[hull[r - 1] - 1])
                elif r == 1:
                    not_convex = i * p[hull[1] - 1] >= hull[1] * p[i - 1]
                else:
                    not_convex = False
                if not_convex:
                    hlen -= 1
                else:
                    break
            hull[hlen] = i
            hlen += 1
    return hull, hlen


@njit(cache=True)
def _hommel_sorted(sorted_p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Apply Hommel adjustment on ascending sorted p-values.

    Simes local test via an O(n) hull sweep. Sorting, ``n_tests`` padding,
    and unscrambling are handled by :func:`~proteoforge.correction._methods.hommel`.

    Parameters
    ----------
    sorted_p
        Raw p-values in ascending order.

    Returns
    -------
    ndarray of float64
        Adjusted p-values in sorted order. Not clipped to ``[0, 1]``;
        the public wrapper applies clipping after restoring input order.
    """
    m = sorted_p.shape[0]
    if m <= 1:
        return sorted_p.copy()

    alpha = np.zeros(m + 1, dtype=np.float64)
    hull, hlen = _find_hull(m, sorted_p)
    k = hlen - 1
    i = 1
    while i <= m:
        if k > 1:
            hi = hull[k] - 1
            lo = hull[k - 1] - 1
            dk = sorted_p[lo] * (hull[k] - m + i) - sorted_p[hi] * (hull[k - 1] - m + i)
        else:
            dk = 0.0
        if k > 1 and dk < 0.0:
            k -= 1
        else:
            alpha[i - 1] = float(i) * sorted_p[hull[k] - 1] / (hull[k] - m + i)
            i += 1

    adjusted = np.empty(m, dtype=np.float64)
    i = 1
    j = m + 1
    while i <= m:
        if float(j - 1) * sorted_p[i - 1] <= alpha[j - 1]:
            if j > m:
                adjusted[i - 1] = alpha[m]
            else:
                bound = float(j) * sorted_p[i - 1]
                adj_val = alpha[j - 1]
                adjusted[i - 1] = bound if bound < adj_val else adj_val
            i += 1
        else:
            j -= 1
    return adjusted
