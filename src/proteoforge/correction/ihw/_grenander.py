"""Grenander ECDF estimator for IHW (``fdrtool::gcmlcm`` parity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit


@dataclass(frozen=True)
class GrenanderFit:
    """Least concave majorant knot representation."""

    x_knots: npt.NDArray[np.float64]
    y_knots: npt.NDArray[np.float64]
    slope_knots: npt.NDArray[np.float64]
    length: int


@njit(cache=True)
def _iso_mean(
    y: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Weighted isotonic regression (non-decreasing), PAVA."""
    n = y.shape[0]
    if n == 1:
        return y.copy()
    cap = n
    values = np.empty(cap, dtype=np.float64)
    weights = np.empty(cap, dtype=np.float64)
    counts = np.empty(cap, dtype=np.int64)
    size = n
    for i in range(n):
        values[i] = y[i]
        weights[i] = w[i]
        counts[i] = 1
    i = 0
    while i < size - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        tw = weights[i] + weights[i + 1]
        avg = (weights[i] * values[i] + weights[i + 1] * values[i + 1]) / tw
        values[i] = avg
        weights[i] = tw
        counts[i] += counts[i + 1]
        for j in range(i + 1, size - 1):
            values[j] = values[j + 1]
            weights[j] = weights[j + 1]
            counts[j] = counts[j + 1]
        size -= 1
        if i > 0:
            i -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for k in range(size):
        val = values[k]
        cnt = counts[k]
        for _ in range(cnt):
            out[pos] = val
            pos += 1
    return out


def presorted_grenander(
    sorted_pvalues: npt.NDArray[np.float64],
    m_total: int,
    *,
    grenander_binsize: int = 1,
) -> GrenanderFit:
    """
    Least concave majorant of the ECDF within a bin.

    Parameters
    ----------
    sorted_pvalues
        Ascending p-values in one covariate stratum.
    m_total
        Total hypothesis count for Grenander scaling (``m_groups_grenander``).
    grenander_binsize
        Thinning stride for unique p-values (R default 1).

    Returns
    -------
    GrenanderFit
        Knot slopes and coordinates for LP constraints.
    """
    if sorted_pvalues.shape[0] == 0:
        return GrenanderFit(
            x_knots=np.array([0.0]),
            y_knots=np.array([0.0]),
            slope_knots=np.array([1.0]),
            length=1,
        )

    unique_p, counts = np.unique(sorted_pvalues, return_counts=True)
    ecdf = np.cumsum(counts, dtype=np.float64) / float(m_total)

    if unique_p[0] > 0.0:
        unique_p = np.concatenate(([0.0], unique_p))
        ecdf = np.concatenate(([0.0], ecdf))

    if unique_p[-1] < 1.0:
        unique_p = np.concatenate((unique_p, [1.0]))
        ecdf = np.concatenate((ecdf, [1.0]))

    if grenander_binsize != 1:
        nmax = unique_p.shape[0]
        idx = np.concatenate(
            [
                np.arange(0, nmax - 1, grenander_binsize),
                np.array([nmax - 1]),
            ]
        )
        unique_p = unique_p[idx]
        ecdf = ecdf[idx]

    dx = np.diff(unique_p)
    dy = np.diff(ecdf)
    rawslope = dy / dx
    rawslope = np.where(rawslope == np.inf, np.finfo(np.float64).max, rawslope)
    rawslope = np.where(rawslope == -np.inf, -np.finfo(np.float64).max, rawslope)

    slope = -_iso_mean(-rawslope, dx)
    dup = np.concatenate(([True], slope[1:] != slope[:-1]))
    x_knots = unique_p[np.concatenate([dup, [True]])]
    dx_knots = np.diff(x_knots)
    slope_knots = slope[dup]
    y_knots = ecdf[0] + np.concatenate(([0.0], np.cumsum(dx_knots * slope_knots)))

    n_seg = int(slope_knots.shape[0])
    x_out = np.delete(x_knots, n_seg - 1)
    y_out = np.delete(y_knots, n_seg - 1)
    return GrenanderFit(
        x_knots=x_out,
        y_knots=y_out,
        slope_knots=slope_knots,
        length=n_seg,
    )
