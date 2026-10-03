"""Huber proposal-2 scale kernel for the RLM backend."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def huber_scale_rows(
    residual: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    h: npt.NDArray[np.float64],
    nobs: float,
    d: float,
    tol: float,
    max_iter: int,
) -> npt.NDArray[np.float64]:
    """
    Iterate the Huber proposal-2 scale for each row of residuals.

    Parameters
    ----------
    residual
        C-contiguous residuals, shape ``(m, n)``.
    start
        Starting scale per row (the MAD), shape ``(m,)``.
    h
        Consistency term per row, shape ``(m,)``.
    nobs
        Number of observations per row.
    d
        Huber threshold for the scale estimate.
    tol
        Absolute change below which a row stops iterating.
    max_iter
        Iteration cap; each row runs at most ``max_iter - 1`` updates.

    Returns
    -------
    numpy.ndarray
        Scale per row, shape ``(m,)``. Rows with ``h <= 0`` or ``start <= 0``
        get 0.

    Notes
    -----
    Same fixed point as statsmodels ``HuberScale``. Each row runs its own loop,
    so no temporary arrays are created; summation order differs from
    ``numpy.sum``, which changes results only in the last bits.
    """
    m, n = residual.shape
    out = np.zeros(m, dtype=np.float64)
    half_d2 = d * d / 2.0
    for i in range(m):
        c = start[i]
        if not (h[i] > 0.0 and c > 0.0):
            continue
        for _ in range(1, max_iter):
            acc = 0.0
            for j in range(n):
                z = residual[i, j] / c
                if abs(z) < d:
                    acc += z * z / 2.0
                else:
                    acc += half_d2
            nscale = math.sqrt(acc / (nobs * h[i]) * c * c)
            converged = abs(nscale - c) <= tol
            c = nscale
            if converged:
                break
        out[i] = c
    return out
