"""Storey pi0 estimation for q-value correction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge.constants import QVALUE_LAMBDAS
from proteoforge.correction.qvalue._gcv import estimate_pi0_gcv

_TINY = np.finfo(np.float64).tiny


def pi0_lambda_curve(
    pvalues: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Build the Storey lambda curve from raw p-values.

    Parameters
    ----------
    pvalues
        Raw p-values.

    Returns
    -------
    counts
        ndarray of float64, number of p-values above each lambda on
        :data:`~proteoforge.constants.QVALUE_LAMBDAS`.
    pi0_lambda
        ndarray of float64, Storey pi0 estimate at each lambda.
    """
    m = len(pvalues)
    sorted_p = np.sort(pvalues)
    counts = (m - np.searchsorted(sorted_p, QVALUE_LAMBDAS, side="right")).astype(
        np.float64
    )
    pi0_lambda = counts / (m * (1.0 - QVALUE_LAMBDAS))
    return counts, pi0_lambda


def pi0_bootstrap(
    pi0_lambda: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    m: int,
) -> float:
    """
    Estimate pi0 from the lambda curve via the Storey bootstrap rule.

    Parameters
    ----------
    pi0_lambda
        Pi0 estimate at each lambda.
    counts
        Counts above each lambda (from :func:`pi0_lambda_curve`).
    lambdas
        Lambda grid, same length as ``pi0_lambda``.
    m
        Number of p-values used to build the curve.

    Returns
    -------
    float
        Bootstrap pi0 estimate in ``(0, 1]``.
    """
    min_pi0 = float(np.quantile(pi0_lambda, 0.1))
    mse = (counts / (m**2 * (1.0 - lambdas) ** 2)) * (1.0 - counts / m) + (
        pi0_lambda - min_pi0
    ) ** 2
    return float(min(np.min(pi0_lambda[mse == np.min(mse)]), 1.0))


def pi0_from_pvalues(pvalues: npt.NDArray[np.float64]) -> float:
    """
    Estimate the proportion of true nulls from raw p-values.

    Uses the Numba GCV smoothing spline on the Storey lambda curve. Falls back
    to :func:`pi0_bootstrap` when the spline fit is ill-posed.

    Parameters
    ----------
    pvalues
        Raw p-values.

    Returns
    -------
    float
        Pi0 estimate in ``(0, 1]``. Returns ``1.0`` when ``pvalues`` is empty.
    """
    m = len(pvalues)
    if m == 0:
        return 1.0

    counts, pi0_lambda = pi0_lambda_curve(pvalues)
    result = estimate_pi0_gcv(pi0_lambda)
    if not np.isfinite(result) or result <= 0.0:
        result = pi0_bootstrap(pi0_lambda, counts, QVALUE_LAMBDAS, m)

    return float(np.clip(result, _TINY, 1.0))
