"""Classical unweighted p-value adjustment kernels (R ``p.adjust`` parity)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


def holm(pvalues: npt.NDArray[np.float64], n_tests: int) -> npt.NDArray[np.float64]:
    """
    Apply Holm step-down adjustment (R ``p.adjust(..., "holm")``).

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.
    """
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, pvalues[order] * (n_tests - np.arange(m)))
    adjusted = np.maximum.accumulate(adjusted)
    result = np.empty_like(pvalues)
    result[order] = adjusted
    return result


@njit(cache=True)
def _hommel_sorted_kernel(
    sorted_p: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hommel closed-test kernel on ascending sorted p-values."""
    n = sorted_p.shape[0]
    adjusted = sorted_p.copy()
    for k in range(n, 1, -1):
        cim = np.inf
        start = n - k
        for j in range(k):
            val = (k * sorted_p[start + j]) / (j + 1)
            if val < cim:
                cim = val
        for j in range(n - k, n):
            if cim > adjusted[j]:
                adjusted[j] = cim
        scale = float(k)
        for j in range(n - k):
            bound = scale * sorted_p[j]
            if bound > cim:
                bound = cim
            if bound > adjusted[j]:
                adjusted[j] = bound
    return adjusted


def hommel(pvalues: npt.NDArray[np.float64], n_tests: int) -> npt.NDArray[np.float64]:
    """
    Apply Hommel closed-test adjustment (R ``p.adjust(..., "hommel")``).

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family. When
        ``n_tests > len(pvalues)``, missing tests are treated as p = 1.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.
    """
    m = len(pvalues)
    if m == 0:
        return pvalues.copy()
    work = (
        np.concatenate([pvalues, np.ones(n_tests - m, dtype=np.float64)])
        if n_tests > m
        else pvalues
    )
    order = np.argsort(work)
    sorted_p = work[order]
    if sorted_p.size <= 1:
        adjusted = sorted_p.copy()
    else:
        adjusted = _hommel_sorted_kernel(sorted_p)
    out_work = np.empty_like(adjusted)
    out_work[order] = adjusted
    return np.minimum(out_work[:m], 1.0)


def hochberg(pvalues: npt.NDArray[np.float64], n_tests: int) -> npt.NDArray[np.float64]:
    """
    Apply Hochberg step-up adjustment (R ``p.adjust(..., "hochberg")``).

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.
    """
    m = len(pvalues)
    order = np.argsort(pvalues)[::-1]
    steps = np.arange(n_tests - m + 1, n_tests + 1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * pvalues[order]))
    result = np.empty_like(pvalues)
    result[order] = q
    return result


def fdr_bh(pvalues: npt.NDArray[np.float64], n_tests: int) -> npt.NDArray[np.float64]:
    """
    Apply Benjamini-Hochberg FDR adjustment (R ``p.adjust(..., "BH")``).

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.
    """
    m = len(pvalues)
    order = np.argsort(pvalues)[::-1]
    steps = n_tests / np.arange(n_tests, n_tests - m, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * pvalues[order]))
    result = np.empty_like(pvalues)
    result[order] = q
    return result


def fdr_by(pvalues: npt.NDArray[np.float64], n_tests: int) -> npt.NDArray[np.float64]:
    """
    Apply Benjamini-Yekutieli FDR adjustment (R ``p.adjust(..., "BY")``).

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.
    """
    m = len(pvalues)
    harmonic = np.sum(1.0 / np.arange(1, n_tests + 1))
    order = np.argsort(pvalues)[::-1]
    steps = n_tests / np.arange(n_tests, n_tests - m, -1) * harmonic
    q = np.minimum(1.0, np.minimum.accumulate(steps * pvalues[order]))
    result = np.empty_like(pvalues)
    result[order] = q
    return result
