"""
Multiple-testing correction for p-values (NumPy only).

Ported from the QuEStVar ``p_adjust`` module into the MIT package. Provides
Bonferroni, Holm, Hochberg, Benjamini-Hochberg, and Benjamini-Yekutieli
adjustments matching R's ``p.adjust``. The Storey q-value path is disabled in
Phase 2: it needs a scipy spline or a NumPy-only pi0 estimator and returns
post-v0.1.0. No scipy import lives here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

VALID_METHODS: set[str | None] = {
    None,
    "none",
    "bonferroni",
    "holm",
    "hochberg",
    "fdr",
    "fdr_bh",
    "BY",
    # "qvalue",  # deferred post-v0.1.0: needs scipy spline or numpy pi0
}

# _QVALUE_DEFAULT_LAMBDA = np.arange(0.05, 0.96, 0.05, dtype=np.float64)


def p_adjust(
    pvalues: npt.NDArray[np.float64],
    method: str | None,
    n_tests: int | None = None,
) -> npt.NDArray[np.float64]:
    """
    Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues
        Unadjusted p-values.
    method
        Correction method: ``None``, ``"none"``, ``"bonferroni"``, ``"holm"``,
        ``"hochberg"``, ``"fdr"``, ``"fdr_bh"``, or ``"BY"``.
    n_tests
        Number of tests. Defaults to ``len(pvalues)``.

    Returns
    -------
    np.ndarray
        Adjusted p-values, dtype float64.

    Raises
    ------
    ValueError
        If ``method`` is not a supported correction method.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)

    if method not in VALID_METHODS:
        valid = sorted(VALID_METHODS, key=str)
        msg = f"Unknown correction method: {method!r}. Valid: {valid}"
        raise ValueError(msg)
    if n <= 0:
        return p.copy()
    if method in (None, "none"):
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method == "holm":
        return _holm(p, n)
    if method == "hochberg":
        return _hochberg(p, n)
    if method in ("fdr", "fdr_bh"):
        return _fdr_bh(p, n)
    if method == "BY":
        return _fdr_by(p, n)
    # if method == "qvalue":
    #     return _qvalue(p, n)  # deferred: needs scipy spline or numpy pi0
    return p.copy()


def p_adjust_by_group(
    pvalues: npt.NDArray[np.float64],
    group_codes: npt.NDArray[np.intp],
    method: str | None,
) -> npt.NDArray[np.float64]:
    """
    Adjust p-values independently within each integer group code.

    Parameters
    ----------
    pvalues
        Raw p-values, same length as ``group_codes``.
    group_codes
        Group index per p-value (for example protein index).
    method
        Correction method passed to :func:`p_adjust`.

    Returns
    -------
    np.ndarray
        Adjusted values; non-finite inputs stay NaN.
    """
    out = np.full(pvalues.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(pvalues)
    if not np.any(finite):
        return out

    if method in (None, "none"):
        out[finite] = pvalues[finite]
        return out

    if method == "bonferroni":
        codes = group_codes[finite]
        p = pvalues[finite]
        _, inverse = np.unique(codes, return_inverse=True)
        counts = np.bincount(inverse)
        out[finite] = np.minimum(p * counts[inverse], 1.0)
        return out

    for code in np.unique(group_codes[finite]):
        index = np.flatnonzero((group_codes == code) & finite)
        group = pvalues[index]
        out[index] = p_adjust(group, method, n_tests=int(group.size))
    return out


def _holm(p: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, p[order] * (n - np.arange(m)))
    adjusted = np.maximum.accumulate(adjusted)
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def _hochberg(p: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)[::-1]
    steps = np.arange(n - m + 1, n + 1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _fdr_bh(p: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    m = len(p)
    order = np.argsort(p)[::-1]
    steps = n / np.arange(n, n - m, -1)
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result


def _fdr_by(p: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    m = len(p)
    harmonic = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p)[::-1]
    steps = n / np.arange(n, n - m, -1) * harmonic
    q = np.minimum(1.0, np.minimum.accumulate(steps * p[order]))
    result = np.empty_like(p)
    result[order] = q
    return result
