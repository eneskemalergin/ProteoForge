"""p.adjust-style dispatch for unweighted and method-specific correctors."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.correction.qvalue import adjust_qvalues

from . import _methods

VALID_METHODS: set[str | None] = {
    None,
    "none",
    "bonferroni",
    "holm",
    "hommel",
    "hochberg",
    "fdr",
    "fdr_bh",
    "BY",
    "qvalue",
}

_UnweightedFn = Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]

_UNWEIGHTED: dict[str, _UnweightedFn] = {
    "holm": _methods.holm,
    "hommel": _methods.hommel,
    "hochberg": _methods.hochberg,
    "fdr": _methods.fdr_bh,
    "fdr_bh": _methods.fdr_bh,
    "BY": _methods.fdr_by,
}


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
        Raw p-values.
    method
        Correction method. One of ``None``, ``"none"``, ``"bonferroni"``,
        ``"holm"``, ``"hommel"``, ``"hochberg"``, ``"fdr"``, ``"fdr_bh"``,
        ``"BY"``, or ``"qvalue"``.
    n_tests
        Total number of tests in the correction family. Defaults to
        ``len(pvalues)``.

    Returns
    -------
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``, clipped to ``[0, 1]``.

    Raises
    ------
    ProteoForgeValidationError
        If ``method`` is not a supported correction method.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = n_tests if n_tests is not None else len(p)

    if method not in VALID_METHODS:
        valid = sorted(VALID_METHODS, key=str)
        msg = f"Unknown correction method: {method!r}. Valid: {valid}"
        raise ProteoForgeValidationError(msg)
    if n <= 0:
        return p.copy()
    if method in (None, "none"):
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * n, 1.0)
    if method == "qvalue":
        return adjust_qvalues(p, n)
    if method not in _UNWEIGHTED:
        valid = sorted(VALID_METHODS, key=str)
        msg = f"Unknown correction method: {method!r}. Valid: {valid}"
        raise ProteoForgeValidationError(msg)
    return _UNWEIGHTED[method](p, n)


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
    ndarray of float64
        Adjusted p-values aligned with ``pvalues``; non-finite inputs stay NaN.

    Raises
    ------
    ProteoForgeValidationError
        If ``method`` is not a supported correction method.
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
