"""Storey q-value step-up adjustment."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge.correction.qvalue._pi0 import pi0_from_pvalues


def adjust_qvalues(
    pvalues: npt.NDArray[np.float64],
    n_tests: int,
) -> npt.NDArray[np.float64]:
    """
    Compute Storey q-values with GCV pi0 estimation.

    Parameters
    ----------
    pvalues
        Raw p-values.
    n_tests
        Total number of tests in the correction family (may exceed
        ``len(pvalues)``).

    Returns
    -------
    ndarray of float64
        Q-values aligned with ``pvalues``, clipped to ``[0, 1]``. Empty input
        returns a copy unchanged.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return p.copy()

    pi0 = pi0_from_pvalues(p)
    order = np.argsort(p)[::-1]
    ranks = np.arange(m, 0, -1, dtype=np.float64)
    base = p[order] * float(n_tests) / ranks
    qvals = pi0 * np.minimum(1.0, np.minimum.accumulate(base))
    out = np.empty_like(p)
    out[order] = qvals
    return out
