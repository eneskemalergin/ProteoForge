"""IHW helpers (R ``helpers.R`` / ``mydiv`` parity)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def safe_divide(
    pvalues: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute weighted p-values ``p / w`` with R ``mydiv`` guards.

    Parameters
    ----------
    pvalues
        Raw p-values.
    weights
        Per-hypothesis weights.

    Returns
    -------
    ndarray of float64
        Weighted p-values clipped to ``[0, 1]``.
    """
    out = np.empty_like(pvalues)
    out[pvalues == 0.0] = 0.0
    zero_w = (pvalues != 0.0) & (weights == 0.0)
    out[zero_w] = 1.0
    valid = (pvalues != 0.0) & (weights != 0.0)
    out[valid] = np.minimum(pvalues[valid] / weights[valid], 1.0)
    return out
