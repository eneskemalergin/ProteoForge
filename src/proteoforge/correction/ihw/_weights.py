"""Weight transforms for IHW (R ``weights.R`` parity)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def thresholds_to_weights(
    thresholds: npt.NDArray[np.float64],
    m_groups: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """
    Convert per-bin rejection thresholds to mean-one weights.

    Parameters
    ----------
    thresholds
        Rejection threshold per covariate bin.
    m_groups
        Hypothesis count per bin.

    Returns
    -------
    ndarray of float64
        Bin weights with weighted mean 1 over ``m_groups``.

    Raises
    ------
    ValueError
        If ``thresholds`` and ``m_groups`` differ in length.
    """
    if thresholds.shape[0] != m_groups.shape[0]:
        msg = (
            f"Length mismatch: {thresholds.shape[0]} thresholds vs "
            f"{m_groups.shape[0]} groups"
        )
        raise ValueError(msg)
    if np.all(thresholds == 0.0):
        return np.ones(thresholds.shape[0], dtype=np.float64)
    m = float(np.sum(m_groups))
    denom = float(np.sum(m_groups.astype(np.float64) * thresholds))
    return thresholds * m / denom
