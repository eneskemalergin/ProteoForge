"""Imputation weights for WLS (inactive under RLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from proteoforge.schema import IS_COMPLETE_MISSING, IS_REAL, WEIGHT

if TYPE_CHECKING:
    import polars as pl

    from proteoforge._config import Config

MEASURED_WEIGHT: float = 1.0
SPARSE_IMPUTED_WEIGHT: float = 1e-5

_WLS_MODELS: frozenset[str] = frozenset({"wls", "ebayes"})


def imputation_weights(
    is_real: npt.NDArray[np.bool_],
    is_complete_missing: npt.NDArray[np.bool_],
    *,
    biological_weight: float,
    sparse_weight: float = SPARSE_IMPUTED_WEIGHT,
    measured_weight: float = MEASURED_WEIGHT,
) -> npt.NDArray[np.float64]:
    """
    Assign weights from imputation provenance masks.

    Parameters
    ----------
    is_real
        True where the value is measured (not imputed).
    is_complete_missing
        True where the peptide is condition-wide imputed (dense imputation).
    biological_weight
        Weight for condition-wide imputed entries.
    sparse_weight
        Weight for sparsely imputed entries.
    measured_weight
        Weight for measured entries.

    Returns
    -------
    np.ndarray
        Per-entry weights, dtype float64.

    Raises
    ------
    ValueError
        If the two masks differ in shape, or weight ordering is invalid.
    """
    real = np.asarray(is_real, dtype=bool)
    dense = np.asarray(is_complete_missing, dtype=bool)
    if real.shape != dense.shape:
        msg = (
            f"Mask shape mismatch: is_real {real.shape} vs "
            f"is_complete_missing {dense.shape}."
        )
        raise ValueError(msg)
    if measured_weight <= biological_weight or measured_weight <= sparse_weight:
        msg = "measured_weight must exceed both biological and sparse weights."
        raise ValueError(msg)

    weights = np.where(
        real,
        measured_weight,
        np.where(dense, biological_weight, sparse_weight),
    )
    return weights.astype(np.float64, copy=False)


def row_weights(
    frame: pl.DataFrame,
    config: Config,
) -> npt.NDArray[np.float64] | None:
    """
    Resolve per-row WLS weights aligned to ``frame`` order.

    Parameters
    ----------
    frame
        Long peptide frame, already in the row order used for fitting.
    config
        Pipeline configuration. RLM returns ``None``.

    Returns
    -------
    np.ndarray or None
        Per-row weights for WLS and eBayes, otherwise ``None``.
    """
    if config.model not in _WLS_MODELS:
        return None
    if WEIGHT in frame.columns:
        return frame.get_column(WEIGHT).to_numpy().astype(np.float64, copy=False)
    if IS_REAL in frame.columns and IS_COMPLETE_MISSING in frame.columns:
        return imputation_weights(
            frame.get_column(IS_REAL).to_numpy(),
            frame.get_column(IS_COMPLETE_MISSING).to_numpy(),
            biological_weight=config.wls_biological_weight,
        )
    return np.ones(frame.height, dtype=np.float64)
