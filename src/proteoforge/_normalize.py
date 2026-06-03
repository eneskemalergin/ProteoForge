"""Control-relative normalization for peptide intensities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.schema import INTENSITY, PEPTIDE_ID, PROTEIN_ID, SAMPLE_ID

NORMALIZED_INTENSITY = "intensity_normalized"


def normalize_control_relative(
    intensity: npt.NDArray[np.float64],
    *,
    control_column_indices: npt.NDArray[np.intp],
    input_is_log2: bool,
) -> npt.NDArray[np.float64]:
    """
    Normalize peptide intensities against a control condition (wide matrix).

    Applies log2 (if needed), per-sample z-scoring across peptides, then
    subtracts each peptide's mean control intensity. Matches reference
    ``normalize.against_condition`` numerically (pandas ``std`` with ddof=1).

    Parameters
    ----------
    intensity
        Wide intensity matrix, shape ``(n_peptides, n_samples)``.
    control_column_indices
        Column indices for control samples within ``intensity``.
    input_is_log2
        When ``False``, apply ``log2`` before z-scoring.

    Returns
    -------
    np.ndarray
        Normalized matrix with the same shape as ``intensity``.

    Raises
    ------
    ProteoForgeValidationError
        If control indices are empty, intensities are non-positive when log2
        is required, or a sample column has zero standard deviation.
    """
    if intensity.size == 0:
        return intensity.copy()

    if control_column_indices.size == 0:
        msg = "At least one control sample column is required for normalization."
        raise ProteoForgeValidationError(msg)

    data = intensity.astype(np.float64, copy=True)

    if not input_is_log2:
        if np.any(data <= 0):
            msg = (
                "Non-positive intensity values found. "
                "Apply log2 upstream or set input_is_log2=True for log-scale data."
            )
            raise ProteoForgeValidationError(msg)
        data = np.log2(data)

    col_mean = data.mean(axis=0)
    col_std = data.std(axis=0, ddof=1)
    zero_std_cols = np.flatnonzero(col_std == 0)
    if zero_std_cols.size > 0:
        msg = (
            f"Sample column(s) {zero_std_cols.tolist()} have zero standard deviation "
            "after log transform. Check for constant or missing intensities."
        )
        raise ProteoForgeValidationError(msg)

    data = (data - col_mean) / col_std

    control_values = data[:, control_column_indices]
    control_row_mean = control_values.mean(axis=1, keepdims=True)
    normalized: npt.NDArray[np.float64] = data - control_row_mean
    return normalized


def normalize_control_relative_long(
    frame: pl.DataFrame,
    *,
    control_sample_ids: tuple[str, ...],
    input_is_log2: bool,
    intensity_col: str = INTENSITY,
) -> pl.DataFrame:
    """
    Normalize a long-format peptide table against a control condition.

    Same math as :func:`normalize_control_relative`, expressed with Polars
    window expressions (no pivot). Row order matches the input frame; each
    observation is keyed by ``(protein_id, peptide_id, sample_id)``, not by
    position. Downstream code must join or group on those columns rather than
    assume a sorted layout.

    Parameters
    ----------
    frame
        Long table with ``protein_id``, ``peptide_id``, ``sample_id``, and
        ``intensity_col``.
    control_sample_ids
        Sample IDs belonging to the control condition.
    input_is_log2
        Skip ``log2`` when intensities are already log-scaled.
    intensity_col
        Source intensity column name.

    Returns
    -------
    polars.DataFrame
        Input columns plus ``intensity_normalized``.

    Raises
    ------
    ProteoForgeValidationError
        If control samples are empty, required columns are missing,
        intensities are non-positive when log2 is required, or a sample
        column has zero standard deviation after transform.
    """
    if frame.is_empty():
        return frame.with_columns(pl.lit([]).alias(NORMALIZED_INTENSITY))

    _long_normalize_validate(
        frame,
        control_sample_ids=control_sample_ids,
        input_is_log2=input_is_log2,
        intensity_col=intensity_col,
    )

    work = frame.with_columns(
        _long_normalize_log_value(intensity_col, input_is_log2=input_is_log2).alias(
            "_value"
        )
    )

    sample_stats = work.group_by(SAMPLE_ID).agg(pl.col("_value").std().alias("_std"))
    _long_normalize_check_zero_std(sample_stats.rename({"_std": "_sig"}))

    control_set = list(control_sample_ids)
    work = work.with_columns(
        (
            (pl.col("_value") - pl.col("_value").mean().over(SAMPLE_ID))
            / pl.col("_value").std().over(SAMPLE_ID)
        ).alias("_z")
    )
    control_mean = (
        work.filter(pl.col(SAMPLE_ID).is_in(control_set))
        .group_by([PROTEIN_ID, PEPTIDE_ID])
        .agg(pl.col("_z").mean().alias("_ctrl_mean"))
    )
    work = work.join(control_mean, on=[PROTEIN_ID, PEPTIDE_ID], how="left")
    work = work.with_columns(
        (pl.col("_z") - pl.col("_ctrl_mean")).alias(NORMALIZED_INTENSITY)
    )
    return work.select([*frame.columns, NORMALIZED_INTENSITY])


def _long_normalize_validate(
    frame: pl.DataFrame,
    *,
    control_sample_ids: tuple[str, ...],
    input_is_log2: bool,
    intensity_col: str,
) -> None:
    if frame.is_empty():
        return

    if not control_sample_ids:
        msg = "At least one control sample is required for normalization."
        raise ProteoForgeValidationError(msg)

    required = {PROTEIN_ID, PEPTIDE_ID, SAMPLE_ID, intensity_col}
    missing = required - set(frame.columns)
    if missing:
        msg = f"Long normalize missing columns: {sorted(missing)}."
        raise ProteoForgeValidationError(msg)

    if not input_is_log2:
        non_positive = frame.filter(pl.col(intensity_col) <= 0).height
        if non_positive > 0:
            msg = (
                "Non-positive intensity values found. "
                "Apply log2 upstream or set input_is_log2=True for log-scale data."
            )
            raise ProteoForgeValidationError(msg)


def _long_normalize_log_value(
    intensity_col: str,
    *,
    input_is_log2: bool,
) -> pl.Expr:
    value = pl.col(intensity_col).cast(pl.Float64)
    if not input_is_log2:
        value = value.log(base=2)
    return value


def _long_normalize_check_zero_std(sample_stats: pl.DataFrame) -> None:
    zero_std_samples = (
        sample_stats.filter(pl.col("_sig") == 0).get_column(SAMPLE_ID).to_list()
    )
    if zero_std_samples:
        preview = zero_std_samples[:8]
        msg = (
            f"Sample(s) {preview} have zero standard deviation after log transform. "
            "Check for constant or missing intensities."
        )
        raise ProteoForgeValidationError(msg)
