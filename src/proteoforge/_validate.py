"""Validation and PreparedDataset assembly for the prepare pipeline."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._normalize import normalize_control_relative_long
from proteoforge.io._design import attach_conditions
from proteoforge.io._ingest import (
    materialize_peptide_table,
    peptide_table_ready,
    sample_ids_in_frame,
)
from proteoforge.io._provenance import attach_provenance
from proteoforge.schema import (
    CANONICAL_PEPTIDE_COLUMNS_WITH_CONDITION,
    CONDITION,
    INTENSITY,
    IS_COMPLETE_MISSING,
    IS_REAL,
    PEPTIDE_ID,
    PROTEIN_ID,
    SAMPLE_ID,
    WEIGHT,
)
from proteoforge.types import DesignTable, PreparedDataset

if TYPE_CHECKING:
    from proteoforge._config import Config

PRIMARY_KEY = (PROTEIN_ID, PEPTIDE_ID, SAMPLE_ID)


def validate_and_prepare(
    peptides: pl.DataFrame | pl.LazyFrame,
    config: Config,
    *,
    provenance: pl.DataFrame | None = None,
) -> PreparedDataset:
    """
    Validate inputs, normalize intensities, and build a PreparedDataset.

    The experimental design and sample scope come from ``config.conditions``.
    Peptide rows are filtered to configured samples; extra samples present in
    the table are dropped with a warning.

    Parameters
    ----------
    peptides
        Long-format peptide intensity table (canonical or mappable columns).
    config
        Pipeline configuration including ``control_condition`` and
        ``conditions``.
    provenance
        Optional provenance masks or weights aligned to peptide keys.

    Returns
    -------
    PreparedDataset
        Validated long-format table with normalized intensities.

    Raises
    ------
    ProteoForgeValidationError
        If any data contract check fails.
    """
    lazy_input = isinstance(peptides, pl.LazyFrame)
    samples_dropped_at_ingest: tuple[str, ...] = ()
    if lazy_input:
        frame, samples_dropped_at_ingest = materialize_peptide_table(peptides, config)
        scope_prefiltered = True
    elif isinstance(peptides, pl.DataFrame) and peptide_table_ready(peptides, config):
        frame = peptides
        scope_prefiltered = True
    else:
        frame, samples_dropped_at_ingest = materialize_peptide_table(peptides, config)
        scope_prefiltered = False

    design = config.to_design_table()
    data_samples = sample_ids_in_frame(frame)
    frame, scope_metadata = _apply_config_scope(
        frame,
        config,
        design,
        data_samples=data_samples,
        scope_prefiltered=scope_prefiltered,
        samples_dropped_at_ingest=samples_dropped_at_ingest,
    )

    if CONDITION in frame.columns:
        frame = frame.drop(CONDITION)
    frame = attach_conditions(frame, design)

    if provenance is not None:
        frame = attach_provenance(frame, provenance)

    frame = _select_peptide_columns(frame)
    _validate_structure(frame, design, config, peptide_samples=data_samples)
    _validate_peptide_coverage(frame, config.min_peptides)

    sample_ids = _ordered_samples(design, config.condition_levels)
    control_samples = design.condition_to_samples[config.control_condition]
    frame = normalize_control_relative_long(
        frame,
        control_sample_ids=control_samples,
        input_is_log2=config.input_is_log2,
    )

    protein_ids = frame.get_column(PROTEIN_ID).to_numpy().astype(np.str_)
    _, protein_index = np.unique(protein_ids, return_inverse=True)
    unique_pairs = frame.select([PROTEIN_ID, PEPTIDE_ID]).n_unique()

    metadata: dict[str, object] = {
        "n_proteins": int(np.unique(protein_ids).size),
        "n_peptides": int(unique_pairs),
        "n_samples": len(sample_ids),
        "nan_fraction": float(frame.select(pl.col(INTENSITY).is_nan().mean()).item()),
        "control_condition": config.control_condition,
        "conditions_used": config.condition_levels,
        **scope_metadata,
    }

    return PreparedDataset(
        config=config,
        peptides=frame,
        sample_ids=sample_ids,
        condition_levels=config.condition_levels,
        protein_index=protein_index.astype(np.intp),
        metadata=metadata,
    )


def _apply_config_scope(
    frame: pl.DataFrame,
    config: Config,
    design: DesignTable,
    *,
    data_samples: frozenset[str],
    scope_prefiltered: bool = False,
    samples_dropped_at_ingest: tuple[str, ...] = (),
) -> tuple[pl.DataFrame, dict[str, object]]:
    """
    Restrict the peptide table to samples listed in ``config.conditions``.

    Returns the filtered frame and metadata describing which samples were used
    or dropped from the input table.
    """
    selected = config.selected_sample_ids
    data_sample_set = set(data_samples)

    if samples_dropped_at_ingest:
        dropped_samples = list(samples_dropped_at_ingest)
    else:
        dropped_samples = sorted(data_sample_set - selected)
    missing_samples = sorted(selected - data_sample_set)

    if dropped_samples:
        preview = dropped_samples[:8]
        suffix = "..." if len(dropped_samples) > 8 else ""
        warnings.warn(
            f"Dropping {len(dropped_samples)} sample(s) not listed in "
            f"config.conditions: {preview}{suffix}. "
            f"Using conditions {list(config.condition_levels)} "
            f"({len(selected)} samples).",
            stacklevel=2,
        )

    if missing_samples:
        msg = (
            f"Config lists {len(missing_samples)} sample(s) missing from the "
            f"peptide table: {missing_samples[:10]}. "
            "Ensure configured samples appear in the input data."
        )
        raise ProteoForgeValidationError(msg)

    extra_samples = data_sample_set - selected
    if extra_samples and not scope_prefiltered:
        filtered = frame.filter(pl.col(SAMPLE_ID).is_in(list(selected)))
    else:
        filtered = frame
    if filtered.is_empty():
        msg = "No peptide rows remain after applying config.conditions sample filter."
        raise ProteoForgeValidationError(msg)

    metadata: dict[str, object] = {
        "samples_used": design.sample_ids,
        "samples_dropped": tuple(dropped_samples),
        "conditions_used": config.condition_levels,
    }
    return filtered, metadata


def _validate_structure(
    frame: pl.DataFrame,
    design: DesignTable,
    config: Config,
    *,
    peptide_samples: frozenset[str] | None = None,
) -> None:
    required = CANONICAL_PEPTIDE_COLUMNS_WITH_CONDITION
    missing = required - set(frame.columns)
    if missing:
        msg = f"Peptide table missing required columns: {sorted(missing)}."
        raise ProteoForgeValidationError(msg)

    present_samples = (
        peptide_samples if peptide_samples is not None else sample_ids_in_frame(frame)
    )
    design_samples = set(design.sample_ids)
    missing_samples = sorted(design_samples - present_samples)
    if missing_samples:
        msg = (
            f"Design samples missing from peptide table: {missing_samples[:10]}. "
            "Ensure all design samples appear in the intensity data."
        )
        raise ProteoForgeValidationError(msg)

    if config.control_condition not in design.condition_to_samples:
        valid = sorted(design.condition_to_samples)
        msg = (
            f"control_condition '{config.control_condition}' not found in design. "
            f"Valid conditions: {valid}."
        )
        raise ProteoForgeValidationError(msg)

    conditions = sorted(design.condition_to_samples)
    if len(conditions) < 2:
        msg = "At least two experimental conditions are required."
        raise ProteoForgeValidationError(msg)

    for condition, samples in design.condition_to_samples.items():
        if len(samples) < 2:
            msg = (
                f"Condition '{condition}' has fewer than 2 samples "
                f"({len(samples)}). Add replicates for interaction modeling."
            )
            raise ProteoForgeValidationError(msg)

    n_rows = frame.height
    n_unique_keys = frame.select(list(PRIMARY_KEY)).n_unique()
    if n_rows != n_unique_keys:
        duplicates = frame.group_by(list(PRIMARY_KEY)).len().filter(pl.col("len") > 1)
        examples = duplicates.head(3).select(list(PRIMARY_KEY)).rows()
        msg = (
            f"Duplicate peptide keys found ({duplicates.height} groups). "
            f"Examples: {examples}. Primary key must be unique."
        )
        raise ProteoForgeValidationError(msg)

    if not frame.get_column(INTENSITY).dtype.is_numeric():
        msg = f"Column '{INTENSITY}' must be numeric."
        raise ProteoForgeValidationError(msg)

    non_finite = frame.filter(
        pl.col(INTENSITY).is_nan() | pl.col(INTENSITY).is_infinite()
    ).height
    if non_finite > 0:
        msg = (
            f"Column '{INTENSITY}' contains {non_finite} non-finite value(s). "
            "Impute missing values before running ProteoForge."
        )
        raise ProteoForgeValidationError(msg)

    nan_count = frame.get_column(INTENSITY).null_count()
    if nan_count > 0:
        warnings.warn(
            f"Peptide intensities contain {nan_count} null value(s). "
            "ProteoForge expects imputed data.",
            stacklevel=2,
        )

    if config.model == "wls" and provenance_columns_missing(frame):
        msg = (
            "model='wls' requires provenance columns (is_real, is_complete_missing) "
            "or a weight column. Attach provenance before prepare()."
        )
        raise ProteoForgeValidationError(msg)


def provenance_columns_missing(frame: pl.DataFrame) -> bool:
    """Return True when WLS provenance columns are absent."""
    has_weight = WEIGHT in frame.columns
    has_masks = IS_REAL in frame.columns or IS_COMPLETE_MISSING in frame.columns
    return not has_weight and not has_masks


def _validate_peptide_coverage(frame: pl.DataFrame, min_peptides: int) -> None:
    """Check min peptides per protein and reject all-NaN peptide rows (long table)."""
    pairs = frame.select([PROTEIN_ID, PEPTIDE_ID]).unique()
    _validate_min_peptides(
        pairs.get_column(PROTEIN_ID).to_numpy().astype(np.str_),
        pairs.get_column(PEPTIDE_ID).to_numpy().astype(np.str_),
        min_peptides,
    )

    all_nan = (
        frame.group_by([PROTEIN_ID, PEPTIDE_ID])
        .agg(pl.col(INTENSITY).is_nan().all().alias("_all_nan"))
        .filter(pl.col("_all_nan"))
    )
    if all_nan.height > 0:
        msg = "At least one peptide has all-NaN intensities."
        raise ProteoForgeValidationError(msg)


def _select_peptide_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep only columns needed for validation and normalization."""
    columns = [PROTEIN_ID, PEPTIDE_ID, SAMPLE_ID, INTENSITY, CONDITION]
    for optional in (IS_REAL, IS_COMPLETE_MISSING, WEIGHT):
        if optional in frame.columns:
            columns.append(optional)
    return frame.select(columns)


def _validate_min_peptides(
    protein_ids: npt.NDArray[np.str_],
    peptide_ids: npt.NDArray[np.str_],
    min_peptides: int,
) -> None:
    """Check each protein has enough unique peptides."""
    if protein_ids.size == 0:
        return

    pair_dtype = np.dtype(
        [("protein", protein_ids.dtype, ()), ("peptide", peptide_ids.dtype, ())]
    )
    pairs = np.empty(protein_ids.size, dtype=pair_dtype)
    pairs["protein"] = protein_ids
    pairs["peptide"] = peptide_ids
    unique_pairs = np.unique(pairs)
    unique_proteins, counts = np.unique(unique_pairs["protein"], return_counts=True)

    too_few_mask = counts < min_peptides
    if not np.any(too_few_mask):
        return

    offenders = sorted(
        zip(unique_proteins[too_few_mask], counts[too_few_mask], strict=True),
        key=lambda entry: entry[1],
    )[:5]
    msg = (
        f"{int(too_few_mask.sum())} protein(s) have fewer than "
        f"{min_peptides} peptides. Examples: {offenders}. "
        "Increase coverage or lower min_peptides."
    )
    raise ProteoForgeValidationError(msg)


def _ordered_samples(
    design: DesignTable,
    condition_levels: tuple[str, ...],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for condition in condition_levels:
        ordered.extend(design.condition_to_samples[condition])
    return tuple(ordered)
