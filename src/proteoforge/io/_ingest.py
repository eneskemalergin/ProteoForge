"""Shared peptide table materialization for read and prepare paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.io._harmonize import (
    frame_is_canonical,
    frame_is_canonical_lazy,
    harmonize_columns,
    protein_ids_need_resolution,
    resolve_protein_ids,
    resolve_protein_ids_lazy,
)
from proteoforge.schema import (
    CANONICAL_PEPTIDE_COLUMNS,
    CONDITION,
    INTENSITY,
    IS_COMPLETE_MISSING,
    IS_REAL,
    PEPTIDE_ID,
    PROTEIN_ID,
    SAMPLE_ID,
    WEIGHT,
)

if TYPE_CHECKING:
    from proteoforge._config import Config


def sample_ids_in_frame(frame: pl.DataFrame) -> frozenset[str]:
    """Return the unique sample IDs present in a peptide table."""
    return frozenset(frame.get_column(SAMPLE_ID).unique().to_list())


def peptide_table_ready(frame: pl.DataFrame, config: Config) -> bool:
    """
    Return True when ``frame`` is canonical, resolved, and scoped to config samples.

    Used to skip redundant harmonization when chaining :func:`read_peptides` with
    :func:`prepare`.
    """
    if frame.is_empty():
        return False
    if not frame_is_canonical(frame, config.column_map):
        return False
    if protein_ids_need_resolution(frame):
        return False
    return sample_ids_in_frame(frame) <= config.selected_sample_ids


def materialize_peptide_table(
    peptides: pl.DataFrame | pl.LazyFrame,
    config: Config,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """
    Harmonize, scope to configured samples, coerce types, and resolve protein groups.

    Applies the same lazy pipeline used by :func:`read_peptides` so ``prepare``
    behaves identically whether input comes from a scan or an in-memory frame.

    Parameters
    ----------
    peptides
        Long-format input table or lazy scan.
    config
        Pipeline configuration.

    Returns
    -------
    frame
        Canonical long-format table restricted to configured samples.
    samples_dropped_at_ingest
        Sample IDs removed during lazy ingest before the main collect. Empty for
        eager inputs or when no extra samples were present in the scan.

    Raises
    ------
    ProteoForgeValidationError
        If the table is empty or required columns are missing after harmonization.
    """
    if isinstance(peptides, pl.LazyFrame):
        frame, samples_dropped_at_ingest = _collect_lazy_peptides(peptides, config)
    elif isinstance(peptides, pl.DataFrame) and peptide_table_ready(peptides, config):
        frame = peptides
        samples_dropped_at_ingest = ()
    else:
        frame = _prepare_eager_peptides(peptides, config)
        samples_dropped_at_ingest = ()

    if frame.is_empty():
        msg = "Peptide table is empty."
        raise ProteoForgeValidationError(msg)

    missing = CANONICAL_PEPTIDE_COLUMNS - set(frame.columns)
    if missing:
        msg = (
            f"Peptide table missing required columns after harmonization: "
            f"{sorted(missing)}. Set column_map in Config to match your file."
        )
        raise ProteoForgeValidationError(msg)

    return frame, samples_dropped_at_ingest


def _collect_lazy_peptides(
    lf: pl.LazyFrame,
    config: Config,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    if not frame_is_canonical_lazy(lf, config.column_map):
        lf = harmonize_columns(lf, config.column_map)
    selected = config.selected_sample_ids
    present = _lazy_sample_ids(lf)
    samples_dropped_at_ingest = tuple(sorted(present - selected))
    lf = lf.filter(pl.col(SAMPLE_ID).is_in(list(selected)))
    lf = lf.with_columns(_coerce_peptide_exprs(lf.collect_schema().names()))
    lf = resolve_protein_ids_lazy(lf)
    return lf.collect(), samples_dropped_at_ingest


def _lazy_sample_ids(lf: pl.LazyFrame) -> frozenset[str]:
    """Collect unique sample IDs from a lazy scan (one column, small collect)."""
    return frozenset(
        lf.select(pl.col(SAMPLE_ID).unique()).collect().get_column(SAMPLE_ID).to_list()
    )


def _prepare_eager_peptides(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    if not frame_is_canonical(frame, config.column_map):
        frame = harmonize_columns(frame, config.column_map)
    if protein_ids_need_resolution(frame):
        frame = resolve_protein_ids(frame)
    return frame.with_columns(_coerce_peptide_exprs(frame.columns))


def _coerce_peptide_exprs(columns: list[str]) -> list[pl.Expr]:
    casts: list[pl.Expr] = [
        pl.col(PROTEIN_ID).cast(pl.String),
        pl.col(PEPTIDE_ID).cast(pl.String),
        pl.col(SAMPLE_ID).cast(pl.String),
        pl.col(INTENSITY).cast(pl.Float64),
    ]
    if CONDITION in columns:
        casts.append(pl.col(CONDITION).cast(pl.String))
    for col in (IS_REAL, IS_COMPLETE_MISSING):
        if col in columns:
            casts.append(pl.col(col).cast(pl.Boolean))
    if WEIGHT in columns:
        casts.append(pl.col(WEIGHT).cast(pl.Float64))
    return casts
