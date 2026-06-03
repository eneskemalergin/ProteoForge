"""Peptide matrix I/O for long-format tables."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from proteoforge._exceptions import ProteoForgeIOError
from proteoforge.io._ingest import materialize_peptide_table

if TYPE_CHECKING:
    from proteoforge._config import Config


def read_peptides(
    path: str | Path,
    config: Config,
) -> pl.DataFrame:
    """
    Load a long-format peptide intensity table.

    Expects a long table with columns mappable to ``protein_id``, ``peptide_id``,
    ``sample_id``, and ``intensity`` via ``config.column_map``. Supported file
    types: Parquet, CSV, TSV.

    Parameters
    ----------
    path
        Parquet or delimited text file.
    config
        Pipeline configuration with column mappings.

    Returns
    -------
    polars.DataFrame
        Harmonized long-format peptide table with canonical input columns.

    Raises
    ------
    ProteoForgeIOError
        If the file cannot be read.
    ProteoForgeValidationError
        If required columns are missing after harmonization.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Peptide file not found: {file_path}"
        raise ProteoForgeIOError(msg)

    return materialize_peptide_table(_scan_peptide_file(file_path, config), config)[0]


def peptides_from_frame(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """
    Harmonize an in-memory long-format table to canonical input columns.

    .. deprecated::
        Use :func:`materialize_peptide_table` instead.

    Parameters
    ----------
    frame
        Long-format input table.
    config
        Pipeline configuration with ``column_map``.

    Returns
    -------
    polars.DataFrame
        Harmonized peptide table.
    """
    warnings.warn(
        "peptides_from_frame() is deprecated; use materialize_peptide_table() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return materialize_peptide_table(frame, config)[0]


def _input_source_columns(config: Config, available: set[str]) -> list[str]:
    """Return source file columns required for the pipeline."""
    column_map = config.column_map
    sources = (
        column_map.protein_id,
        column_map.peptide_id,
        column_map.sample_id,
        column_map.intensity,
        column_map.condition,
        column_map.is_real,
        column_map.is_complete_missing,
        column_map.weight,
    )
    return [source for source in sources if source in available]


def _scan_peptide_file(path: Path, config: Config) -> pl.LazyFrame:
    """
    Lazy-scan a peptide file with column projection from ``config.column_map``.

    Raises
    ------
    ProteoForgeIOError
        If the extension is not ``.parquet``, ``.csv``, or ``.tsv``.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif suffix in {".csv"}:
        lf = pl.scan_csv(path)
    elif suffix in {".tsv", ".txt"}:
        lf = pl.scan_csv(path, separator="\t")
    else:
        msg = (
            f"Unsupported peptide file extension '{suffix}'. "
            "Use .parquet, .csv, or .tsv."
        )
        raise ProteoForgeIOError(msg)

    available = set(lf.collect_schema().names())
    columns = _input_source_columns(config, available)
    if columns:
        lf = lf.select(columns)
    return lf
