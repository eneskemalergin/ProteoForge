"""Public prepare entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from proteoforge._exceptions import ProteoForgeIOError
from proteoforge._validate import validate_and_prepare
from proteoforge.io._peptides import _scan_peptide_file

if TYPE_CHECKING:
    import polars as pl

    from proteoforge._config import Config
    from proteoforge.types import PreparedDataset

__all__ = ["prepare", "prepare_from_parquet", "validate_and_prepare"]


def prepare(
    peptides: pl.DataFrame | pl.LazyFrame,
    config: Config,
    *,
    provenance: pl.DataFrame | None = None,
) -> PreparedDataset:
    """
    Validate peptide data and return a normalized PreparedDataset.

    Experimental design and sample scope are taken from ``config.conditions``.
    Only samples listed under configured conditions are used.

    Parameters
    ----------
    peptides
        Long-format peptide intensity table or lazy scan.
    config
        Pipeline configuration including ``control_condition`` and
        ``conditions``.
    provenance
        Optional provenance masks or weights.

    Returns
    -------
    PreparedDataset
        Validated, normalized data for downstream modeling.

    Raises
    ------
    ProteoForgeValidationError
        If validation or normalization fails.
    """
    return validate_and_prepare(
        peptides,
        config,
        provenance=provenance,
    )


def prepare_from_parquet(
    path: str | Path,
    config: Config,
    *,
    provenance: pl.DataFrame | None = None,
) -> PreparedDataset:
    """
    Load a parquet peptide table and return a normalized PreparedDataset.

    Uses a lazy scan with column projection and sample filtering before
    materialization. Prefer this over ``read_peptides`` + ``prepare`` when
    starting from a file path.

    Parameters
    ----------
    path
        Parquet file with long-format intensities.
    config
        Pipeline configuration.
    provenance
        Optional provenance masks or weights.

    Returns
    -------
    PreparedDataset
        Validated, normalized data for downstream modeling.

    Raises
    ------
    ProteoForgeIOError
        If ``path`` is missing or not readable parquet.
    ProteoForgeValidationError
        If validation or normalization fails.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Peptide file not found: {file_path}"
        raise ProteoForgeIOError(msg)
    lf = _scan_peptide_file(file_path, config)
    return validate_and_prepare(lf, config, provenance=provenance)
