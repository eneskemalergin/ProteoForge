"""Provenance mask and weight matrix I/O."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from proteoforge._exceptions import ProteoForgeIOError, ProteoForgeValidationError
from proteoforge.schema import (
    IS_COMPLETE_MISSING,
    IS_REAL,
    PEPTIDE_ID,
    PROTEIN_ID,
    SAMPLE_ID,
    WEIGHT,
)


def read_provenance(
    path: str | Path,
    *,
    fmt: str = "auto",
) -> pl.DataFrame:
    """
    Load provenance masks or precomputed weights from file.

    Parameters
    ----------
    path
        Parquet or delimited file with the same keys as the peptide table.
    fmt
        Reserved for future wide-format support. Currently long-format only.

    Returns
    -------
    polars.DataFrame
        Provenance table with canonical key columns and mask/weight fields.

    Raises
    ------
    ProteoForgeIOError
        If the file cannot be read.
    ProteoForgeValidationError
        If required key columns are missing.
    """
    _ = fmt
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Provenance file not found: {file_path}"
        raise ProteoForgeIOError(msg)

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(file_path)
    elif suffix == ".csv":
        frame = pl.read_csv(file_path)
    elif suffix in {".tsv", ".txt"}:
        frame = pl.read_csv(file_path, separator="\t")
    else:
        msg = (
            f"Unsupported provenance file extension '{suffix}'. "
            "Use .parquet, .csv, or .tsv."
        )
        raise ProteoForgeIOError(msg)

    return provenance_from_frame(frame)


def provenance_from_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """
    Validate and coerce a provenance table.

    Parameters
    ----------
    frame
        Long-format provenance with peptide keys and mask/weight columns.

    Returns
    -------
    polars.DataFrame
        Coerced provenance table.

    Raises
    ------
    ProteoForgeValidationError
        If key columns or value columns are missing.
    """
    key_cols = {PROTEIN_ID, PEPTIDE_ID, SAMPLE_ID}
    value_cols = {IS_REAL, IS_COMPLETE_MISSING, WEIGHT}
    missing_keys = key_cols - set(frame.columns)
    if missing_keys:
        msg = (
            f"Provenance table missing key columns: {sorted(missing_keys)}. "
            f"Required: {sorted(key_cols)}."
        )
        raise ProteoForgeValidationError(msg)

    if not value_cols.intersection(frame.columns):
        msg = f"Provenance table must include at least one of: {sorted(value_cols)}."
        raise ProteoForgeValidationError(msg)

    casts: list[pl.Expr] = [
        pl.col(PROTEIN_ID).cast(pl.String),
        pl.col(PEPTIDE_ID).cast(pl.String),
        pl.col(SAMPLE_ID).cast(pl.String),
    ]
    if IS_REAL in frame.columns:
        casts.append(pl.col(IS_REAL).cast(pl.Boolean))
    if IS_COMPLETE_MISSING in frame.columns:
        casts.append(pl.col(IS_COMPLETE_MISSING).cast(pl.Boolean))
    if WEIGHT in frame.columns:
        casts.append(pl.col(WEIGHT).cast(pl.Float64))

    return frame.with_columns(casts)


def attach_provenance(peptides: pl.DataFrame, provenance: pl.DataFrame) -> pl.DataFrame:
    """
    Join provenance columns onto a peptide table by primary key.

    Parameters
    ----------
    peptides
        Canonical long-format peptide table.
    provenance
        Provenance table with matching keys.

    Returns
    -------
    polars.DataFrame
        Peptide table with provenance columns attached.

    Raises
    ------
    ProteoForgeValidationError
        If provenance keys do not align with the peptide table.
    """
    key_cols = [PROTEIN_ID, PEPTIDE_ID, SAMPLE_ID]
    prov_cols = [
        col
        for col in (IS_REAL, IS_COMPLETE_MISSING, WEIGHT)
        if col in provenance.columns
    ]

    joined = peptides.join(
        provenance.select([*key_cols, *prov_cols]),
        on=key_cols,
        how="left",
        suffix="_prov",
    )

    peptide_keys = peptides.select(key_cols)
    prov_keys = provenance.select(key_cols).unique()
    missing = prov_keys.join(peptide_keys, on=key_cols, how="anti")
    if missing.height > 0:
        examples = missing.head(3).rows()
        msg = (
            f"Provenance contains {missing.height} key(s) not present in peptide "
            f"table. Examples: {examples}. Align keys before calling prepare()."
        )
        raise ProteoForgeValidationError(msg)

    extra = peptide_keys.join(prov_keys, on=key_cols, how="anti")
    if extra.height > 0 and prov_keys.height > 0:
        msg = (
            f"Peptide table has {extra.height} row(s) without matching provenance. "
            "Provide provenance for all peptide entries when using WLS."
        )
        raise ProteoForgeValidationError(msg)

    return joined
