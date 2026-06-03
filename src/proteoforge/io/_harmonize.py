"""Column harmonization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import polars as pl

from proteoforge.schema import CANONICAL_PEPTIDE_COLUMNS, PROTEIN_ID

if TYPE_CHECKING:
    from proteoforge.types import ColumnMap

PRIORITY_ACCESSION_LENGTH = 6
PROTEIN_GROUP_SEPARATOR = ";"


def select_representative_protein(
    proteins: str,
    *,
    separator: str = PROTEIN_GROUP_SEPARATOR,
    priority_length: int = PRIORITY_ACCESSION_LENGTH,
) -> str:
    """
    Select a representative protein ID from a delimited string.

    Prefers UniProt accessions with ``priority_length`` characters when
    multiple IDs are present.

    Parameters
    ----------
    proteins
        Delimited protein identifier string.
    separator
        Delimiter between protein IDs.
    priority_length
        Preferred accession length (default 6 for canonical UniProt).

    Returns
    -------
    str
        Representative protein ID.
    """
    protein_ids = proteins.split(separator)
    if len(protein_ids) == 1:
        return protein_ids[0]

    priority_ids = [entry for entry in protein_ids if len(entry) == priority_length]
    return priority_ids[0] if priority_ids else protein_ids[0]


@overload
def harmonize_columns(
    frame: pl.LazyFrame,
    column_map: ColumnMap,
) -> pl.LazyFrame: ...


@overload
def harmonize_columns(
    frame: pl.DataFrame,
    column_map: ColumnMap,
) -> pl.DataFrame: ...


def harmonize_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    column_map: ColumnMap,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Rename source columns to canonical ProteoForge names.

    Parameters
    ----------
    frame
        Input peptide table.
    column_map
        Source-to-canonical column mapping.

    Returns
    -------
    polars.DataFrame | polars.LazyFrame
        Frame with canonical column names where mappings exist.
    """
    if isinstance(frame, pl.LazyFrame):
        columns = frame.collect_schema().names()
    else:
        columns = frame.columns
    rename_map = {
        source: target
        for source, target in column_map.as_dict().items()
        if source in columns and source != target
    }
    if rename_map:
        frame = frame.rename(rename_map)
    return frame


def frame_is_canonical(frame: pl.DataFrame, column_map: ColumnMap) -> bool:
    """
    Return True when ``frame`` already uses canonical column names from ``column_map``.
    """
    return _frame_is_canonical_names(set(frame.columns), column_map)


def frame_is_canonical_lazy(lf: pl.LazyFrame, column_map: ColumnMap) -> bool:
    """Lazy variant of :func:`frame_is_canonical`."""
    return _frame_is_canonical_names(set(lf.collect_schema().names()), column_map)


def _frame_is_canonical_names(columns: set[str], column_map: ColumnMap) -> bool:
    if not CANONICAL_PEPTIDE_COLUMNS.issubset(columns):
        return False
    for source, target in column_map.as_dict().items():
        if source != target and source in columns:
            return False
    return True


def protein_ids_need_resolution(
    frame: pl.DataFrame,
    *,
    protein_col: str = PROTEIN_ID,
) -> bool:
    """Return True when any protein ID contains a group delimiter."""
    if protein_col not in frame.columns:
        return False
    return bool(
        frame.select(
            pl.col(protein_col).str.contains(PROTEIN_GROUP_SEPARATOR).any()
        ).item()
    )


def resolve_protein_ids(
    frame: pl.DataFrame,
    *,
    protein_col: str = PROTEIN_ID,
) -> pl.DataFrame:
    """
    Collapse multi-accession protein groups to a representative ID.

    Parameters
    ----------
    frame
        Peptide table containing ``protein_col``.
    protein_col
        Protein identifier column.

    Returns
    -------
    polars.DataFrame
        Copy with representative protein IDs.
    """
    if protein_col not in frame.columns:
        return frame

    if not protein_ids_need_resolution(frame, protein_col=protein_col):
        return frame

    return frame.with_columns(_resolve_protein_ids_expr(protein_col))


def _resolve_protein_ids_expr(protein_col: str) -> pl.Expr:
    parts = pl.col(protein_col).str.split(PROTEIN_GROUP_SEPARATOR)
    preferred = parts.list.eval(
        pl.element().filter(pl.element().str.len_chars() == PRIORITY_ACCESSION_LENGTH)
    ).list.first()
    return pl.coalesce(preferred, parts.list.first()).alias(protein_col)


def resolve_protein_ids_lazy(
    lf: pl.LazyFrame,
    *,
    protein_col: str = PROTEIN_ID,
) -> pl.LazyFrame:
    """Lazy variant of :func:`resolve_protein_ids`."""
    schema = lf.collect_schema().names()
    if protein_col not in schema:
        return lf
    return lf.with_columns(_resolve_protein_ids_expr(protein_col))
