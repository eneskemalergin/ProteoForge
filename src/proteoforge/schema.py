"""Canonical column name constants for ProteoForge data contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

PROTEIN_ID = "protein_id"
PEPTIDE_ID = "peptide_id"
SAMPLE_ID = "sample_id"
CONDITION = "condition"
INTENSITY = "intensity"
NORMALIZED_INTENSITY = "intensity_normalized"
IS_REAL = "is_real"
IS_COMPLETE_MISSING = "is_complete_missing"
WEIGHT = "weight"
FIT_STATUS = "fit_status"

RAW_P_VALUE = "raw_p_value"
WITHIN_P_VALUE = "within_p_value"
ADJUSTED_P_VALUE = "adjusted_p_value"
IS_DISCORDANT = "is_discordant"
CLUSTER_ID = "cluster_id"
CUT_METHOD = "cut_method"
LINKAGE_METHOD = "linkage_method"
DPF_ID = "dpf_id"

DESIGN_SAMPLE_ID = "sample_id"
DESIGN_CONDITION = "condition"

CANONICAL_PEPTIDE_COLUMNS: frozenset[str] = frozenset(
    {
        PROTEIN_ID,
        PEPTIDE_ID,
        SAMPLE_ID,
        INTENSITY,
    }
)

CANONICAL_PEPTIDE_COLUMNS_WITH_CONDITION: frozenset[str] = CANONICAL_PEPTIDE_COLUMNS | {
    CONDITION,
}

OPTIONAL_PEPTIDE_COLUMNS: frozenset[str] = frozenset(
    {
        IS_REAL,
        IS_COMPLETE_MISSING,
        WEIGHT,
    }
)

DISCORDANCE_RESULT_COLUMNS: frozenset[str] = frozenset(
    {
        PROTEIN_ID,
        PEPTIDE_ID,
        RAW_P_VALUE,
        WITHIN_P_VALUE,
        ADJUSTED_P_VALUE,
        IS_DISCORDANT,
        FIT_STATUS,
    }
)

CLUSTER_RESULT_COLUMNS: frozenset[str] = frozenset(
    {
        PROTEIN_ID,
        PEPTIDE_ID,
        CLUSTER_ID,
        CUT_METHOD,
        LINKAGE_METHOD,
    }
)

PROTEOFORM_MAPPING_COLUMNS: frozenset[str] = frozenset(
    {
        PROTEIN_ID,
        PEPTIDE_ID,
        IS_DISCORDANT,
        CLUSTER_ID,
        DPF_ID,
    }
)


def validate_discordance_result_table(table: pl.DataFrame) -> None:
    """
    Raise when a discordance result frame does not match the contract.

    Parameters
    ----------
    table
        Candidate :class:`~polars.DataFrame`.

    Raises
    ------
    ProteoForgeValidationError
        If required columns are missing or unexpected columns are present.
    """
    from proteoforge._exceptions import ProteoForgeValidationError

    columns = set(table.columns)
    if columns != DISCORDANCE_RESULT_COLUMNS:
        missing = sorted(DISCORDANCE_RESULT_COLUMNS - columns)
        extra = sorted(columns - DISCORDANCE_RESULT_COLUMNS)
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unexpected columns: {extra}")
        msg = "DiscordanceResult.table has invalid columns"
        if parts:
            msg = f"{msg} ({'; '.join(parts)})."
        raise ProteoForgeValidationError(msg)


def validate_cluster_result_table(table: pl.DataFrame) -> None:
    """
    Raise when a cluster result frame does not match the contract.

    Parameters
    ----------
    table
        Candidate :class:`~polars.DataFrame`.

    Raises
    ------
    ProteoForgeValidationError
        If required columns are missing or unexpected columns are present.
    """
    from proteoforge._exceptions import ProteoForgeValidationError

    columns = set(table.columns)
    if columns != CLUSTER_RESULT_COLUMNS:
        missing = sorted(CLUSTER_RESULT_COLUMNS - columns)
        extra = sorted(columns - CLUSTER_RESULT_COLUMNS)
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unexpected columns: {extra}")
        msg = "ClusterResult.table has invalid columns"
        if parts:
            msg = f"{msg} ({'; '.join(parts)})."
        raise ProteoForgeValidationError(msg)


def validate_proteoform_mapping_table(table: pl.DataFrame) -> None:
    """
    Raise when a proteoform mapping frame does not match the contract.

    Parameters
    ----------
    table
        Candidate :class:`~polars.DataFrame`.

    Raises
    ------
    ProteoForgeValidationError
        If required columns are missing or unexpected columns are present.
    """
    from proteoforge._exceptions import ProteoForgeValidationError

    columns = set(table.columns)
    if columns != PROTEOFORM_MAPPING_COLUMNS:
        missing = sorted(PROTEOFORM_MAPPING_COLUMNS - columns)
        extra = sorted(columns - PROTEOFORM_MAPPING_COLUMNS)
        parts: list[str] = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if extra:
            parts.append(f"unexpected columns: {extra}")
        msg = "ProteoformMappingResult.table has invalid columns"
        if parts:
            msg = f"{msg} ({'; '.join(parts)})."
        raise ProteoForgeValidationError(msg)
