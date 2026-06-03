"""Canonical column names for ProteoForge data contracts."""

from __future__ import annotations

PROTEIN_ID = "protein_id"
PEPTIDE_ID = "peptide_id"
SAMPLE_ID = "sample_id"
CONDITION = "condition"
INTENSITY = "intensity"
IS_REAL = "is_real"
IS_COMPLETE_MISSING = "is_complete_missing"
WEIGHT = "weight"

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
