"""Peptide condition-profile blocks for clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.clustering._protocol import ProteinProfileBlock
from proteoforge.schema import (
    CONDITION,
    IS_DISCORDANT,
    NORMALIZED_INTENSITY,
    PEPTIDE_ID,
    PROTEIN_ID,
)

if TYPE_CHECKING:
    from proteoforge.types import DiscordanceResult, PreparedDataset


def build_profile_blocks(
    prepared: PreparedDataset,
    discordance: DiscordanceResult,
) -> list[ProteinProfileBlock]:
    """
    Build condition profiles for all peptides on discordant proteins.

    Parameters
    ----------
    prepared
        Normalized handoff from :func:`proteoforge.prepare.prepare`.
    discordance
        Discordance result from :func:`proteoforge.discordance.run_discordance`.

    Returns
    -------
    list of ProteinProfileBlock
        Blocks sorted by protein ID.
    """
    _validate_handoff(prepared, discordance)
    discordant_proteins = (
        discordance.table.filter(pl.col(IS_DISCORDANT))
        .get_column(PROTEIN_ID)
        .unique()
        .sort()
        .to_list()
    )
    if not discordant_proteins:
        return []

    flags = discordance.table.select([PROTEIN_ID, PEPTIDE_ID, IS_DISCORDANT])
    work = (
        prepared.peptides.join(flags, on=[PROTEIN_ID, PEPTIDE_ID], how="left")
        .with_columns(pl.col(IS_DISCORDANT).fill_null(False))
        .filter(pl.col(PROTEIN_ID).is_in(discordant_proteins))
    )
    grouped = (
        work.group_by([PROTEIN_ID, PEPTIDE_ID, CONDITION])
        .agg(pl.col(NORMALIZED_INTENSITY).median().alias(NORMALIZED_INTENSITY))
        .sort([PROTEIN_ID, PEPTIDE_ID, CONDITION])
    )
    peptide_flags = (
        work.select([PROTEIN_ID, PEPTIDE_ID, IS_DISCORDANT])
        .unique(subset=[PROTEIN_ID, PEPTIDE_ID])
        .sort([PROTEIN_ID, PEPTIDE_ID])
        .with_columns(pl.int_range(pl.len()).over(PROTEIN_ID).alias("_row"))
    )
    condition_levels = prepared.condition_levels
    if grouped.filter(~pl.col(CONDITION).is_in(list(condition_levels))).height > 0:
        msg = "Grouped profiles contain a condition outside prepared.condition_levels."
        raise ProteoForgeValidationError(msg)

    indexed = (
        grouped.join(
            pl.DataFrame(
                {
                    CONDITION: list(condition_levels),
                    "_col": np.arange(len(condition_levels), dtype=np.intp),
                }
            ),
            on=CONDITION,
            how="inner",
        )
        .join(
            peptide_flags.select([PROTEIN_ID, PEPTIDE_ID, "_row"]),
            on=[PROTEIN_ID, PEPTIDE_ID],
            how="inner",
        )
        .select([PROTEIN_ID, PEPTIDE_ID, "_row", "_col", NORMALIZED_INTENSITY])
    )
    if indexed.height != grouped.height:
        msg = "Profile indexing dropped grouped rows (peptide ID mismatch)."
        raise ProteoForgeValidationError(msg)

    n_conditions = len(condition_levels)
    blocks: list[ProteinProfileBlock] = []
    for values_sub, flags_sub in zip(
        indexed.partition_by(PROTEIN_ID, maintain_order=True),
        peptide_flags.partition_by(PROTEIN_ID, maintain_order=True),
        strict=True,
    ):
        protein_id = str(flags_sub.get_column(PROTEIN_ID).item(0))
        peptide_ids = tuple(
            str(value) for value in flags_sub.get_column(PEPTIDE_ID).to_list()
        )
        n_peptides = len(peptide_ids)
        profiles = _scatter_profile_matrix(
            values_sub,
            n_peptides=n_peptides,
            n_conditions=n_conditions,
            protein_id=protein_id,
        )
        is_discordant = (
            flags_sub.get_column(IS_DISCORDANT)
            .to_numpy()
            .astype(
                np.bool_,
                copy=False,
            )
        )
        blocks.append(
            ProteinProfileBlock(
                protein_id=protein_id,
                peptide_ids=peptide_ids,
                profiles=profiles,
                is_discordant=is_discordant,
                condition_levels=condition_levels,
            )
        )
    return blocks


def _validate_handoff(
    prepared: PreparedDataset,
    discordance: DiscordanceResult,
) -> None:
    if prepared.config != discordance.config:
        msg = "PreparedDataset and DiscordanceResult were built with different configs."
        raise ProteoForgeValidationError(msg)
    missing = {PROTEIN_ID, PEPTIDE_ID, IS_DISCORDANT} - set(discordance.table.columns)
    if missing:
        msg = f"DiscordanceResult.table is missing required columns: {sorted(missing)}."
        raise ProteoForgeValidationError(msg)
    missing_prepared = {
        PROTEIN_ID,
        PEPTIDE_ID,
        CONDITION,
        NORMALIZED_INTENSITY,
    } - set(prepared.peptides.columns)
    if missing_prepared:
        msg = (
            f"PreparedDataset.peptides is missing columns: {sorted(missing_prepared)}."
        )
        raise ProteoForgeValidationError(msg)


def _scatter_profile_matrix(
    frame: pl.DataFrame,
    *,
    n_peptides: int,
    n_conditions: int,
    protein_id: str,
) -> npt.NDArray[np.float64]:
    """Scatter pre-indexed long values into a dense peptide x condition matrix."""
    matrix = np.zeros((n_peptides, n_conditions), dtype=np.float64)
    if frame.is_empty():
        return matrix
    rows = frame.get_column("_row").to_numpy()
    cols = frame.get_column("_col").to_numpy()
    matrix[rows, cols] = frame.get_column(NORMALIZED_INTENSITY).to_numpy()
    if not np.all(np.isfinite(matrix)):
        msg = (
            f"Protein '{protein_id}' has non-finite condition profiles "
            "after aggregation."
        )
        raise ProteoForgeValidationError(msg)
    return matrix
