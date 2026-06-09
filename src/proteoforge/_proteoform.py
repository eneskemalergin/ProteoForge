"""Differential proteoform assignment from discordance and clusters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.schema import (
    CLUSTER_ID,
    DPF_ID,
    IS_DISCORDANT,
    PEPTIDE_ID,
    PROTEIN_ID,
)
from proteoforge.types import ProteoformMappingResult

if TYPE_CHECKING:
    from proteoforge.types import ClusterResult, DiscordanceResult, PreparedDataset

_PEPTIDE_COUNT = "_peptide_count"
_HAS_DISCORDANT = "_has_discordant"


def assign_proteoforms(
    prepared: PreparedDataset,
    discordance: DiscordanceResult,
    cluster: ClusterResult,
) -> ProteoformMappingResult:
    """
    Assign dPF IDs to every peptide in the prepared scope.

    Parameters
    ----------
    prepared
        Normalized handoff from :func:`proteoforge.prepare.prepare`.
    discordance
        Discordance result from :func:`proteoforge.discordance.run_discordance`.
    cluster
        Cluster result from :func:`proteoforge.cluster.run_cluster`.

    Returns
    -------
    ProteoformMappingResult
        Full-scope per-peptide dPF mapping.

    Raises
    ------
    ProteoForgeValidationError
        If handoff configs differ or the mapping table violates dPF rules.
    """
    _validate_configs(prepared, discordance, cluster)
    base = (
        prepared.peptides.select([PROTEIN_ID, PEPTIDE_ID])
        .unique()
        .sort([PROTEIN_ID, PEPTIDE_ID])
    )
    flags = discordance.table.select([PROTEIN_ID, PEPTIDE_ID, IS_DISCORDANT])
    mapping = (
        base.join(flags, on=[PROTEIN_ID, PEPTIDE_ID], how="left")
        .with_columns(pl.col(IS_DISCORDANT).fill_null(False))
        .join(
            cluster.table.select([PROTEIN_ID, PEPTIDE_ID, CLUSTER_ID]),
            on=[PROTEIN_ID, PEPTIDE_ID],
            how="left",
        )
    )

    if cluster.table.is_empty():
        table = mapping.with_columns(pl.lit(0, dtype=pl.Int64).alias(DPF_ID))
    else:
        dpf_map = _cluster_dpf_map(mapping)
        table = (
            mapping.join(dpf_map, on=[PROTEIN_ID, CLUSTER_ID], how="left")
            .with_columns(pl.col(DPF_ID).fill_null(0).cast(pl.Int64))
            .sort([PROTEIN_ID, PEPTIDE_ID])
        )

    _validate_mapping(table)
    metadata = _mapping_metadata(table)
    return ProteoformMappingResult(
        config=prepared.config,
        table=table.select([PROTEIN_ID, PEPTIDE_ID, IS_DISCORDANT, CLUSTER_ID, DPF_ID]),
        metadata=metadata,
    )


def _validate_configs(
    prepared: PreparedDataset,
    discordance: DiscordanceResult,
    cluster: ClusterResult,
) -> None:
    if prepared.config != discordance.config or prepared.config != cluster.config:
        msg = "PreparedDataset, DiscordanceResult, and ClusterResult configs differ."
        raise ProteoForgeValidationError(msg)


def _cluster_dpf_map(mapping: pl.DataFrame) -> pl.DataFrame:
    clustered = mapping.filter(pl.col(CLUSTER_ID).is_not_null())
    stats = (
        clustered.group_by([PROTEIN_ID, CLUSTER_ID])
        .agg(
            pl.len().alias(_PEPTIDE_COUNT),
            pl.col(IS_DISCORDANT).any().alias(_HAS_DISCORDANT),
        )
        .sort([PROTEIN_ID, CLUSTER_ID])
    )

    rows: list[dict[str, object]] = []
    for protein_df in stats.partition_by(PROTEIN_ID, maintain_order=True):
        next_dpf = 1
        for row in protein_df.iter_rows(named=True):
            has_discordant = bool(row[_HAS_DISCORDANT])
            peptide_count = int(row[_PEPTIDE_COUNT])
            if not has_discordant:
                dpf_id = 0
            elif peptide_count == 1:
                dpf_id = -1
            else:
                dpf_id = next_dpf
                next_dpf += 1
            rows.append(
                {
                    PROTEIN_ID: row[PROTEIN_ID],
                    CLUSTER_ID: row[CLUSTER_ID],
                    DPF_ID: dpf_id,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            PROTEIN_ID: pl.String,
            CLUSTER_ID: pl.Int64,
            DPF_ID: pl.Int64,
        },
    )


def _validate_mapping(table: pl.DataFrame) -> None:
    invalid_negative = table.filter(pl.col(DPF_ID) < -1)
    if not invalid_negative.is_empty():
        msg = "Proteoform mapping produced dpf_id values below -1."
        raise ProteoForgeValidationError(msg)

    discordant_zero = table.filter(pl.col(IS_DISCORDANT) & (pl.col(DPF_ID) == 0))
    if not discordant_zero.is_empty():
        msg = "Discordant peptides must not receive dpf_id = 0."
        raise ProteoForgeValidationError(msg)

    singleton_dpf = table.filter(pl.col(DPF_ID) == -1)
    if not singleton_dpf.is_empty():
        non_discordant_singleton = singleton_dpf.filter(~pl.col(IS_DISCORDANT))
        if not non_discordant_singleton.is_empty():
            msg = "Non-discordant peptides must not receive dpf_id = -1."
            raise ProteoForgeValidationError(msg)

    clustered = table.filter(pl.col(CLUSTER_ID).is_not_null())
    if clustered.is_empty():
        return

    stats = clustered.group_by([PROTEIN_ID, CLUSTER_ID]).agg(
        pl.len().alias(_PEPTIDE_COUNT),
        pl.col(IS_DISCORDANT).any().alias(_HAS_DISCORDANT),
        pl.col(DPF_ID).n_unique().alias("_dpf_unique"),
        pl.col(DPF_ID).first().alias(DPF_ID),
    )
    bad_multi = stats.filter(
        (pl.col(_PEPTIDE_COUNT) > 1) & pl.col(_HAS_DISCORDANT) & (pl.col(DPF_ID) <= 0)
    )
    if not bad_multi.is_empty():
        msg = "Multi-peptide discordant clusters must receive positive dPF IDs."
        raise ProteoForgeValidationError(msg)

    multi_singleton = stats.filter(
        (pl.col(_PEPTIDE_COUNT) > 1) & (pl.col(DPF_ID) == -1)
    )
    if not multi_singleton.is_empty():
        msg = "Multi-peptide clusters must not receive dpf_id = -1."
        raise ProteoForgeValidationError(msg)

    inconsistent = stats.filter(pl.col("_dpf_unique") != 1)
    if not inconsistent.is_empty():
        msg = "All peptides in a cluster must share the same dPF ID."
        raise ProteoForgeValidationError(msg)


def _mapping_metadata(table: pl.DataFrame) -> dict[str, object]:
    dpf_counts = {
        str(row[DPF_ID]): int(row["count"])
        for row in table.group_by(DPF_ID)
        .len()
        .sort(DPF_ID)
        .rename({"len": "count"})
        .iter_rows(named=True)
    }
    return {
        "n_peptides": table.height,
        "n_discordant_peptides": int(table.select(pl.col(IS_DISCORDANT).sum()).item()),
        "n_singleton_peptides": int(table.select((pl.col(DPF_ID) == -1).sum()).item()),
        "n_differential_peptides": int(table.select((pl.col(DPF_ID) > 0).sum()).item()),
        "dpf_counts": dpf_counts,
    }
