"""Clustering orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._progress import WeightedProgress
from proteoforge.clustering._cuts import select_cut_strategy
from proteoforge.clustering._distance import euclidean_condensed
from proteoforge.clustering._linkage import ward_linkage
from proteoforge.clustering._profiles import build_profile_blocks
from proteoforge.schema import (
    CLUSTER_ID,
    CUT_METHOD,
    IS_DISCORDANT,
    LINKAGE_METHOD,
    PEPTIDE_ID,
    PROTEIN_ID,
)
from proteoforge.types import ClusterResult

if TYPE_CHECKING:
    from proteoforge._config import Config
    from proteoforge.clustering._protocol import ProteinProfileBlock
    from proteoforge.types import DiscordanceResult, PreparedDataset


def run_cluster(
    prepared: PreparedDataset,
    discordance: DiscordanceResult,
    *,
    show_progress: bool = False,
) -> ClusterResult:
    """
    Cluster peptide condition profiles for every protein in the prepared scope.

    Parameters
    ----------
    prepared
        Normalized handoff from :func:`proteoforge.prepare.prepare`.
    discordance
        Discordance result from :func:`proteoforge.discordance.run_discordance`.
    show_progress
        When True, show a peptide-weighted progress bar.

    Returns
    -------
    ClusterResult
        Per-peptide cluster labels for all proteins in the prepared scope.

    Notes
    -----
    Proteins are clustered in-process. Each protein is a small Ward problem, so
    a process pool costs more in start-up and data transfer than it saves.

    Raises
    ------
    ProteoForgeValidationError
        If ``config.linkage`` is not ``ward`` or the prepare/discordance
        handoff fails validation.
    """
    config = prepared.config
    if config.linkage != "ward":
        msg = "Only linkage='ward' is supported."
        raise ProteoForgeValidationError(msg)

    blocks = build_profile_blocks(prepared, discordance)
    if not blocks:
        return ClusterResult(
            config=config,
            table=_empty_cluster_table(),
            metadata={
                "linkage": config.linkage,
                "cut": config.cut,
                "n_proteins": 0,
                "n_discordant_proteins": 0,
                "n_clustered_peptides": 0,
            },
        )

    rows = _cluster_blocks(
        blocks,
        cut=config.cut,
        config=config,
        show_progress=show_progress,
    )
    table = pl.DataFrame(rows).sort([PROTEIN_ID, PEPTIDE_ID])
    n_discordant_proteins = int(
        discordance.table.filter(pl.col(IS_DISCORDANT)).select(PROTEIN_ID).n_unique()
    )
    metadata: dict[str, object] = {
        "linkage": config.linkage,
        "cut": config.cut,
        "n_proteins": len(blocks),
        "n_discordant_proteins": n_discordant_proteins,
        "n_clustered_peptides": table.height,
    }
    return ClusterResult(config=config, table=table, metadata=metadata)


def _empty_cluster_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            PROTEIN_ID: pl.String,
            PEPTIDE_ID: pl.String,
            CLUSTER_ID: pl.Int64,
            CUT_METHOD: pl.String,
            LINKAGE_METHOD: pl.String,
        }
    )


def _cluster_blocks(
    blocks: list[ProteinProfileBlock],
    *,
    cut: str,
    config: Config,
    show_progress: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with WeightedProgress(
        enabled=show_progress,
        total=sum(block.n_peptides for block in blocks),
        desc="Clustering",
        unit="peptide",
    ) as progress:
        for block in blocks:
            rows.extend(_cluster_block_rows(block, cut=cut, config=config))
            progress.update(block.n_peptides)
    return rows


def _cluster_block_rows(
    block: ProteinProfileBlock,
    *,
    cut: str,
    config: Config,
) -> list[dict[str, object]]:
    labels = _cluster_profile_block(block, cut=cut, config=config)
    return [
        {
            PROTEIN_ID: block.protein_id,
            PEPTIDE_ID: peptide_id,
            CLUSTER_ID: int(label),
            CUT_METHOD: cut,
            LINKAGE_METHOD: "ward",
        }
        for peptide_id, label in zip(block.peptide_ids, labels, strict=True)
    ]


def _cluster_profile_block(
    block: ProteinProfileBlock,
    *,
    cut: str,
    config: Config,
) -> npt.NDArray[np.intp]:
    """Cluster one protein profile block and return 1-based labels."""
    if block.n_peptides == 0:
        return np.empty(0, dtype=np.intp)
    if block.n_peptides == 1 or block.n_peptides < config.cluster_min_peptides:
        return np.ones(block.n_peptides, dtype=np.intp)
    condensed = euclidean_condensed(block.profiles)
    linkage_matrix = ward_linkage(condensed, n_samples=block.n_peptides)
    strategy = select_cut_strategy(cut)
    return strategy.cut(
        linkage_matrix,
        condensed,
        n_samples=block.n_peptides,
        config=config,
    )
