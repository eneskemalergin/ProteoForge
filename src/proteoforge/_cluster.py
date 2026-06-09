"""Clustering orchestration."""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._discordance import (
    _PROCESS_POOL_CTX,
    _pool_worker_init,
    _resolve_n_jobs,
    _spawn_main_usable,
)
from proteoforge._exceptions import (
    ProteoForgeParallelFallbackWarning,
    ProteoForgeValidationError,
)
from proteoforge._progress import WeightedProgress
from proteoforge.clustering._cuts import select_cut_strategy
from proteoforge.clustering._distance import euclidean_condensed
from proteoforge.clustering._linkage import ward_linkage
from proteoforge.clustering._profiles import build_profile_blocks
from proteoforge.schema import (
    CLUSTER_ID,
    CUT_METHOD,
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
    n_jobs: int | None = None,
) -> ClusterResult:
    """
    Cluster peptide condition profiles on proteins with discordant members.

    Parameters
    ----------
    prepared
        Normalized handoff from :func:`proteoforge.prepare.prepare`.
    discordance
        Discordance result from :func:`proteoforge.discordance.run_discordance`.
    show_progress
        When True, show a peptide-weighted progress bar.
    n_jobs
        Override ``config.n_jobs`` for protein-level parallelism.

    Returns
    -------
    ClusterResult
        Per-peptide cluster labels for proteins with at least one discordant
        peptide.

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
    jobs_requested = config.n_jobs if n_jobs is None else n_jobs
    if not blocks:
        return ClusterResult(
            config=config,
            table=_empty_cluster_table(),
            metadata={
                "linkage": config.linkage,
                "cut": config.cut,
                "n_discordant_proteins": 0,
                "n_clustered_peptides": 0,
                **_cluster_metadata(jobs_requested, effective=1),
            },
        )

    rows, run_meta = _cluster_blocks(
        blocks,
        cut=config.cut,
        config=config,
        show_progress=show_progress,
        n_jobs=jobs_requested,
    )
    table = pl.DataFrame(rows).sort([PROTEIN_ID, PEPTIDE_ID])
    metadata: dict[str, object] = {
        "linkage": config.linkage,
        "cut": config.cut,
        "n_discordant_proteins": len(blocks),
        "n_clustered_peptides": table.height,
        **run_meta,
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
    n_jobs: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    workers = _resolve_n_jobs(n_jobs)
    if workers == 1 or len(blocks) == 1:
        return (
            _cluster_blocks_serial(
                blocks,
                cut=cut,
                config=config,
                show_progress=show_progress,
            ),
            _cluster_metadata(n_jobs, effective=1),
        )
    if not _spawn_main_usable():
        reason = "entry point is not a script file (e.g. stdin or notebook cell)"
        _warn_parallel_fallback(reason)
        return (
            _cluster_blocks_serial(
                blocks,
                cut=cut,
                config=config,
                show_progress=show_progress,
            ),
            _cluster_metadata(
                n_jobs,
                effective=1,
                parallel_fallback=True,
                parallel_fallback_reason=reason,
            ),
        )

    try:
        rows = _cluster_blocks_parallel(
            blocks,
            cut=cut,
            config=config,
            show_progress=show_progress,
            workers=workers,
        )
    except BrokenProcessPool as exc:
        reason = f"worker pool failed ({exc})"
        _warn_parallel_fallback(reason)
        return (
            _cluster_blocks_serial(
                blocks,
                cut=cut,
                config=config,
                show_progress=show_progress,
            ),
            _cluster_metadata(
                n_jobs,
                effective=1,
                parallel_fallback=True,
                parallel_fallback_reason=reason,
            ),
        )
    return rows, _cluster_metadata(n_jobs, effective=workers)


def _cluster_blocks_serial(
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


def _cluster_blocks_parallel(
    blocks: list[ProteinProfileBlock],
    *,
    cut: str,
    config: Config,
    show_progress: bool,
    workers: int,
) -> list[dict[str, object]]:
    rows_by_index: dict[int, list[dict[str, object]]] = {}
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_PROCESS_POOL_CTX,
            initializer=_pool_worker_init,
        ) as executor,
        WeightedProgress(
            enabled=show_progress,
            total=sum(block.n_peptides for block in blocks),
            desc="Clustering",
            unit="peptide",
        ) as progress,
    ):
        futures = {
            executor.submit(_cluster_block_rows, block, cut=cut, config=config): idx
            for idx, block in enumerate(blocks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            rows_by_index[idx] = future.result()
            progress.update(blocks[idx].n_peptides)
    rows: list[dict[str, object]] = []
    for idx in range(len(blocks)):
        rows.extend(rows_by_index[idx])
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


def _cluster_metadata(
    n_jobs_requested: int,
    *,
    effective: int,
    parallel_fallback: bool = False,
    parallel_fallback_reason: str | None = None,
) -> dict[str, object]:
    return {
        "n_jobs_requested": n_jobs_requested,
        "n_jobs_effective": effective,
        "parallel_applicable": True,
        "parallel_fallback": parallel_fallback,
        "parallel_fallback_reason": parallel_fallback_reason,
    }


def _warn_parallel_fallback(reason: str) -> None:
    warnings.warn(
        "Parallel clustering requested (n_jobs>1) but the process pool is "
        f"unavailable ({reason}). Falling back to serial clustering.",
        ProteoForgeParallelFallbackWarning,
        stacklevel=3,
    )
