"""Discordance orchestration for one-vs-rest fitting and FDR."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._correction import p_adjust, p_adjust_by_group
from proteoforge._exceptions import (
    ProteoForgeParallelFallbackWarning,
    ProteoForgeValidationError,
)
from proteoforge._layout import ProteinBlock, build_protein_blocks
from proteoforge._progress import WeightedProgress
from proteoforge.models import select_model
from proteoforge.models._fit_status import count_statuses
from proteoforge.models._protocol import build_design_stack
from proteoforge.schema import (
    ADJUSTED_P_VALUE,
    FIT_STATUS,
    IS_DISCORDANT,
    PEPTIDE_ID,
    PROTEIN_ID,
    RAW_P_VALUE,
    WITHIN_P_VALUE,
)
from proteoforge.types import DiscordanceResult

if TYPE_CHECKING:
    from proteoforge.models._protocol import DiscordanceModel
    from proteoforge.types import PreparedDataset

_BATCHING_STRATEGIES: frozenset[str] = frozenset({"scalar", "protein", "shape"})

_FitGroupKey = tuple[int, int, int, bool]
_PROCESS_POOL_CTX = mp.get_context("spawn")
_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pool_worker_init() -> None:
    """Pin BLAS/OpenMP to one thread per worker (spawn starts a clean interpreter)."""
    for var in _BLAS_THREAD_VARS:
        os.environ[var] = "1"


def _spawn_main_usable() -> bool:
    """
    Return whether ``spawn`` workers can import ``__main__``.

    Spawn re-executes the interpreter entry script. Interactive contexts
    (``python -``, notebooks, some REPLs) expose ``__main__.__file__`` as
    ``'<stdin>'`` or omit it, which makes worker startup fail with
    ``BrokenProcessPool``.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return False
    main_file = getattr(main, "__file__", None)
    if not main_file:
        return False
    if str(main_file).startswith("<"):
        return False
    return Path(main_file).is_file()


def _peptide_count_blocks(blocks: list[ProteinBlock]) -> int:
    return sum(block.n_peptides for block in blocks)


def _peptide_count_shape_group(blocks: list[ProteinBlock], indices: list[int]) -> int:
    return sum(blocks[index].n_peptides for index in indices)


def _warn_parallel_fallback(reason: str) -> None:
    warnings.warn(
        "Parallel discordance fitting requested (n_jobs>1) but the process pool "
        f"is unavailable ({reason}). Falling back to serial fitting with "
        "identical p-values. Run from a script file or pass n_jobs=1 to silence "
        "this warning.",
        ProteoForgeParallelFallbackWarning,
        stacklevel=3,
    )


def run_discordance(
    prepared: PreparedDataset,
    *,
    batching: str = "shape",
    show_progress: bool = False,
    n_jobs: int | None = None,
) -> DiscordanceResult:
    """
    Fit one-vs-rest discordance models and apply two-step FDR correction.

    Parameters
    ----------
    prepared
        Validated, normalized dataset from :func:`proteoforge.prepare`.
    batching
        Fitting grouping strategy: ``"scalar"`` (one design at a time),
        ``"protein"`` (one protein at a time), or ``"shape"`` (designs grouped
        by shape across proteins). All produce identical p-values.
    show_progress
        When ``True``, show a peptide-weighted progress bar during fitting
        (terminal or notebook). Suppressed when stderr/stdout is not
        interactive, in piped logs, or when ``TQDM_DISABLE`` is set.
    n_jobs
        Override ``config.n_jobs`` for shape-group parallelism (``1`` forces serial).
        When ``n_jobs>1`` but the interpreter entry point cannot start spawn workers
        (stdin, notebook cell, etc.), fitting falls back to serial with
        :class:`~proteoforge._exceptions.ProteoForgeParallelFallbackWarning`.

    Returns
    -------
    DiscordanceResult
        Per-peptide raw and adjusted p-values with discordance flags.
        ``metadata`` includes ``parallel_fallback`` and
        ``parallel_fallback_reason`` when shape-group parallelism was
        requested but ran serially (notebooks, stdin, or pool failure).

    Raises
    ------
    ProteoForgeValidationError
        If ``batching`` is not a known strategy or the model has no backend.
    """
    if batching not in _BATCHING_STRATEGIES:
        valid = sorted(_BATCHING_STRATEGIES)
        msg = f"Unknown batching strategy '{batching}'. Valid: {valid}."
        raise ProteoForgeValidationError(msg)

    config = prepared.config
    model = select_model(config.model)
    blocks = build_protein_blocks(prepared)
    jobs_requested = config.n_jobs if n_jobs is None else n_jobs

    if batching == "shape":
        raw, fit_status, fit_meta = _fit_shape_streaming(
            model,
            blocks,
            show_progress=show_progress,
            n_jobs=jobs_requested,
        )
    elif batching == "protein":
        raw, fit_status = _fit_protein_streaming(
            model, blocks, show_progress=show_progress
        )
        fit_meta = _serial_fit_metadata(jobs_requested, parallel_applicable=False)
    else:
        raw, fit_status = _fit_scalar_streaming(
            model, blocks, show_progress=show_progress
        )
        fit_meta = _serial_fit_metadata(jobs_requested, parallel_applicable=False)

    protein_ids: list[str] = []
    peptide_ids: list[str] = []
    protein_code: list[int] = []
    for code, block in enumerate(blocks):
        for peptide in block.peptide_ids:
            protein_ids.append(block.protein_id)
            peptide_ids.append(peptide)
            protein_code.append(code)

    codes = np.asarray(protein_code, dtype=np.intp)
    within = _adjust_within(raw, codes, config.correction_within)
    adjusted = _adjust_global(within, config.correction_global)
    is_discordant = np.isfinite(adjusted) & (adjusted <= config.fdr)

    table = pl.DataFrame(
        {
            PROTEIN_ID: protein_ids,
            PEPTIDE_ID: peptide_ids,
            RAW_P_VALUE: raw,
            WITHIN_P_VALUE: within,
            ADJUSTED_P_VALUE: adjusted,
            IS_DISCORDANT: is_discordant,
            FIT_STATUS: fit_status.astype(str),
        }
    )

    metadata: dict[str, object] = {
        "model": config.model,
        "correction_within": config.correction_within,
        "correction_global": config.correction_global,
        "batching": batching,
        "n_proteins": len(blocks),
        "n_peptides_tested": int(raw.size),
        "n_peptides_flagged": int(np.count_nonzero(is_discordant)),
        "n_peptides_skipped": int(np.count_nonzero(~np.isfinite(raw))),
        "skip_reason_counts": count_statuses(fit_status),
        "fdr": config.fdr,
        **fit_meta,
    }
    return DiscordanceResult(config=config, table=table, metadata=metadata)


def _fit_group_key(block: ProteinBlock) -> _FitGroupKey:
    n_params = 2 * block.n_conditions
    n_interaction = block.n_conditions - 1
    needs_weights = block.weight is not None and not np.all(block.weight == 1.0)
    return (block.n_obs, n_params, n_interaction, needs_weights)


def _stack_group_blocks(
    group_blocks: list[ProteinBlock],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    list[int],
    int,
]:
    """
    Stack one shape group.

    Returns design/response/weight tensors, per-protein counts, and interaction dof.
    """
    design_parts: list[npt.NDArray[np.float64]] = []
    response_parts: list[npt.NDArray[np.float64]] = []
    weight_parts: list[npt.NDArray[np.float64]] = []
    counts: list[int] = []
    n_interaction = 0
    for block in group_blocks:
        n_interaction = block.n_conditions - 1
        design_parts.append(build_design_stack(block))
        response_parts.append(
            np.broadcast_to(block.response, (block.n_peptides, block.n_obs))
        )
        counts.append(block.n_peptides)
        if block.weight is not None:
            weight_parts.append(
                np.broadcast_to(block.weight, (block.n_peptides, block.n_obs))
            )

    design = np.concatenate(design_parts, axis=0)
    response = np.concatenate(response_parts, axis=0)
    weight: npt.NDArray[np.float64] | None = None
    if weight_parts:
        weight = np.concatenate(weight_parts, axis=0)
    return design, response, weight, counts, n_interaction


def _serial_fit_metadata(
    n_jobs_requested: int,
    *,
    parallel_applicable: bool,
    n_shape_groups: int = 0,
    parallel_fallback: bool = False,
    parallel_fallback_reason: str | None = None,
) -> dict[str, object]:
    """Build parallel-run metadata for ``DiscordanceResult.metadata``."""
    return {
        "n_jobs_requested": n_jobs_requested,
        "n_jobs_effective": 1,
        "parallel_applicable": parallel_applicable,
        "n_shape_groups": n_shape_groups,
        "parallel_fallback": parallel_fallback,
        "parallel_fallback_reason": parallel_fallback_reason,
    }


def _resolve_n_jobs(n_jobs: int) -> int:
    """
    Map config ``n_jobs`` to a process-pool worker count.

    ``-1`` caps workers at 8 and uses half the reported CPUs so pool processes
    do not oversubscribe BLAS threads (empirically best on 32-core hosts).
    """
    if n_jobs == -1:
        cpus = os.cpu_count() or 1
        return max(1, min(8, cpus // 2))
    return max(1, n_jobs)


def _fit_shape_group(
    model: DiscordanceModel,
    group_blocks: list[ProteinBlock],
    key: _FitGroupKey,
    starts: list[int],
) -> list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.object_]]]:
    """Fit one shape group and return ``(offset, pvalues, status)`` slices."""
    design, response, weight, counts, n_interaction = _stack_group_blocks(group_blocks)
    if not key[3]:
        weight = None
    pvalues, status = model.fit_pvalues_and_status(
        design,
        response,
        weight,
        n_interaction=n_interaction,
    )
    out: list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.object_]]] = []
    cursor = 0
    for start, count in zip(starts, counts, strict=True):
        out.append(
            (
                start,
                pvalues[cursor : cursor + count],
                status[cursor : cursor + count],
            )
        )
        cursor += count
    return out


def _fit_shape_group_task(
    payload: tuple[str, list[ProteinBlock], _FitGroupKey, list[int]],
) -> list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.object_]]]:
    """Process-pool entry point: rebuild model backend in the worker."""
    model_name, group_blocks, key, starts = payload
    model = select_model(model_name)
    return _fit_shape_group(model, group_blocks, key, starts)


def _shape_group_items(
    blocks: list[ProteinBlock],
) -> tuple[list[int], list[tuple[_FitGroupKey, list[int]]]]:
    """Return peptide offsets and sorted shape-group index lists."""
    groups: dict[_FitGroupKey, list[int]] = {}
    offsets: list[int] = []
    pos = 0
    for index, block in enumerate(blocks):
        offsets.append(pos)
        pos += block.n_peptides
        groups.setdefault(_fit_group_key(block), []).append(index)
    return offsets, sorted(groups.items())


def _fit_shape_serial(
    model: DiscordanceModel,
    blocks: list[ProteinBlock],
    offsets: list[int],
    group_items: list[tuple[_FitGroupKey, list[int]]],
    *,
    show_progress: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    """Fit all shape groups in-process."""
    n_peptides = _peptide_count_blocks(blocks)
    raw = np.full(n_peptides, np.nan, dtype=np.float64)
    status = np.empty(n_peptides, dtype=object)
    with WeightedProgress(
        enabled=show_progress,
        total=n_peptides,
        desc="Fitting (by shape)",
    ) as progress:
        for key, indices in group_items:
            group_blocks = [blocks[index] for index in indices]
            starts = [offsets[index] for index in indices]
            for start, chunk, chunk_status in _fit_shape_group(
                model, group_blocks, key, starts
            ):
                raw[start : start + chunk.size] = chunk
                status[start : start + chunk.size] = chunk_status
            progress.update(_peptide_count_shape_group(blocks, indices))
    return raw, status


def _apply_shape_group_results(
    raw: npt.NDArray[np.float64],
    status: npt.NDArray[np.object_],
    results: list[list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.object_]]]],
) -> None:
    for group_result in results:
        for start, chunk, chunk_status in group_result:
            raw[start : start + chunk.size] = chunk
            status[start : start + chunk.size] = chunk_status


def _fit_shape_parallel(
    model: DiscordanceModel,
    blocks: list[ProteinBlock],
    offsets: list[int],
    group_items: list[tuple[_FitGroupKey, list[int]]],
    *,
    workers: int,
    show_progress: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    """Fit shape groups via a spawn process pool."""
    n_peptides = sum(block.n_peptides for block in blocks)
    raw = np.full(n_peptides, np.nan, dtype=np.float64)
    status = np.empty(n_peptides, dtype=object)
    tasks = [
        (
            model.name,
            [blocks[index] for index in indices],
            key,
            [offsets[index] for index in indices],
        )
        for key, indices in group_items
    ]
    task_peptides = [
        _peptide_count_shape_group(blocks, indices) for _key, indices in group_items
    ]
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=_PROCESS_POOL_CTX,
        initializer=_pool_worker_init,
    )
    try:
        future_to_idx = {
            executor.submit(_fit_shape_group_task, task): idx
            for idx, task in enumerate(tasks)
        }
        results: list[
            list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.object_]]]
        ] = [[] for _ in tasks]
        with WeightedProgress(
            enabled=show_progress,
            total=n_peptides,
            desc="Fitting (by shape)",
        ) as progress:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                progress.update(task_peptides[idx])
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    _apply_shape_group_results(raw, status, results)
    return raw, status


def _fit_shape_streaming(
    model: DiscordanceModel,
    blocks: list[ProteinBlock],
    *,
    show_progress: bool = False,
    n_jobs: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_], dict[str, object]]:
    offsets, group_items = _shape_group_items(blocks)
    n_shape_groups = len(group_items)
    workers = _resolve_n_jobs(n_jobs)

    def _serial_meta(
        *,
        parallel_fallback: bool = False,
        parallel_fallback_reason: str | None = None,
    ) -> dict[str, object]:
        return _serial_fit_metadata(
            n_jobs,
            parallel_applicable=True,
            n_shape_groups=n_shape_groups,
            parallel_fallback=parallel_fallback,
            parallel_fallback_reason=parallel_fallback_reason,
        )

    if workers == 1 or n_shape_groups == 1:
        raw, fit_status = _fit_shape_serial(
            model,
            blocks,
            offsets,
            group_items,
            show_progress=show_progress,
        )
        return raw, fit_status, _serial_meta()

    if not _spawn_main_usable():
        reason = "entry point is not a script file (e.g. stdin or notebook cell)"
        _warn_parallel_fallback(reason)
        raw, fit_status = _fit_shape_serial(
            model,
            blocks,
            offsets,
            group_items,
            show_progress=show_progress,
        )
        return (
            raw,
            fit_status,
            _serial_meta(
                parallel_fallback=True,
                parallel_fallback_reason=reason,
            ),
        )

    try:
        raw, fit_status = _fit_shape_parallel(
            model,
            blocks,
            offsets,
            group_items,
            workers=workers,
            show_progress=show_progress,
        )
        return (
            raw,
            fit_status,
            {
                "n_jobs_requested": n_jobs,
                "n_jobs_effective": workers,
                "parallel_applicable": True,
                "n_shape_groups": n_shape_groups,
                "parallel_fallback": False,
                "parallel_fallback_reason": None,
            },
        )
    except BrokenProcessPool as exc:
        reason = f"worker pool failed ({exc})"
        _warn_parallel_fallback(reason)
        raw, fit_status = _fit_shape_serial(
            model,
            blocks,
            offsets,
            group_items,
            show_progress=show_progress,
        )
        return (
            raw,
            fit_status,
            _serial_meta(
                parallel_fallback=True,
                parallel_fallback_reason=reason,
            ),
        )


def _fit_protein_streaming(
    model: DiscordanceModel,
    blocks: list[ProteinBlock],
    *,
    show_progress: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    n_peptides = _peptide_count_blocks(blocks)
    raw = np.full(n_peptides, np.nan, dtype=np.float64)
    status = np.empty(n_peptides, dtype=object)
    offset = 0
    with WeightedProgress(
        enabled=show_progress,
        total=n_peptides,
        desc="Fitting (per protein)",
    ) as progress:
        for block in blocks:
            design = build_design_stack(block)
            response = np.broadcast_to(block.response, (block.n_peptides, block.n_obs))
            weight = (
                np.broadcast_to(block.weight, (block.n_peptides, block.n_obs))
                if block.weight is not None
                else None
            )
            if weight is not None and np.all(weight == 1.0):
                weight = None
            pvalues, chunk_status = model.fit_pvalues_and_status(
                design,
                response,
                weight,
                n_interaction=block.n_conditions - 1,
            )
            count = block.n_peptides
            raw[offset : offset + count] = pvalues
            status[offset : offset + count] = chunk_status
            offset += count
            progress.update(count)
    return raw, status


def _fit_scalar_streaming(
    model: DiscordanceModel,
    blocks: list[ProteinBlock],
    *,
    show_progress: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    n_peptides = _peptide_count_blocks(blocks)
    raw = np.full(n_peptides, np.nan, dtype=np.float64)
    status = np.empty(n_peptides, dtype=object)
    offset = 0
    with WeightedProgress(
        enabled=show_progress,
        total=n_peptides,
        desc="Fitting (scalar)",
    ) as progress:
        for block in blocks:
            design = build_design_stack(block)
            response = np.broadcast_to(block.response, (block.n_peptides, block.n_obs))
            weight_stack = (
                np.broadcast_to(block.weight, (block.n_peptides, block.n_obs))
                if block.weight is not None
                else None
            )
            for position in range(block.n_peptides):
                weight = (
                    None
                    if weight_stack is None
                    else weight_stack[position : position + 1]
                )
                if weight is not None and np.all(weight == 1.0):
                    weight = None
                pvalue, row_status = model.fit_pvalues_and_status(
                    design[position : position + 1],
                    response[position : position + 1],
                    weight,
                    n_interaction=block.n_conditions - 1,
                )
                raw[offset + position] = pvalue[0]
                status[offset + position] = row_status[0]
                progress.update(1)
            offset += block.n_peptides
    return raw, status


def _adjust_within(
    raw: npt.NDArray[np.float64],
    protein_code: npt.NDArray[np.intp],
    method: str,
) -> npt.NDArray[np.float64]:
    return p_adjust_by_group(raw, protein_code, method)


def _adjust_global(
    within: npt.NDArray[np.float64],
    method: str,
) -> npt.NDArray[np.float64]:
    adjusted = np.full(within.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(within)
    if np.any(finite):
        adjusted[finite] = p_adjust(within[finite], method, n_tests=int(finite.sum()))
    return adjusted
