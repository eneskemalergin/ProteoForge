"""Covariate stratification and CV fold assignment for IHW."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def _assign_tie_ranks(
    order: npt.NDArray[np.intp],
    cov_sorted: npt.NDArray[np.float64],
    ranks: npt.NDArray[np.float64],
    tie_offsets: npt.NDArray[np.float64],
) -> None:
    n = order.shape[0]
    i = 0
    while i < n:
        j = i + 1
        cov_i = cov_sorted[i]
        while j < n and cov_sorted[j] == cov_i:
            j += 1
        base = float(i + 1)
        block_len = j - i
        if block_len == 1:
            ranks[order[i]] = base
        else:
            for k in range(block_len):
                ranks[order[i + k]] = base + tie_offsets[i + k]
        i = j


def groups_by_filter(
    covariates: npt.NDArray[np.float64],
    nbins: int,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.intp]:
    """
    Stratify hypotheses into equal-count covariate bins.

    Parameters
    ----------
    covariates
        Ordinal covariate values (one per hypothesis).
    nbins
        Number of strata.
    rng
        Generator for random tie-breaking (R ``ties.method='random'``).

    Returns
    -------
    ndarray of intp
        Zero-based bin index per hypothesis.
    """
    n = covariates.shape[0]
    if n == 0:
        return np.array([], dtype=np.intp)
    if rng is None:
        rng = np.random.default_rng()

    order = np.argsort(covariates, kind="mergesort")
    cov_sorted = covariates[order]
    ranks = np.empty(n, dtype=np.float64)

    tie_offsets = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and cov_sorted[j] == cov_sorted[i]:
            j += 1
        block_len = j - i
        if block_len > 1:
            tie_offsets[i:j] = rng.permutation(block_len).astype(np.float64)
        i = j

    _assign_tie_ranks(order, cov_sorted, ranks, tie_offsets)
    rfs = ranks / n
    groups = np.ceil(rfs * nbins).astype(np.intp) - 1
    return np.clip(groups, 0, nbins - 1).astype(np.intp)


def assign_folds(
    n: int,
    nfolds: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    """
    Assign hypotheses to cross-validation folds.

    Parameters
    ----------
    n
        Number of hypotheses.
    nfolds
        Number of folds.
    rng
        Random generator (R ``sample(1:nfolds, n, replace=TRUE)``).

    Returns
    -------
    ndarray of intp
        Zero-based fold index per hypothesis.
    """
    if nfolds <= 0:
        msg = f"nfolds must be positive, got {nfolds}"
        raise ValueError(msg)
    return rng.integers(0, nfolds, size=n, dtype=np.intp)
