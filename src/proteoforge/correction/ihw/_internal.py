"""IHW cross-validation loop (R ``ihw_internal`` parity)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge.correction import p_adjust
from proteoforge.correction.ihw._convex import ihw_convex
from proteoforge.correction.ihw._splitting import assign_folds
from proteoforge.correction.ihw._utils import safe_divide


def _split_pvalues_by_group(
    pvalues: npt.NDArray[np.float64],
    groups: npt.NDArray[np.intp],
    nbins: int,
) -> list[npt.NDArray[np.float64]]:
    """Return ascending p-value arrays per covariate bin."""
    if pvalues.shape[0] == 0:
        return [np.array([], dtype=np.float64) for _ in range(nbins)]
    order = np.argsort(groups, kind="mergesort")
    g_sorted = groups[order]
    p_sorted = pvalues[order]
    counts = np.bincount(g_sorted, minlength=nbins)
    out: list[npt.NDArray[np.float64]] = []
    pos = 0
    for cnt in counts:
        if cnt == 0:
            out.append(np.array([], dtype=np.float64))
        else:
            out.append(np.sort(p_sorted[pos : pos + cnt]))
            pos += cnt
    return out


def _select_lambda(
    sorted_groups: npt.NDArray[np.intp],
    sorted_pvalues: npt.NDArray[np.float64],
    alpha: float,
    lambdas: npt.NDArray[np.float64],
    m_groups: npt.NDArray[np.intp],
    *,
    penalty: str,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    rng: np.random.Generator,
) -> float:
    """
    Choose the regularization strength with the highest nested-CV rejection count.

    For each random split, the same internal fold assignment is reused across
    all candidate ``lambdas`` so candidates are compared on identical held-out data.
    """
    order = np.argsort(sorted_pvalues)
    internal_p = sorted_pvalues[order]
    internal_g = sorted_groups[order]
    n_internal = internal_p.shape[0]
    scores = np.zeros(lambdas.shape[0], dtype=np.float64)
    for _ in range(nsplits_internal):
        inner_folds = assign_folds(n_internal, nfolds_internal, rng)
        for lam_idx, lam in enumerate(lambdas):
            result = ihw_internal(
                internal_g,
                internal_p,
                alpha,
                np.array([lam], dtype=np.float64),
                m_groups,
                penalty=penalty,
                nfolds=nfolds_internal,
                nfolds_internal=1,
                nsplits_internal=1,
                adjustment_type=adjustment_type,
                rng=rng,
                sorted_folds=inner_folds,
            )
            rjs = result["rjs"]
            if not isinstance(rjs, (int, np.integer)):
                msg = "ihw_internal rjs must be integral"
                raise TypeError(msg)
            scores[lam_idx] += float(rjs)
    scores /= float(nsplits_internal)
    return float(lambdas[int(np.argmax(scores))])


def ihw_internal(
    sorted_groups: npt.NDArray[np.intp],
    sorted_pvalues: npt.NDArray[np.float64],
    alpha: float,
    lambdas: npt.NDArray[np.float64],
    m_groups: npt.NDArray[np.intp],
    *,
    penalty: str,
    nfolds: int,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    rng: np.random.Generator,
    sorted_folds: npt.NDArray[np.intp] | None = None,
) -> dict[str, object]:
    """
    Run the IHW k-fold cross-validation weight-learning loop.

    Parameters
    ----------
    sorted_groups
        Zero-based bin index per hypothesis (p-value sorted order).
    sorted_pvalues
        P-values in ascending order.
    alpha
        Nominal significance level.
    lambdas
        Candidate regularization grid.
    m_groups
        Hypothesis count per bin (full data).
    penalty
        ``"total_variation"`` or ``"uniform_deviation"``.
    nfolds
        Outer CV fold count.
    nfolds_internal
        Nested CV folds for lambda selection.
    nsplits_internal
        Random repeats for nested lambda selection.
    adjustment_type
        ``"bh"`` or ``"bonferroni"``.
    rng
        Random generator for fold assignment.
    sorted_folds
        Optional pre-specified zero-based fold labels (sorted order).

    Returns
    -------
    dict
        Keys: ``sorted_weights``, ``sorted_adj_p``, ``sorted_weighted_pvalues``,
        ``rjs``, ``fold_lambdas``, ``weight_matrix``, ``sorted_folds``.
    """
    n = sorted_pvalues.shape[0]
    nbins = m_groups.shape[0]
    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = assign_folds(n, nfolds, rng)

    m_groups_available = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)
    sorted_weights = np.full(n, np.nan, dtype=np.float64)
    weight_matrix = np.full((nbins, nfolds), np.nan, dtype=np.float64)
    fold_lambdas = np.full(nfolds, np.nan, dtype=np.float64)

    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"

    for fold_idx in range(nfolds):
        fold_mask = sorted_folds == fold_idx
        if not np.any(fold_mask):
            weight_matrix[:, fold_idx] = 1.0
            continue

        train_mask = ~fold_mask
        if nfolds == 1:
            train_mask = np.ones(n, dtype=bool)
            fold_weight_mask = np.ones(n, dtype=bool)
        else:
            fold_weight_mask = fold_mask

        train_groups = sorted_groups[train_mask]
        train_pvalues = sorted_pvalues[train_mask]

        if nfolds == 1:
            m_holdout = m_groups.copy()
            m_train = m_groups.copy()
        elif folds_prespecified:
            holdout_counts = np.bincount(
                sorted_groups[fold_mask], minlength=nbins
            ).astype(np.intp)
            m_holdout = holdout_counts
            m_train = m_groups - m_holdout
        else:
            train_counts = np.bincount(train_groups, minlength=nbins).astype(np.intp)
            m_holdout = (
                (m_groups - m_groups_available) / nfolds
                + m_groups_available
                - train_counts
            ).astype(np.intp)
            m_train = (m_groups - m_holdout).astype(np.intp)

        m_holdout = np.maximum(m_holdout, 0)
        m_train = np.maximum(m_train, 0)

        train_split = _split_pvalues_by_group(train_pvalues, train_groups, nbins)

        if lambdas.shape[0] == 1:
            best_lambda = float(lambdas[0])
        else:
            best_lambda = _select_lambda(
                train_groups,
                train_pvalues,
                alpha,
                lambdas,
                m_train,
                penalty=penalty,
                nfolds_internal=nfolds_internal,
                nsplits_internal=nsplits_internal,
                adjustment_type=adjustment_type,
                rng=rng,
            )
        fold_lambdas[fold_idx] = best_lambda

        ws = ihw_convex(
            train_split,
            alpha,
            m_holdout,
            m_train,
            penalty=penalty,
            lambda_=best_lambda,
            adjustment_type=adjustment_type,
        )
        weight_matrix[:, fold_idx] = ws
        sorted_weights[fold_weight_mask] = ws[sorted_groups[fold_weight_mask]]

    sorted_weighted = safe_divide(sorted_pvalues, sorted_weights)
    m_total = int(np.sum(m_groups))
    sorted_adj = p_adjust(sorted_weighted, pad_method, n_tests=m_total)
    rjs = int(np.sum(sorted_adj <= alpha))

    return {
        "fold_lambdas": fold_lambdas,
        "rjs": rjs,
        "sorted_weighted_pvalues": sorted_weighted,
        "sorted_adj_p": sorted_adj,
        "sorted_weights": sorted_weights,
        "sorted_folds": sorted_folds,
        "weight_matrix": weight_matrix,
    }
