"""IHW convex relaxation LP (R ``ihw_convex`` parity)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge.correction.ihw._grenander import presorted_grenander
from proteoforge.correction.ihw._lp import solve_lp_max
from proteoforge.correction.ihw._weights import thresholds_to_weights


def ihw_convex(
    split_sorted_pvalues: list[npt.NDArray[np.float64]],
    alpha: float,
    m_groups: npt.NDArray[np.intp],
    m_groups_grenander: npt.NDArray[np.intp],
    *,
    penalty: str,
    lambda_: float,
    adjustment_type: str,
    grenander_binsize: int = 1,
) -> npt.NDArray[np.float64]:
    """
    Solve the IHW convex LP for per-bin weights.

    Parameters
    ----------
    split_sorted_pvalues
        Ascending p-values per covariate bin.
    alpha
        Nominal FDR or FWER level.
    m_groups
        Hypothesis counts for the weight budget (holdout).
    m_groups_grenander
        Hypothesis counts for Grenander estimation (training).
    penalty
        ``"total_variation"`` or ``"uniform_deviation"``.
    lambda_
        Regularization strength; ``0`` returns uniform weights.
    adjustment_type
        ``"bh"`` or ``"bonferroni"``.
    grenander_binsize
        Thinning stride for Grenander knots.

    Returns
    -------
    ndarray of float64
        Per-bin weights with weighted mean 1.

    Raises
    ------
    ValueError
        If ``penalty`` or ``adjustment_type`` is unknown.
    """
    nbins = len(split_sorted_pvalues)
    if lambda_ == 0.0:
        return np.ones(nbins, dtype=np.float64)
    if nbins != m_groups.shape[0]:
        msg = "length of m_groups should equal number of bins"
        raise ValueError(msg)

    clipped = [
        np.where(pv > 1e-20, pv, 0.0).astype(np.float64) for pv in split_sorted_pvalues
    ]
    m = int(np.sum(m_groups))
    grenander_list = [
        presorted_grenander(pv, int(mg), grenander_binsize=grenander_binsize)
        for pv, mg in zip(clipped, m_groups_grenander, strict=True)
    ]

    n_constraints = sum(g.length for g in grenander_list)
    rows = np.zeros((n_constraints, 2 * nbins), dtype=np.float64)
    rhs = np.empty(n_constraints, dtype=np.float64)
    row = 0
    for g_idx, gr in enumerate(grenander_list):
        for k in range(gr.length):
            slope = gr.slope_knots[k]
            rows[row, g_idx] = 1.0
            rows[row, nbins + g_idx] = -slope
            rhs[row] = gr.y_knots[k] - slope * gr.x_knots[k]
            row += 1

    objective = np.zeros(2 * nbins, dtype=np.float64)
    for g in range(nbins):
        objective[g] = float(m_groups[g]) / m * nbins

    n_base = 2 * nbins
    n_aux = 0
    if lambda_ < np.inf:
        if penalty == "total_variation":
            n_aux = nbins - 1
            aux_rows = np.zeros((2 * (nbins - 1) + 1, n_base + n_aux), dtype=np.float64)
            aux_rhs = np.zeros(2 * (nbins - 1) + 1, dtype=np.float64)
            for g in range(nbins - 1):
                r1 = g
                aux_rows[r1, nbins + g + 1] = 1.0
                aux_rows[r1, nbins + g] = -1.0
                aux_rows[r1, n_base + g] = -1.0
                r2 = (nbins - 1) + g
                aux_rows[r2, nbins + g + 1] = -1.0
                aux_rows[r2, nbins + g] = 1.0
                aux_rows[r2, n_base + g] = -1.0
            tv_row = 2 * (nbins - 1)
            for g in range(nbins - 1):
                aux_rows[tv_row, n_base + g] = 1.0
            for g in range(nbins):
                aux_rows[tv_row, nbins + g] = -lambda_ * float(m_groups[g]) / m
        elif penalty == "uniform_deviation":
            n_aux = nbins
            aux_rows = np.zeros((2 * nbins + 1, n_base + n_aux), dtype=np.float64)
            aux_rhs = np.zeros(2 * nbins + 1, dtype=np.float64)
            for g in range(nbins):
                r1 = g
                for h in range(nbins):
                    coeff = float(m) if h == g else 0.0
                    coeff -= float(m_groups[h])
                    if coeff != 0.0:
                        aux_rows[r1, nbins + h] = coeff
                aux_rows[r1, n_base + g] = -1.0
                r2 = nbins + g
                for h in range(nbins):
                    coeff = -float(m) if h == g else 0.0
                    coeff += float(m_groups[h])
                    if coeff != 0.0:
                        aux_rows[r2, nbins + h] = coeff
                aux_rows[r2, n_base + g] = -1.0
            ud_row = 2 * nbins
            for g in range(nbins):
                aux_rows[ud_row, n_base + g] = 1.0
            for g in range(nbins):
                aux_rows[ud_row, nbins + g] = -lambda_ * float(m_groups[g])
        else:
            msg = f"Unknown penalty: {penalty!r}"
            raise ValueError(msg)
        pad = np.zeros((rows.shape[0], n_aux), dtype=np.float64)
        rows = np.vstack([np.hstack([rows, pad]), aux_rows])
        rhs = np.concatenate([rhs, aux_rhs])
        objective = np.concatenate([objective, np.zeros(n_aux, dtype=np.float64)])

    n_vars = objective.shape[0]
    if adjustment_type == "bh":
        fdr_row = np.zeros(n_vars, dtype=np.float64)
        for g in range(nbins):
            fdr_row[g] = -alpha * float(m_groups[g])
            fdr_row[nbins + g] = float(m_groups[g])
        rows = np.vstack([rows, fdr_row])
        rhs = np.concatenate([rhs, np.array([0.0])])
    elif adjustment_type == "bonferroni":
        fwer_row = np.zeros(n_vars, dtype=np.float64)
        for g in range(nbins):
            fwer_row[nbins + g] = float(m_groups[g])
        rows = np.vstack([rows, fwer_row])
        rhs = np.concatenate([rhs, np.array([alpha])])
    else:
        msg = f"Unknown adjustment_type: {adjustment_type!r}"
        raise ValueError(msg)

    lb = np.zeros(n_vars, dtype=np.float64)
    ub = np.full(n_vars, 2.0, dtype=np.float64)
    if n_aux:
        ub[n_base:] = np.inf

    sol = solve_lp_max(objective, rows, rhs, lb, ub)
    if not np.all(np.isfinite(sol)):
        return np.ones(nbins, dtype=np.float64)

    thresholds = np.maximum(sol[nbins : 2 * nbins], 0.0)
    return thresholds_to_weights(thresholds, m_groups)
