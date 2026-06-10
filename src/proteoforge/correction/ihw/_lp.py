"""Dense linear programming for IHW subproblems (no SciPy runtime)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def _simplex_solve(
    c_shift: npt.NDArray[np.float64],
    a_std: npt.NDArray[np.float64],
    b_std: npt.NDArray[np.float64],
    span: npt.NDArray[np.float64],
    lo: npt.NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> npt.NDArray[np.float64]:
    m, n = a_std.shape
    ncols = n + m + 1
    tableau = np.zeros((m + 1, ncols), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            tableau[i, j] = a_std[i, j]
        tableau[i, n + i] = 1.0
        tableau[i, ncols - 1] = b_std[i]
    for j in range(n):
        tableau[m, j] = -c_shift[j]

    basis = np.empty(m, dtype=np.int64)
    for i in range(m):
        basis[i] = n + i

    for _ in range(max_iter):
        last_row = tableau[m]
        col = -1
        best = 0.0
        for j in range(n + m):
            v = last_row[j]
            if v < -tol and (col < 0 or v < best):
                best = v
                col = j
        if col < 0:
            break

        pivot_row = -1
        best_ratio = np.inf
        for row in range(m):
            pivot = tableau[row, col]
            if pivot > tol:
                ratio = tableau[row, ncols - 1] / pivot
                if ratio < best_ratio:
                    best_ratio = ratio
                    pivot_row = row
        if pivot_row < 0:
            return np.full(n, np.nan)

        pivot_val = tableau[pivot_row, col]
        for j in range(ncols):
            tableau[pivot_row, j] /= pivot_val
        for row in range(m + 1):
            if row == pivot_row:
                continue
            factor = tableau[row, col]
            if factor != 0.0:
                for j in range(ncols):
                    tableau[row, j] -= factor * tableau[pivot_row, j]
        basis[pivot_row] = col
    else:
        return np.full(n, np.nan)

    z = np.zeros(n, dtype=np.float64)
    for row in range(m):
        var = basis[row]
        if var < n:
            z[var] = tableau[row, ncols - 1]

    for j in range(n):
        if z[j] < 0.0:
            z[j] = 0.0
        if z[j] > span[j]:
            z[j] = span[j]

    out = np.empty(n, dtype=np.float64)
    for j in range(n):
        out[j] = lo[j] + z[j]
    return out


def solve_lp_max(
    objective: npt.NDArray[np.float64],
    a_ub: npt.NDArray[np.float64],
    b_ub: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    *,
    tol: float = 1e-9,
) -> npt.NDArray[np.float64]:
    """
    Maximize ``objective @ x`` subject to ``a_ub @ x <= b_ub`` and bounds.

    Parameters
    ----------
    objective
        Linear objective coefficients.
    a_ub
        Inequality constraint matrix.
    b_ub
        Inequality right-hand side.
    lb
        Lower bounds per variable.
    ub
        Upper bounds per variable.
    tol
        Feasibility tolerance.

    Returns
    -------
    ndarray of float64
        Optimal ``x``, or ``nan`` if the solver fails.

    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    c = np.asarray(objective, dtype=np.float64)
    a = np.asarray(a_ub, dtype=np.float64)
    b = np.asarray(b_ub, dtype=np.float64)
    lo = np.asarray(lb, dtype=np.float64)
    hi = np.asarray(ub, dtype=np.float64)
    n = c.shape[0]
    if a.shape[1] != n or lo.shape[0] != n or hi.shape[0] != n:
        msg = "LP dimensions do not match"
        raise ValueError(msg)

    span = hi - lo
    if np.any(span < 0.0):
        msg = "Invalid bounds: ub < lb"
        raise ValueError(msg)

    c_shift = c.copy()
    a_shift = a.copy()
    b_shift = b - a @ lo

    row_blocks: list[npt.NDArray[np.float64]] = []
    rhs: list[float] = []
    if a_shift.size:
        row_blocks.append(a_shift)
        rhs.extend(b_shift.tolist())

    if np.any(np.isfinite(span)):
        bound_rows = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(bound_rows, 1.0)
        finite = np.isfinite(span)
        if np.any(finite):
            row_blocks.append(bound_rows[finite])
            rhs.extend(span[finite].tolist())

    if not row_blocks:
        z = np.where(c_shift > 0.0, span, 0.0)
        z = np.where(np.isfinite(span), z, 0.0)
        return np.asarray(lo + z, dtype=np.float64)

    a_std = np.vstack(row_blocks)
    b_std = np.asarray(rhs, dtype=np.float64)
    return _simplex_solve(c_shift, a_std, b_std, span, lo, tol, 10_000)
