"""Numba GCV smoothing-spline pi0 on the fixed Storey lambda grid."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
from numba import njit

from proteoforge.constants import load_pi0_gcv_spline_matrices

_SQRT_EPS = np.sqrt(2.2e-16)
_GOLDEN_MEAN = 0.5 * (3.0 - np.sqrt(5.0))
_BOUNDED_XATOL = 1e-5
_BOUNDED_MAXITER = 500


def _banded_to_dense(ab: npt.NDArray[np.float64], lower: int, upper: int) -> npt.NDArray[np.float64]:
    n = ab.shape[1]
    dense = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - upper), min(n, j + lower + 1)):
            dense[i, j] = ab[upper + i - j, j]
    return dense


def _compute_banded_symmetric_xt_wy(
    x_banded: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    y_banded: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    w_y = y_banded.copy()
    w_y[2] *= w
    for i in range(2):
        w_y[i, 2 - i :] *= w[:-2 + i]
        w_y[3 + i, :-1 - i] *= w[1 + i :]
    n = x_banded.shape[1]
    out = np.zeros((4, n), dtype=np.float64)
    for i in range(n):
        for j in range(min(n - i, 4)):
            out[-j - 1, i + j] = np.sum(x_banded[j:, i] * w_y[: 5 - j, i + j])
    return out


@njit(cache=True)
def _cholesky_lower(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    l_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = a[i, j]
            for k in range(j):
                s -= l_mat[i, k] * l_mat[j, k]
            if i == j:
                if s <= 0.0:
                    return np.full((n, n), np.nan)
                l_mat[i, j] = np.sqrt(s)
            else:
                l_mat[i, j] = s / l_mat[j, j]
    return l_mat


@njit(cache=True)
def _banded4_to_dense(ab: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = ab.shape[1]
    a = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - 3), j + 1):
            v = ab[3 + i - j, j]
            a[i, j] = v
            a[j, i] = v
    return a


@njit(cache=True)
def _banded5_to_dense(ab: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = ab.shape[1]
    a = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - 2), min(n, j + 3)):
            a[i, j] = ab[2 + i - j, j]
    return a


@njit(cache=True)
def _solve_dense(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = a.shape[0]
    aug = np.empty((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            aug[i, j] = a[i, j]
        aug[i, n] = b[i]
    for col in range(n):
        pivot = col
        best = abs(aug[col, col])
        for row in range(col + 1, n):
            v = abs(aug[row, col])
            if v > best:
                best = v
                pivot = row
        if best == 0.0:
            return np.full(n, np.nan)
        if pivot != col:
            for k in range(n + 1):
                tmp = aug[col, k]
                aug[col, k] = aug[pivot, k]
                aug[pivot, k] = tmp
        div = aug[col, col]
        for k in range(col, n + 1):
            aug[col, k] /= div
        for row in range(n):
            if row == col:
                continue
            factor = aug[row, col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row, k] -= factor * aug[col, k]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = aug[i, n]
    return out


@njit(cache=True)
def _cholesky_banded4_from_dense(lhs_banded: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    a = _banded4_to_dense(lhs_banded)
    l_mat = _cholesky_lower(a)
    if np.any(np.isnan(l_mat)):
        return np.full((4, lhs_banded.shape[1]), np.nan)
    n = a.shape[0]
    u = 3
    out = np.zeros((4, n), dtype=np.float64)
    for j in range(n):
        for i in range(max(0, j - u), j + 1):
            out[u + i - j, j] = l_mat[j, i]
    return out


@njit(cache=True)
def _compute_b_inv_numba(lhs_banded: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = lhs_banded.shape[1]
    nrows = 4
    u = _cholesky_banded4_from_dense(lhs_banded)
    if np.any(np.isnan(u)):
        return np.full((4, n), np.nan)
    for i in range(2, 5):
        row = nrows - i
        for j in range(n - i + 1):
            u[row, (i - 1) + j] /= u[nrows - 1, j]
    d = 1.0 / (u[nrows - 1] ** 2)
    u[nrows - 1, :] = 1.0

    b = np.zeros((4, n), dtype=np.float64)
    for i in range(n - 1, -1, -1):
        for j in range(min(3, n - i - 1), -1, -1):
            rng = min(3, n - i - 1)
            total = 0.0
            if j == 0:
                for k in range(1, rng + 1):
                    total -= u[nrows - k - 1, i + k] * b[nrows - k - 1, i + k]
                total += d[i]
                b[nrows - 1, i] = total
            else:
                for k in range(1, rng + 1):
                    diag = abs(k - j)
                    ind = i + min(k, j)
                    total -= u[nrows - k - 1, i + k] * b[nrows - diag - 1, ind + diag]
                b[nrows - j - 1, i + j] = total
    b[0, :] = 0.0
    return b


@njit(cache=True)
def _gcv_at_lam(
    lam: float,
    x_banded: npt.NDArray[np.float64],
    xtwx_banded: npt.NDArray[np.float64],
    w_e_banded: npt.NDArray[np.float64],
    xte_banded: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> float:
    n = y.shape[0]
    ab = x_banded.copy()
    for row in range(5):
        for col in range(n):
            ab[row, col] = x_banded[row, col] + lam * w_e_banded[row, col]
    a_dense = _banded5_to_dense(ab)
    c = _solve_dense(a_dense, y)
    if np.any(np.isnan(c)):
        return np.inf

    res = np.zeros(n, dtype=np.float64)
    tmp = w_e_banded * c
    for i in range(n):
        s = 0.0
        for j in range(max(0, i - n + 3), min(5, i + 3)):
            s += tmp[j, i + 2 - j]
        res[i] = s
    numer = 0.0
    for i in range(n):
        v = lam * res[i]
        numer += v * v
    numer /= n

    lhs = xtwx_banded.copy()
    for row in range(4):
        for col in range(n):
            lhs[row, col] = xtwx_banded[row, col] + lam * xte_banded[row, col]
    b_inv = _compute_b_inv_numba(lhs)
    if np.any(np.isnan(b_inv)):
        return np.inf
    tr = 0.0
    for row in range(4):
        factor = 2.0 if row < 3 else 1.0
        for col in range(n):
            tr += factor * b_inv[row, col] * xtwx_banded[row, col]
    denom = (1.0 - tr / n) ** 2
    if denom <= 0.0:
        return np.inf
    return numer / denom


@njit(cache=True)
def _minimize_scalar_bounded(
    x_banded: npt.NDArray[np.float64],
    xtwx_banded: npt.NDArray[np.float64],
    w_e_banded: npt.NDArray[np.float64],
    xte_banded: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x1: float,
    x2: float,
    xatol: float,
    maxfun: int,
) -> float:
    a = x1
    b = x2
    fulc = a + _GOLDEN_MEAN * (b - a)
    nfc = fulc
    xf = fulc
    rat = 0.0
    e = 0.0
    x = xf
    fx = _gcv_at_lam(x, x_banded, xtwx_banded, w_e_banded, xte_banded, y)
    num = 1
    fu = np.inf
    ffulc = fx
    fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = _SQRT_EPS * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        if abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and (p < q * (b - xf)):
                rat = p / q
                x = xf + rat
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = 1.0 if xm - xf >= 0.0 else -1.0
                    if xm == xf:
                        si = 0.0
                    rat = tol1 * si
            else:
                golden = 1

        if golden == 1:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = _GOLDEN_MEAN * e

        si = 1.0 if rat >= 0.0 else -1.0
        if rat == 0.0:
            si = 0.0
        ad = abs(rat)
        if ad < tol1:
            ad = tol1
        x = xf + si * ad
        fu = _gcv_at_lam(x, x_banded, xtwx_banded, w_e_banded, xte_banded, y)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc = nfc
            ffulc = fnfc
            nfc = xf
            fnfc = fx
            xf = x
            fx = fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc = nfc
                ffulc = fnfc
                nfc = x
                fnfc = fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc = x
                ffulc = fu

        xm = 0.5 * (a + b)
        tol1 = _SQRT_EPS * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            break

    return xf


@njit(cache=True)
def _endpoint_from_coeffs(x_last: npt.NDArray[np.float64], c: npt.NDArray[np.float64]) -> float:
    s = 0.0
    for i in range(c.shape[0]):
        s += x_last[i] * c[i]
    return s


@njit(cache=True)
def _pi0_gcv_numba_kernel(
    x_banded: npt.NDArray[np.float64],
    xtwx_banded: npt.NDArray[np.float64],
    w_e_banded: npt.NDArray[np.float64],
    xte_banded: npt.NDArray[np.float64],
    x_dense: npt.NDArray[np.float64],
    e_dense: npt.NDArray[np.float64],
    x_last: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    hi: float,
) -> float:
    lam = _minimize_scalar_bounded(
        x_banded,
        xtwx_banded,
        w_e_banded,
        xte_banded,
        y,
        0.0,
        hi,
        _BOUNDED_XATOL,
        _BOUNDED_MAXITER,
    )
    n = y.shape[0]
    a = x_dense.copy()
    for i in range(n):
        for j in range(n):
            a[i, j] = x_dense[i, j] + lam * e_dense[i, j]
    c = _solve_dense(a, y)
    if np.any(np.isnan(c)):
        return np.nan
    return _endpoint_from_coeffs(x_last, c)


class _SplineGCVContext:
    __slots__ = (
        "e_dense",
        "n",
        "w",
        "w_e_banded",
        "x_banded",
        "x_dense",
        "x_dense_last",
        "xte_banded",
        "xtwx_banded",
    )

    def __init__(self) -> None:
        x_banded, w_e_banded = load_pi0_gcv_spline_matrices()
        self.x_banded = x_banded
        self.w_e_banded = w_e_banded
        self.n = self.x_banded.shape[1]
        self.w = np.ones(self.n, dtype=np.float64)
        self.x_dense = _banded_to_dense(self.x_banded, 2, 2)
        self.e_dense = _banded_to_dense(self.w_e_banded, 2, 2)
        self.xtwx_banded = _compute_banded_symmetric_xt_wy(self.x_banded, self.w, self.x_banded)
        self.xte_banded = _compute_banded_symmetric_xt_wy(self.x_banded, self.w, self.w_e_banded)
        self.x_dense_last = np.ascontiguousarray(self.x_dense[-1], dtype=np.float64)


@lru_cache(maxsize=1)
def _spline_context() -> _SplineGCVContext:
    return _SplineGCVContext()


def estimate_pi0_gcv(pi0_lambda: npt.NDArray[np.float64]) -> float:
    """
    Estimate pi0 by GCV smoothing spline on the Storey lambda curve.

    Parameters
    ----------
    pi0_lambda
        Pi0 estimate at each lambda on :data:`~proteoforge.constants.QVALUE_LAMBDAS`.

    Returns
    -------
    float
        GCV pi0 estimate in ``(0, 1]``, or ``nan`` when ``pi0_lambda`` length does
        not match the shipped spline grid.
    """
    ctx = _spline_context()
    y = np.ascontiguousarray(pi0_lambda, dtype=np.float64)
    if y.shape[0] != ctx.n:
        return np.nan
    return float(
        _pi0_gcv_numba_kernel(
            ctx.x_banded,
            ctx.xtwx_banded,
            ctx.w_e_banded,
            ctx.xte_banded,
            ctx.x_dense,
            ctx.e_dense,
            ctx.x_dense_last,
            y,
            float(ctx.n),
        )
    )
