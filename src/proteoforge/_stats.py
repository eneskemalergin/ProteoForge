"""NumPy-only tail probabilities for statsmodels-parity p-values (Numerical Recipes)."""

from __future__ import annotations

from math import lgamma

import numpy as np
import numpy.typing as npt

_MAXIT: int = 400
_EPS: float = 3.0e-16
_FPMIN: float = 1.0e-300


def betainc(
    a: float,
    b: float,
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Regularized incomplete beta function ``I_x(a, b)``.

    Parameters
    ----------
    a
        First shape parameter, ``a > 0``.
    b
        Second shape parameter, ``b > 0``.
    x
        Evaluation points in ``[0, 1]``, any shape.

    Returns
    -------
    np.ndarray
        ``I_x(a, b)`` with the same shape as ``x``, dtype float64.

    Notes
    -----
    Uses the continued fraction of Lentz with the symmetry relation
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` to keep convergence fast on both sides
    of ``x = (a + 1) / (a + b + 2)``.
    """
    values = np.asarray(x, dtype=np.float64)
    out = np.empty(values.shape, dtype=np.float64)
    out[values <= 0.0] = 0.0
    out[values >= 1.0] = 1.0

    interior = (values > 0.0) & (values < 1.0)
    if not np.any(interior):
        return out

    xi = values[interior]
    log_beta = lgamma(a + b) - lgamma(a) - lgamma(b)
    front = np.exp(log_beta + a * np.log(xi) + b * np.log1p(-xi))

    threshold = (a + 1.0) / (a + b + 2.0)
    lower = xi < threshold
    result = np.empty_like(xi)
    if np.any(lower):
        result[lower] = front[lower] * _betacf(a, b, xi[lower]) / a
    upper = ~lower
    if np.any(upper):
        result[upper] = 1.0 - front[upper] * _betacf(b, a, 1.0 - xi[upper]) / b

    out[interior] = result
    return out


def gammaincc(a: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Regularized upper incomplete gamma function ``Q(a, x)``.

    Parameters
    ----------
    a
        Shape parameter, ``a > 0``.
    x
        Evaluation points, ``x >= 0``, any shape.

    Returns
    -------
    np.ndarray
        ``Q(a, x) = 1 - P(a, x)`` with the same shape as ``x``, dtype float64.
    """
    values = np.asarray(x, dtype=np.float64)
    out = np.ones(values.shape, dtype=np.float64)
    positive = values > 0.0
    if not np.any(positive):
        return out

    xp = values[positive]
    use_series = xp < (a + 1.0)
    q = np.empty_like(xp)
    if np.any(use_series):
        q[use_series] = 1.0 - _gser(a, xp[use_series])
    use_cf = ~use_series
    if np.any(use_cf):
        q[use_cf] = _gcf(a, xp[use_cf])
    out[positive] = q
    return out


def sf_f(
    f: npt.NDArray[np.float64],
    dfn: float,
    dfd: float,
) -> npt.NDArray[np.float64]:
    """
    Survival function of the F distribution, ``P(F > f)``.

    Parameters
    ----------
    f
        F statistics, any shape.
    dfn
        Numerator degrees of freedom.
    dfd
        Denominator degrees of freedom.

    Returns
    -------
    np.ndarray
        Upper-tail probabilities, dtype float64.
    """
    stat = np.asarray(f, dtype=np.float64)
    out = np.ones(stat.shape, dtype=np.float64)
    positive = stat > 0.0
    if np.any(positive):
        fp = stat[positive]
        out[positive] = betainc(dfd / 2.0, dfn / 2.0, dfd / (dfd + dfn * fp))
    return out


def sf_chi2(stat: npt.NDArray[np.float64], dof: float) -> npt.NDArray[np.float64]:
    """
    Survival function of the chi-square distribution, ``P(X > stat)``.

    Parameters
    ----------
    stat
        Chi-square statistics, any shape.
    dof
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Upper-tail probabilities, dtype float64.
    """
    return gammaincc(dof / 2.0, np.asarray(stat, dtype=np.float64) / 2.0)


def wald_pvalue(
    block_coef: npt.NDArray[np.float64],
    block_cov: npt.NDArray[np.float64],
    *,
    use_f: bool,
    df_resid: float,
) -> float:
    """
    Joint Wald p-value for one coefficient block (scalar oracle path).

    Parameters
    ----------
    block_coef
        Coefficients in the tested block, shape ``(J,)``.
    block_cov
        Covariance submatrix for the block, shape ``(J, J)``.
    use_f
        When ``True`` use an F test (WLS, OLS); otherwise chi-square (RLM).
    df_resid
        Residual degrees of freedom, used for the F denominator.

    Returns
    -------
    float
        Upper-tail p-value, or ``nan`` when the block is degenerate.
    """
    dof = float(block_coef.size)
    try:
        solved = np.linalg.solve(block_cov, block_coef)
    except np.linalg.LinAlgError:
        return float("nan")
    stat = float(block_coef @ solved)
    if not np.isfinite(stat) or stat < 0.0:
        return float("nan")
    if use_f:
        if df_resid <= 0.0:
            return float("nan")
        return float(sf_f(np.array([stat / dof]), dof, df_resid)[0])
    return float(sf_chi2(np.array([stat]), dof)[0])


def wald_pvalue_batch(
    block_coef: npt.NDArray[np.float64],
    block_cov: npt.NDArray[np.float64],
    *,
    use_f: bool,
    df_resid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Joint Wald p-values for a batch of coefficient blocks.

    Parameters
    ----------
    block_coef
        Shape ``(m, J)``.
    block_cov
        Shape ``(m, J, J)``.
    use_f
        When ``True`` use an F test; otherwise chi-square (RLM).
    df_resid
        Residual degrees of freedom per row, shape ``(m,)``.

    Returns
    -------
    np.ndarray
        Shape ``(m,)`` upper-tail p-values.
    """
    coef = np.asarray(block_coef, dtype=np.float64)
    cov = np.asarray(block_cov, dtype=np.float64)
    m, j = coef.shape
    out = np.full(m, np.nan, dtype=np.float64)
    if j == 0 or m == 0:
        return out

    try:
        solved = np.linalg.solve(cov, coef[..., None])[..., 0]
    except np.linalg.LinAlgError:
        for i in range(m):
            out[i] = wald_pvalue(
                coef[i],
                cov[i],
                use_f=use_f,
                df_resid=float(df_resid[i]),
            )
        return out

    stat = np.einsum("mj,mj->m", coef, solved)
    bad = ~np.isfinite(stat) | (stat < 0.0)
    ok = ~bad
    if not np.any(ok):
        return out
    if use_f:
        dof = float(j)
        out[ok] = sf_f(stat[ok] / dof, dof, df_resid[ok])
    else:
        out[ok] = sf_chi2(stat[ok], float(j))
    return out


def _betacf(
    a: float,
    b: float,
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
    d = 1.0 / d
    h = d.copy()

    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _FPMIN, _FPMIN, c)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < _FPMIN, _FPMIN, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if np.all(np.abs(delta - 1.0) < _EPS):
            break
    return h


def _gser(a: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    gln = lgamma(a)
    ap = np.full_like(x, a)
    total = np.full_like(x, 1.0 / a)
    delta = total.copy()
    done = np.zeros(x.shape, dtype=bool)
    for _ in range(_MAXIT):
        ap = ap + 1.0
        delta = delta * x / ap
        total = total + delta
        done |= np.abs(delta) < np.abs(total) * _EPS
        if np.all(done):
            break
    return total * np.exp(-x + a * np.log(x) - gln)


def _gcf(a: float, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    gln = lgamma(a)
    b = x + 1.0 - a
    c = np.full_like(x, 1.0 / _FPMIN)
    d = 1.0 / b
    h = d.copy()
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < _FPMIN, _FPMIN, d)
        c = b + an / c
        c = np.where(np.abs(c) < _FPMIN, _FPMIN, c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        if np.all(np.abs(delta - 1.0) < _EPS):
            break
    return np.exp(-x + a * np.log(x) - gln) * h
