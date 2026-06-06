"""Robust linear model (Huber M-estimation) one-vs-rest backend."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge._stats import wald_pvalue_batch
from proteoforge.models._fit_status import (
    FIT_STATUS_OK,
    FIT_STATUS_RANK_DEFICIENT,
    FIT_STATUS_WALD_FAILED,
    FIT_STATUS_ZERO_ROBUST_SCALE,
    empty_status,
)
from proteoforge.models._wls import _solve_normal_batch

HUBER_T: float = 1.345
_HUBER_SCALE_D: float = 2.5
_NORM_CDF_D: float = 0.9937903346742238
_MAD_CONSTANT: float = 0.6744897501960817
_SQRT_2PI: float = 2.5066282746310002
_MAX_IRLS_ITER: int = 50
_IRLS_TOL: float = 1e-8
_HUBER_SCALE_MAX_ITER: int = 30


def _irls_converged(dev_prev: float, dev_curr: float, iteration: int) -> bool:
    """Match statsmodels deviance stopping for one fit."""
    change = abs(dev_curr - dev_prev)
    return not (change > _IRLS_TOL and iteration < _MAX_IRLS_ITER)


def _irls_converged_batch(
    dev_prev: npt.NDArray[np.float64],
    dev_curr: npt.NDArray[np.float64],
    iteration: int,
) -> npt.NDArray[np.bool_]:
    """Vectorized ``_irls_converged`` for active rows."""
    change = np.abs(dev_curr - dev_prev)
    return ~((change > _IRLS_TOL) & (iteration < _MAX_IRLS_ITER))


class RLMModel:
    """
    Huber robust linear model interaction backend.

    Iteratively reweighted least squares with Huber-T and Huber proposal-2
    scale, then a chi-square test on the interaction block using the H1
    covariance. Matches statsmodels ``RLM`` with ``HuberScale`` and
    ``conv='dev'``.
    """

    name: str = "rlm"
    use_f: bool = False

    def fit_pvalues(
        self,
        design: npt.NDArray[np.float64],
        response: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64] | None,
        *,
        n_interaction: int,
    ) -> npt.NDArray[np.float64]:
        """Fit designs and return interaction p-values, shape ``(m,)``."""
        pvalues, _ = self.fit_pvalues_and_status(
            design, response, weight, n_interaction=n_interaction
        )
        return pvalues

    def fit_pvalues_and_status(
        self,
        design: npt.NDArray[np.float64],
        response: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64] | None,
        *,
        n_interaction: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
        """Fit designs and return interaction p-values with per-row status."""
        del weight  # RLM uses internal robust weights, not external weights.
        m, _, n_params = design.shape
        out = np.full(m, np.nan, dtype=np.float64)
        status = empty_status(m)
        if m == 0 or n_interaction <= 0:
            return out, status

        valid, df_resid, df_model = _design_degrees_batch(design)
        status[~valid] = FIT_STATUS_RANK_DEFICIENT

        beta, resid, huber_scale, df_resid, df_model, usable = _irls_batch(
            design, response
        )
        status[valid & ~usable] = FIT_STATUS_ZERO_ROBUST_SCALE
        if not np.any(usable):
            return out, status

        first = n_params - n_interaction
        pvalues = _wald_interaction_batch(
            beta,
            resid,
            huber_scale,
            design,
            df_resid,
            df_model,
            first=first,
            n_interaction=n_interaction,
        )
        out[usable] = pvalues[usable]
        wald_ok = usable & np.isfinite(out)
        status[wald_ok] = FIT_STATUS_OK
        status[usable & ~np.isfinite(out)] = FIT_STATUS_WALD_FAILED
        return out, status


def _huber_weights(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Huber-T IRLS weights (statsmodels ``HuberT.weights``)."""
    absz = np.abs(np.asarray(z, dtype=np.float64))
    out = np.ones_like(absz, dtype=np.float64)
    tail = absz > HUBER_T
    out[tail] = HUBER_T / absz[tail]
    return out


def _huber_rho(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Huber-T objective ``rho(z)`` for deviance convergence."""
    absz = np.abs(z)
    test = absz <= HUBER_T
    return np.where(test, 0.5 * z**2, HUBER_T * absz - 0.5 * HUBER_T**2)


def _mad(values: npt.NDArray[np.float64]) -> float:
    center = float(np.median(values))
    return float(np.median(np.abs(values - center))) / _MAD_CONSTANT


def _median_axis1(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Row medians matching ``np.median(values, axis=1)`` without ``_ureduce``."""
    n = values.shape[1]
    mid = n // 2
    if n % 2 == 1:
        part = np.partition(values, mid, axis=1)
        return part[:, mid]
    part = np.partition(values, (mid - 1, mid), axis=1)
    return 0.5 * (part[:, mid - 1] + part[:, mid])


def _mad_batch(residual: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Median absolute deviation per row, shape ``(m,)``."""
    center = _median_axis1(residual)[:, None]
    return _median_axis1(np.abs(residual - center)) / _MAD_CONSTANT


def _huber_scale(
    residual: npt.NDArray[np.float64],
    *,
    df_resid: float,
    nobs: int,
    tol: float = _IRLS_TOL,
    max_iter: int = _HUBER_SCALE_MAX_ITER,
) -> float:
    """Huber proposal-2 scale (statsmodels ``HuberScale``), scalar oracle."""
    return float(
        _huber_scale_batch(
            residual[None, :],
            np.array([df_resid], dtype=np.float64),
            nobs=nobs,
            tol=tol,
            max_iter=max_iter,
        )[0]
    )


def _huber_scale_batch(
    residual: npt.NDArray[np.float64],
    df_resid: npt.NDArray[np.float64],
    *,
    nobs: int,
    tol: float = _IRLS_TOL,
    max_iter: int = _HUBER_SCALE_MAX_ITER,
) -> npt.NDArray[np.float64]:
    """Huber proposal-2 scale for each row of residuals."""
    m = residual.shape[0]
    d = _HUBER_SCALE_D
    h = (
        df_resid
        / nobs
        * (
            d**2
            + (1.0 - d**2) * _NORM_CDF_D
            - 0.5
            - d / _SQRT_2PI * np.exp(-0.5 * d**2)
        )
    )
    s = _mad_batch(residual)
    still = (h > 0.0) & (s > 0.0)
    if not np.any(still):
        return np.zeros(m, dtype=np.float64)

    curr = np.zeros(m, dtype=np.float64)
    curr[still] = s[still]
    for _niter in range(1, max_iter):
        idx = np.flatnonzero(still)
        if idx.size == 0:
            break
        r = residual[idx]
        c = curr[idx]
        ha = h[idx]
        inside = np.abs(r / c[:, None]) < d
        chi = np.where(inside, (r / c[:, None]) ** 2 / 2.0, d**2 / 2.0)
        nscale = np.sqrt(np.sum(chi, axis=1) / (nobs * ha) * c**2)
        converged = np.abs(nscale - c) <= tol
        curr[idx] = nscale
        still[idx[converged]] = False

    return curr


def _matrix_rank_batch(design: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Per-slice rank matching ``np.linalg.matrix_rank``."""
    m, n_rows, n_cols = design.shape
    singular = np.linalg.svd(design, compute_uv=False)
    if singular.size == 0:
        return np.zeros(m, dtype=np.intp)
    leading = np.maximum(singular[..., 0], 1e-300)
    tol = max(n_rows, n_cols) * np.finfo(np.float64).eps * leading
    ranks = np.sum(singular > tol[:, None], axis=-1)
    return np.asarray(ranks, dtype=np.intp)


def _design_degrees_batch(
    design: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validity mask and residual/model degrees of freedom per stacked design."""
    _, n_obs, n_params = design.shape
    rank = _matrix_rank_batch(design)
    valid = (rank >= n_params) & (n_obs > rank)
    df_resid = np.where(valid, n_obs - rank, np.nan).astype(np.float64)
    df_model = np.where(valid, rank - 1, np.nan).astype(np.float64)
    return valid, df_resid, df_model


def _pinv_design_batch(design: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Batched ``pinv(design)`` via normal equations on the small ``p x p`` system.

    Matches ``np.linalg.pinv`` for the full-rank one-vs-rest designs used here.
    """
    xt = np.matmul(design.transpose(0, 2, 1), design)
    p = design.shape[2]
    eye = np.broadcast_to(np.eye(p, dtype=np.float64), (design.shape[0], p, p))
    inv = np.linalg.solve(xt, eye)
    return np.asarray(np.matmul(inv, design.transpose(0, 2, 1)), dtype=np.float64)


def _wls_beta_pinv_batch(
    wexog: npt.NDArray[np.float64],
    wendog: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """``pinv(wexog) @ wendog`` for a stacked weighted design."""
    pinv_w = np.linalg.pinv(wexog)
    return np.asarray(np.einsum("mpn,mn->mp", pinv_w, wendog), dtype=np.float64)


def _wls_step_batch(
    design: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Batched weighted least squares matching statsmodels ``pinv`` WLS.

    Uses batched normal equations when they agree with ``pinv`` (``atol``
    ``1e-13`` on ``beta``); otherwise falls back to batched ``pinv`` for the
    mismatched rows only.
    """
    _, n_obs, n_params = design.shape
    sqrt_w = np.sqrt(weights)
    wexog = design * sqrt_w[:, :, None]
    wendog = y * sqrt_w
    normal = np.matmul(wexog.transpose(0, 2, 1), wexog)
    rhs = np.einsum("mnp,mn->mp", wexog, wendog)
    beta, usable = _solve_normal_batch(normal, rhs)
    need_pinv = ~usable
    if np.any(need_pinv):
        pinv_rows = np.flatnonzero(need_pinv)
        beta[need_pinv] = _wls_beta_pinv_batch(
            wexog[pinv_rows],
            wendog[pinv_rows],
        )

    fitted = np.einsum("mnp,mp->mn", design, beta)
    resid = y - fitted
    wresid = wendog - np.einsum("mnp,mp->mn", wexog, beta)
    denom = max(n_obs - n_params, 1)
    wls_scale = np.einsum("mn,mn->m", wresid, wresid) / denom
    return beta, resid, wls_scale


def _wls_pinv(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    One weighted least squares solve via pseudoinverse (scalar oracle).

    Returns parameters, unweighted residuals, and the WLS scale used in the
    deviance criterion (statsmodels ``_MinimalWLS``).
    """
    beta, resid, wls_scale = _wls_step_batch(
        x[None, :, :],
        y[None, :],
        weights[None, :],
    )
    return beta[0], resid[0], float(wls_scale[0])


def _irls_one(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, float, bool]:
    """
    Fit one design with statsmodels-faithful IRLS (scalar oracle).

    Returns
    -------
    beta, resid, huber_scale, df_resid, df_model, usable
    """
    result = _irls_batch(x[None, :, :], y)
    beta = result[0][0]
    resid = result[1][0]
    huber_scale = float(result[2][0])
    df_resid = float(result[3][0])
    df_model = float(result[4][0])
    usable = bool(result[5][0])
    return beta, resid, huber_scale, df_resid, df_model, usable


def _h1_cov(
    x: npt.NDArray[np.float64],
    resid: npt.NDArray[np.float64],
    huber_scale: float,
    *,
    df_resid: float,
    df_model: float,
) -> npt.NDArray[np.float64]:
    """H1 covariance matrix (statsmodels ``RLMResults.bcov_scaled``)."""
    return np.asarray(
        _h1_cov_batch(
            x[None, :, :],
            resid[None, :],
            np.array([huber_scale], dtype=np.float64),
            df_resid=np.array([df_resid], dtype=np.float64),
            df_model=np.array([df_model], dtype=np.float64),
        )[0],
        dtype=np.float64,
    )


def _h1_cov_batch(
    design: npt.NDArray[np.float64],
    resid: npt.NDArray[np.float64],
    huber_scale: npt.NDArray[np.float64],
    *,
    df_resid: npt.NDArray[np.float64],
    df_model: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """H1 covariance matrices, shape ``(m, p, p)``."""
    m, n_obs, p = design.shape
    cov = np.full((m, p, p), np.nan, dtype=np.float64)
    ok = (huber_scale > 0.0) & (df_resid > 0.0) & np.isfinite(huber_scale)
    if not np.any(ok):
        return cov

    sresid = resid[ok] / huber_scale[ok, None]
    psi_deriv = (np.abs(sresid) <= HUBER_T).astype(np.float64)
    m_term = psi_deriv.mean(axis=1)
    rows = np.flatnonzero(ok)
    good = m_term > 0.0
    if not np.any(good):
        return cov

    rows_good = rows[good]
    var_psiprime = psi_deriv[good].var(axis=1)
    k = 1.0 + (df_model[rows_good] + 1.0) / n_obs * var_psiprime / m_term[good] ** 2
    psi = np.clip(sresid[good], -HUBER_T, HUBER_T)
    ss_psi = np.sum(psi**2, axis=1)
    pinv_x = _pinv_design_batch(design[rows_good])
    normalized = pinv_x @ np.swapaxes(pinv_x, -1, -2)
    factor = (
        k**2
        * (ss_psi / df_resid[rows_good] * huber_scale[rows_good] ** 2)
        / m_term[good] ** 2
    )
    cov[rows_good] = factor[:, None, None] * normalized
    return cov


def _wald_interaction_batch(
    beta: npt.NDArray[np.float64],
    resid: npt.NDArray[np.float64],
    huber_scale: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    df_resid: npt.NDArray[np.float64],
    df_model: npt.NDArray[np.float64],
    *,
    first: int,
    n_interaction: int,
) -> npt.NDArray[np.float64]:
    """Chi-square Wald p-values on the interaction block for each design."""
    del n_interaction  # block width is ``beta.shape[1] - first``
    cov = _h1_cov_batch(
        design,
        resid,
        huber_scale,
        df_resid=df_resid,
        df_model=df_model,
    )
    block_coef = beta[:, first:]
    block_cov = cov[:, first:, first:]
    return wald_pvalue_batch(
        block_coef,
        block_cov,
        use_f=False,
        df_resid=df_resid,
    )


def _response_matrix(
    response: npt.NDArray[np.float64],
    m: int,
    n_obs: int,
) -> npt.NDArray[np.float64]:
    """Materialize per-row responses as ``(m, n_obs)``."""
    if response.ndim == 1:
        if response.shape[0] != n_obs:
            msg = f"response length {response.shape[0]} != n_obs {n_obs}."
            raise ValueError(msg)
        return np.broadcast_to(response, (m, n_obs)).copy()
    if response.shape != (m, n_obs):
        msg = f"response shape {response.shape} != ({m}, {n_obs})."
        raise ValueError(msg)
    if not response.flags.writeable:
        return np.array(response, dtype=np.float64, copy=False)
    return response


def _irls_deviance_batch(
    resid: npt.NDArray[np.float64],
    wls_scale: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sum of Huber rho for deviance convergence, shape ``(m,)``."""
    return np.asarray(
        np.sum(_huber_rho(resid / wls_scale[:, None]), axis=1),
        dtype=np.float64,
    )


def _irls_batch(
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    """
    Batched IRLS over stacked designs.

    Only convergence loops remain scalar in structure: Huber-scale refinement
    and the shared IRLS iteration gate (statsmodels ``conv='dev'``).
    """
    m, n_obs, n_params = design.shape
    y = _response_matrix(response, m, n_obs)

    beta = np.full((m, n_params), np.nan, dtype=np.float64)
    resid = np.full((m, n_obs), np.nan, dtype=np.float64)
    huber_scale = np.zeros(m, dtype=np.float64)
    df_resid = np.full(m, np.nan, dtype=np.float64)
    df_model = np.full(m, np.nan, dtype=np.float64)
    active = np.zeros(m, dtype=bool)
    dev_prev = np.full(m, np.inf, dtype=np.float64)
    wls_dev = np.full(m, np.nan, dtype=np.float64)

    valid, df_resid, df_model = _design_degrees_batch(design)
    if not np.any(valid):
        usable = np.zeros(m, dtype=bool)
        return beta, resid, huber_scale, df_resid, df_model, usable

    unit = np.ones((int(valid.sum()), n_obs), dtype=np.float64)
    b, r, wd = _wls_step_batch(design[valid], y[valid], unit)
    beta[valid] = b
    resid[valid] = r
    wls_dev[valid] = wd

    huber_scale[valid] = _huber_scale_batch(r, df_resid[valid], nobs=n_obs)
    active[valid] = huber_scale[valid] > 0.0
    dev_prev[valid] = _irls_deviance_batch(r, wd)

    iteration = 1
    while active.any():
        act = active
        weights = _huber_weights(resid[act] / huber_scale[act, None])
        b, r, wd = _wls_step_batch(design[act], y[act], weights)
        beta[act] = b
        resid[act] = r
        wls_dev[act] = wd

        huber_scale[act] = _huber_scale_batch(r, df_resid[act], nobs=n_obs)
        active[act] &= huber_scale[act] > 0.0
        if not active.any():
            break

        act = active
        dev_curr = _irls_deviance_batch(resid[act], wls_dev[act])
        iteration += 1
        converged = _irls_converged_batch(dev_prev[act], dev_curr, iteration)
        # Huber weights all 1.0: further IRLS steps unchanged (statsmodels parity).
        converged |= np.all(weights == 1.0, axis=1)
        dev_prev[act] = dev_curr
        idx = np.flatnonzero(act)
        active[idx[converged]] = False

    usable = (huber_scale > 0.0) & np.all(np.isfinite(beta), axis=1)
    return beta, resid, huber_scale, df_resid, df_model, usable
