"""Weighted least squares one-vs-rest backend (batched)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from proteoforge._stats import sf_f
from proteoforge.models._fit_status import (
    FIT_STATUS_ILL_CONDITIONED,
    FIT_STATUS_INSUFFICIENT_DF,
    FIT_STATUS_OK,
    FIT_STATUS_WALD_FAILED,
    FIT_STATUS_ZERO_SCALE,
    empty_status,
)

_RCOND: float = 1e-12


class WLSModel:
    """
    Weighted least squares interaction backend.

    Solves a batch of one-vs-rest designs in closed form and tests the
    interaction block with an F test, matching statsmodels ``wls`` plus
    ``wald_test_terms``. With unit weights this reduces to OLS.
    """

    name: str = "wls"
    use_f: bool = True

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
        m, n_obs, n_params = design.shape
        out = np.full(m, np.nan, dtype=np.float64)
        status = empty_status(m)
        df_resid = float(n_obs - n_params)
        if df_resid <= 0.0 or m == 0:
            status[:] = FIT_STATUS_INSUFFICIENT_DF
            return out, status

        unit = _is_unit_weight(weight, m, n_obs)
        if unit:
            x_w = design
            y_w = response
            w = None
        else:
            w = np.ones((m, n_obs), dtype=np.float64) if weight is None else weight
            sqrt_w = np.sqrt(w)
            x_w = design * sqrt_w[:, :, None]
            y_w = response * sqrt_w

        normal = np.matmul(x_w.transpose(0, 2, 1), x_w)
        rhs = np.einsum("mnp,mn->mp", x_w, y_w)

        beta, usable = _solve_normal_batch(normal, rhs)
        status[~usable] = FIT_STATUS_ILL_CONDITIONED
        if not np.any(usable):
            return out, status

        index = np.flatnonzero(usable)
        beta_ok = beta[usable]
        design_ok = design[usable]
        response_ok = response[usable]
        if w is None:
            w_ok = np.ones((usable.sum(), n_obs), dtype=np.float64)
        else:
            w_ok = w[usable]

        prediction = np.einsum("mnp,mp->mn", design_ok, beta_ok)
        residual = response_ok - prediction
        ssr = np.einsum("mn,mn->m", w_ok, residual**2)
        scale = ssr / df_resid

        positive_scale = scale > 0.0
        status[index[~positive_scale]] = FIT_STATUS_ZERO_SCALE
        if not np.any(positive_scale):
            return out, status

        first = n_params - n_interaction
        normal_ok = normal[usable][positive_scale]
        beta_block = beta_ok[positive_scale][:, first:]
        cov = scale[positive_scale, None, None] * np.linalg.inv(normal_ok)
        cov_block = cov[:, first:, first:]
        solved = np.linalg.solve(cov_block, beta_block[:, :, None])[:, :, 0]
        stat = np.einsum("mj,mj->m", beta_block, solved)

        f_value = stat / float(n_interaction)
        p_value = sf_f(f_value, float(n_interaction), df_resid)
        p_value[~np.isfinite(stat) | (stat < 0.0)] = np.nan

        resolved = np.full(index.size, np.nan, dtype=np.float64)
        resolved[positive_scale] = p_value
        out[index] = resolved

        ok_rows = index[positive_scale]
        wald_ok = np.isfinite(out[ok_rows])
        status[ok_rows[wald_ok]] = FIT_STATUS_OK
        status[ok_rows[~wald_ok]] = FIT_STATUS_WALD_FAILED
        return out, status


def _is_unit_weight(
    weight: npt.NDArray[np.float64] | None,
    m: int,
    n_obs: int,
) -> bool:
    if weight is None:
        return True
    if weight.shape != (m, n_obs):
        return False
    return bool(np.all(weight == 1.0))


def _solve_normal_batch(
    normal: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Solve batched normal equations; mark non-finite or ill-conditioned rows."""
    m = normal.shape[0]
    beta = np.full((m, rhs.shape[1]), np.nan, dtype=np.float64)
    usable = np.zeros(m, dtype=bool)

    try:
        solved = np.linalg.solve(normal, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        solved = beta.copy()
        for i in range(m):
            try:
                solved[i] = np.linalg.solve(normal[i], rhs[i])
            except np.linalg.LinAlgError:
                continue

    finite = np.all(np.isfinite(solved), axis=1)
    if not np.any(finite):
        return beta, usable

    check = np.einsum("mij,mj->mi", normal[finite], solved[finite])
    rel = np.max(np.abs(check - rhs[finite]), axis=1)
    denom = np.maximum(1.0, np.max(np.abs(rhs[finite]), axis=1))
    ok = rel <= 1e-8 * denom + 1e-10
    good_idx = np.flatnonzero(finite)[ok]
    beta[good_idx] = solved[good_idx]
    usable[good_idx] = True

    bad_idx = np.flatnonzero(finite)[~ok]
    for i in bad_idx:
        singular = np.linalg.svd(normal[i], compute_uv=False)
        denom_s = singular[0]
        rcond = singular[-1] / denom_s if denom_s > 0.0 else 0.0
        if not (np.isfinite(rcond) and rcond > _RCOND):
            continue
        try:
            beta[i] = np.linalg.solve(normal[i], rhs[i])
        except np.linalg.LinAlgError:
            continue
        usable[i] = True
    return beta, usable
