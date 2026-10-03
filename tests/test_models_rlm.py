"""Tests for the RLM one-vs-rest backend."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._layout import ProteinBlock
from proteoforge.models._protocol import build_design_stack
from proteoforge.models._rlm import RLMModel, _irls_one
from proteoforge.models._wls import WLSModel


def _block(
    condition_code: np.ndarray,
    peptide_code: np.ndarray,
    response: np.ndarray,
    *,
    n_conditions: int = 2,
) -> ProteinBlock:
    peptide_ids = tuple(f"PEP{i}" for i in range(int(peptide_code.max()) + 1))
    return ProteinBlock(
        protein_id="P1",
        peptide_ids=peptide_ids,
        response=response.astype(np.float64),
        condition_code=condition_code.astype(np.intp),
        peptide_code=peptide_code.astype(np.intp),
        weight=None,
        n_conditions=n_conditions,
    )


def test_median_axis1_matches_numpy() -> None:
    """Fast MAD medians must match ``np.median`` on every row."""
    from proteoforge.models._rlm import _median_axis1

    rng = np.random.default_rng(9)
    for n_obs in (7, 8, 35, 147):
        values = rng.normal(size=(16, n_obs))
        np.testing.assert_allclose(
            np.median(values, axis=1),
            _median_axis1(values),
            rtol=0.0,
            atol=0.0,
        )


def test_pinv_design_batch_matches_numpy() -> None:
    """Fast H1 pinv path must match ``np.linalg.pinv`` on full-rank designs."""
    from proteoforge.models._rlm import _pinv_design_batch

    rng = np.random.default_rng(3)
    m, n_obs, n_params = 12, 40, 6
    design = rng.normal(size=(m, n_obs, n_params))
    design[:, :, 0] = 1.0
    fast = _pinv_design_batch(design)
    for i in range(m):
        expected = np.linalg.pinv(design[i])
        np.testing.assert_allclose(fast[i], expected, rtol=0.0, atol=1e-12)


def test_irls_early_unit_weights_matches_scalar_loop() -> None:
    """Early stop when all Huber weights are 1 must match the scalar dev loop."""
    from proteoforge.models._rlm import (
        _huber_rho,
        _huber_scale,
        _huber_weights,
        _irls_batch,
        _irls_converged,
        _wls_pinv,
    )

    rng = np.random.default_rng(99)
    for _ in range(40):
        n_obs, n_params = int(rng.integers(18, 45)), int(rng.integers(3, 9))
        x = rng.normal(size=(n_obs, n_params))
        x[:, 0] = 1.0
        y = rng.normal(size=n_obs)
        rank = int(np.linalg.matrix_rank(x))
        df_resid = float(n_obs - rank)
        beta, resid, wls_scale = _wls_pinv(x, y, np.ones(n_obs))
        huber_scale = _huber_scale(resid, df_resid=df_resid, nobs=n_obs)
        dev_prev = float(np.sum(_huber_rho(resid / wls_scale)))
        iteration = 1
        while True:
            weights = _huber_weights(resid / huber_scale)
            beta, resid, wls_scale = _wls_pinv(x, y, weights)
            huber_scale = _huber_scale(resid, df_resid=df_resid, nobs=n_obs)
            dev_curr = float(np.sum(_huber_rho(resid / wls_scale)))
            iteration += 1
            if _irls_converged(dev_prev, dev_curr, iteration) or np.all(weights == 1.0):
                break
            dev_prev = dev_curr

        beta_b, resid_b, _, _, _, _ = _irls_batch(x[None], y)
        np.testing.assert_allclose(beta_b[0], beta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(resid_b[0], resid, rtol=0, atol=1e-12)


def test_irls_batch_matches_scalar_loop() -> None:
    """Batched IRLS must match the scalar ``conv='dev'`` loop (iteration semantics)."""
    from proteoforge.models._rlm import (
        _huber_rho,
        _huber_scale,
        _huber_weights,
        _irls_batch,
        _irls_converged,
        _wls_pinv,
    )

    rng = np.random.default_rng(21)
    n_obs, n_params = 30, 6
    x = rng.normal(size=(n_obs, n_params))
    x[:, 0] = 1.0
    y = rng.normal(size=n_obs)
    rank = int(np.linalg.matrix_rank(x))
    df_resid = float(n_obs - rank)
    beta, resid, wls_scale = _wls_pinv(x, y, np.ones(n_obs))
    huber_scale = _huber_scale(resid, df_resid=df_resid, nobs=n_obs)
    dev_prev = float(np.sum(_huber_rho(resid / wls_scale)))
    iteration = 1
    while True:
        beta, resid, wls_scale = _wls_pinv(x, y, _huber_weights(resid / huber_scale))
        huber_scale = _huber_scale(resid, df_resid=df_resid, nobs=n_obs)
        dev_curr = float(np.sum(_huber_rho(resid / wls_scale)))
        iteration += 1
        if _irls_converged(dev_prev, dev_curr, iteration):
            break
        dev_prev = dev_curr

    beta_b, resid_b, _, _, _, _ = _irls_batch(x[None], y)
    np.testing.assert_allclose(beta_b[0], beta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(resid_b[0], resid, rtol=0, atol=1e-12)


def test_rlm_batch_matches_irls_one() -> None:
    """Batched IRLS must match the per-row scalar oracle."""
    rng = np.random.default_rng(12)
    reps, k, n_conditions = 5, 6, 2
    condition_code = np.tile(np.repeat(np.arange(n_conditions), reps), k)
    peptide_code = np.repeat(np.arange(k), n_conditions * reps)
    y = rng.normal(size=condition_code.size)
    block = _block(condition_code, peptide_code, y, n_conditions=n_conditions)
    design = build_design_stack(block)
    response = np.broadcast_to(y, (k, y.size))
    batch = RLMModel().fit_pvalues(design, response, None, n_interaction=1)
    for i in range(k):
        b, r, s, dr, dm, u = _irls_one(design[i], y)
        if not u:
            assert np.isnan(batch[i])
            continue
        from proteoforge._stats import wald_pvalue
        from proteoforge.models._rlm import _h1_cov

        first = design.shape[2] - 1
        cov = _h1_cov(design[i], r, s, df_resid=dr, df_model=dm)
        scalar = wald_pvalue(b[first:], cov[first:, first:], use_f=False, df_resid=dr)
        np.testing.assert_allclose(batch[i], scalar, rtol=1e-10, equal_nan=True)


def _statsmodels_rlm_pvalue(
    y: np.ndarray,
    condition_code: np.ndarray,
    peptide_code: np.ndarray,
    target: int,
    *,
    n_conditions: int,
) -> float:
    pd = pytest.importorskip("pandas")
    smf = pytest.importorskip("statsmodels.formula.api")
    HuberT = pytest.importorskip(
        "statsmodels.robust.norms", exc_type=ImportError
    ).HuberT
    HuberScale = pytest.importorskip(
        "statsmodels.robust.scale", exc_type=ImportError
    ).HuberScale
    frame = pd.DataFrame(
        {
            "y": y,
            "cond": [f"c{c}" for c in condition_code],
            "allothers": np.where(peptide_code == target, "target", "rest"),
        }
    )
    fitted = smf.rlm(
        "y ~ C(cond) * C(allothers)",
        frame,
        M=HuberT(),
    ).fit(scale_est=HuberScale())
    return float(fitted.wald_test_terms(scalar=False).pvalues[-1])


@pytest.mark.parametrize("n_conditions", [2, 3])
def test_rlm_matches_statsmodels(n_conditions: int) -> None:
    pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(4)
    reps = 4
    k = 3
    condition_code = np.tile(np.repeat(np.arange(n_conditions), reps), k)
    peptide_code = np.repeat(np.arange(k), n_conditions * reps)
    y = rng.normal(size=condition_code.size) + 0.3 * peptide_code

    block = _block(condition_code, peptide_code, y, n_conditions=n_conditions)
    design = build_design_stack(block)
    response = np.broadcast_to(y, (k, y.size))
    p_ours = RLMModel().fit_pvalues(
        design, response, None, n_interaction=n_conditions - 1
    )
    for target in range(k):
        expected = _statsmodels_rlm_pvalue(
            y,
            condition_code,
            peptide_code,
            target,
            n_conditions=n_conditions,
        )
        assert p_ours[target] == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_rlm_downweights_single_outlier() -> None:
    reps, k = 6, 4
    condition_code = np.tile(np.repeat(np.arange(2), reps), k)
    peptide_code = np.repeat(np.arange(k), 2 * reps)
    # No real interaction: every peptide flat across conditions.
    y = np.zeros(condition_code.size, dtype=np.float64)
    rng = np.random.default_rng(11)
    y += rng.normal(scale=0.01, size=y.size)
    # Inject one large outlier into peptide 0 in the treated condition.
    target_rows = np.flatnonzero((peptide_code == 0) & (condition_code == 1))
    y[target_rows[0]] += 25.0

    block = _block(condition_code, peptide_code, y)
    design = build_design_stack(block)
    response = np.broadcast_to(y, (k, y.size))

    ols_p = WLSModel().fit_pvalues(design, response, None, n_interaction=1)
    rlm_p = RLMModel().fit_pvalues(design, response, None, n_interaction=1)

    # OLS is dragged toward significance by the outlier; RLM resists it.
    assert rlm_p[0] > ols_p[0]


def _huber_scale_batch_numpy(
    residual: np.ndarray, df_resid: np.ndarray, *, nobs: int
) -> np.ndarray:
    # NumPy implementation used before the numba kernel; kept as the oracle.
    from proteoforge.models import _rlm

    d = _rlm._HUBER_SCALE_D
    h = (
        df_resid
        / nobs
        * (
            d**2
            + (1.0 - d**2) * _rlm._NORM_CDF_D
            - 0.5
            - d / _rlm._SQRT_2PI * np.exp(-0.5 * d**2)
        )
    )
    s = _rlm._mad_batch(residual)
    still = (h > 0.0) & (s > 0.0)
    curr = np.zeros(residual.shape[0], dtype=np.float64)
    curr[still] = s[still]
    for _ in range(1, _rlm._HUBER_SCALE_MAX_ITER):
        idx = np.flatnonzero(still)
        if idx.size == 0:
            break
        r, c, ha = residual[idx], curr[idx], h[idx]
        inside = np.abs(r / c[:, None]) < d
        chi = np.where(inside, (r / c[:, None]) ** 2 / 2.0, d**2 / 2.0)
        nscale = np.sqrt(np.sum(chi, axis=1) / (nobs * ha) * c**2)
        converged = np.abs(nscale - c) <= _rlm._IRLS_TOL
        curr[idx] = nscale
        still[idx[converged]] = False
    return curr


@pytest.mark.parametrize("n_obs", [8, 31, 240])
def test_huber_scale_batch_matches_numpy_reference(n_obs: int) -> None:
    from proteoforge.models._rlm import _huber_scale_batch

    rng = np.random.default_rng(n_obs)
    residual = rng.standard_t(df=3, size=(50, n_obs))
    df_resid = rng.integers(1, n_obs, size=50).astype(np.float64)
    expected = _huber_scale_batch_numpy(residual, df_resid, nobs=n_obs)
    got = _huber_scale_batch(residual, df_resid, nobs=n_obs)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_huber_scale_batch_edge_rows() -> None:
    from proteoforge.models._rlm import _huber_scale_batch

    rng = np.random.default_rng(5)
    residual = rng.normal(size=(4, 12))
    residual[1] = 0.0  # zero spread: MAD is 0
    df_resid = np.array([6.0, 6.0, 0.0, -1.0])  # rows 2 and 3: h <= 0
    got = _huber_scale_batch(residual, df_resid, nobs=12)
    expected = _huber_scale_batch_numpy(residual, df_resid, nobs=12)
    assert got[0] > 0.0
    np.testing.assert_array_equal(got[1:], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)

    one = _huber_scale_batch(residual[:1], df_resid[:1], nobs=12)
    np.testing.assert_allclose(one, expected[:1], rtol=1e-12, atol=0.0)
    empty = _huber_scale_batch(np.empty((0, 12)), np.empty(0), nobs=12)
    assert empty.shape == (0,)


def test_wls_step_batch_matches_per_row_lstsq() -> None:
    from proteoforge.models._rlm import _wls_step_batch

    rng = np.random.default_rng(17)
    m, n_obs, n_params = 12, 40, 6
    design = rng.normal(size=(m, n_obs, n_params))
    design[:, :, 0] = 1.0
    y = rng.normal(size=(m, n_obs))
    weights = rng.uniform(0.05, 1.0, size=(m, n_obs))

    beta, resid, scale = _wls_step_batch(design, y, weights)

    for i in range(m):
        sqrt_w = np.sqrt(weights[i])
        expected_beta = np.linalg.lstsq(
            design[i] * sqrt_w[:, None], y[i] * sqrt_w, rcond=None
        )[0]
        expected_resid = y[i] - design[i] @ expected_beta
        expected_scale = np.sum(weights[i] * expected_resid**2) / (n_obs - n_params)
        np.testing.assert_allclose(beta[i], expected_beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(resid[i], expected_resid, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(scale[i], expected_scale, rtol=1e-10, atol=0.0)
