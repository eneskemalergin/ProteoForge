"""Tests for :mod:`proteoforge.correction.ihw`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.stats import norm

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.correction import p_adjust
from proteoforge.correction.ihw import (
    IHWResult,
    adjust_ihw,
    ihw_convex,
    presorted_grenander,
    thresholds_to_weights,
)
from proteoforge.correction.ihw._splitting import assign_folds, groups_by_filter
from proteoforge.correction.ihw._utils import safe_divide


def test_thresholds_to_weights_mean_one() -> None:
    ts = np.array([0.01, 0.05, 0.02])
    m_groups = np.array([100, 200, 100], dtype=np.intp)
    ws = thresholds_to_weights(ts, m_groups)
    np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0)


def test_thresholds_to_weights_all_zero() -> None:
    ts = np.array([0.0, 0.0])
    m_groups = np.array([10, 10], dtype=np.intp)
    np.testing.assert_array_equal(thresholds_to_weights(ts, m_groups), [1.0, 1.0])


def test_thresholds_to_weights_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        thresholds_to_weights(np.array([0.1, 0.2]), np.array([10], dtype=np.intp))


def test_safe_divide_guards() -> None:
    p = np.array([0.0, 0.2, 0.3], dtype=np.float64)
    w = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    out = safe_divide(p, w)
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.15])


def test_groups_by_filter_and_assign_folds() -> None:
    cov = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    groups = groups_by_filter(cov, 2, rng=rng)
    assert groups.shape == cov.shape
    assert np.all((groups >= 0) & (groups < 2))

    folds = assign_folds(8, 4, np.random.default_rng(1))
    assert folds.shape == (8,)
    assert np.all((folds >= 0) & (folds < 4))

    with pytest.raises(ValueError, match="nfolds"):
        assign_folds(4, 0, np.random.default_rng(0))


def test_presorted_grenander_empty_bin() -> None:
    gr = presorted_grenander(np.array([], dtype=np.float64), 10)
    assert gr.length == 1
    np.testing.assert_allclose(gr.slope_knots, [1.0])


def test_ihw_convex_lambda_zero_uniform() -> None:
    split = [np.sort(np.linspace(0.01, 0.9, 50)) for _ in range(3)]
    m = np.array([50, 50, 50], dtype=np.intp)
    ws = ihw_convex(
        split,
        0.1,
        m,
        m,
        penalty="total_variation",
        lambda_=0.0,
        adjustment_type="bh",
    )
    np.testing.assert_array_equal(ws, [1.0, 1.0, 1.0])


def test_ihw_convex_weight_budget() -> None:
    rng = np.random.default_rng(0)
    split = [np.sort(rng.uniform(size=200)) for _ in range(4)]
    m = np.array([200, 200, 200, 200], dtype=np.intp)
    ws = ihw_convex(
        split,
        0.1,
        m,
        m,
        penalty="total_variation",
        lambda_=np.inf,
        adjustment_type="bh",
    )
    np.testing.assert_allclose(np.sum(ws * m) / np.sum(m), 1.0, atol=1e-6)
    assert np.all(ws >= 0.0)


def test_ihw_convex_uniform_deviation_and_bonferroni() -> None:
    rng = np.random.default_rng(2)
    split = [np.sort(rng.uniform(size=80)) for _ in range(3)]
    m = np.array([80, 80, 80], dtype=np.intp)
    ws = ihw_convex(
        split,
        0.1,
        m,
        m,
        penalty="uniform_deviation",
        lambda_=1.0,
        adjustment_type="bonferroni",
    )
    np.testing.assert_allclose(np.sum(ws * m) / np.sum(m), 1.0, atol=1e-6)


def test_ihw_convex_rejects_unknown_penalty() -> None:
    split = [np.array([0.1], dtype=np.float64)]
    m = np.array([1], dtype=np.intp)
    with pytest.raises(ValueError, match="Unknown penalty"):
        ihw_convex(
            split,
            0.1,
            m,
            m,
            penalty="bad",
            lambda_=1.0,
            adjustment_type="bh",
        )


def test_ihw_convex_matches_scipy_lp() -> None:
    rng = np.random.default_rng(1)
    split = [np.sort(rng.beta(0.4, 4, size=150)) for _ in range(3)]
    m = np.array([150, 150, 150], dtype=np.intp)
    ours = ihw_convex(
        split,
        0.1,
        m,
        m,
        penalty="total_variation",
        lambda_=np.inf,
        adjustment_type="bh",
    )

    nbins = 3
    clipped = [np.where(pv > 1e-20, pv, 0.0) for pv in split]
    gren = [presorted_grenander(pv, int(mg)) for pv, mg in zip(clipped, m, strict=True)]
    n_constraints = sum(g.length for g in gren)
    rows = np.zeros((n_constraints, 2 * nbins))
    rhs = np.empty(n_constraints)
    row = 0
    for g_idx, gr in enumerate(gren):
        for k in range(gr.length):
            slope = gr.slope_knots[k]
            rows[row, g_idx] = 1.0
            rows[row, nbins + g_idx] = -slope
            rhs[row] = gr.y_knots[k] - slope * gr.x_knots[k]
            row += 1
    m_total = int(np.sum(m))
    c_obj = np.zeros(2 * nbins)
    for g in range(nbins):
        c_obj[g] = -float(m[g]) / m_total * nbins
    fdr_row = np.zeros(2 * nbins)
    for g in range(nbins):
        fdr_row[g] = -0.1 * float(m[g])
        fdr_row[nbins + g] = float(m[g])
    a_ub = np.vstack([rows, fdr_row])
    b_ub = np.concatenate([rhs, [0.0]])
    bounds = [(0.0, 2.0)] * (2 * nbins)
    res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert res.success
    ts = np.maximum(res.x[nbins : 2 * nbins], 0.0)
    ref = thresholds_to_weights(ts, m)
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_adjust_ihw_single_bin_is_bh() -> None:
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=50))
    x = rng.uniform(size=50)
    result = adjust_ihw(p, x, 0.1, nbins=1, rng=np.random.default_rng(1))
    assert isinstance(result, IHWResult)
    np.testing.assert_allclose(result.adj_pvalues, p_adjust(p, "fdr_bh"))
    np.testing.assert_allclose(result.weights, 1.0)


def test_adjust_ihw_auto_lambdas_runs() -> None:
    rng = np.random.default_rng(0)
    p = np.sort(rng.uniform(size=200))
    x = rng.uniform(size=200)
    result = adjust_ihw(
        p,
        x,
        0.1,
        nbins=4,
        nfolds=2,
        nfolds_internal=2,
        lambdas="auto",
        rng=np.random.default_rng(3),
        seed=3,
    )
    assert result.adj_pvalues.shape == p.shape
    assert np.all(np.isfinite(result.weights))


def test_adjust_ihw_rejects_invalid_inputs() -> None:
    with pytest.raises(ProteoForgeValidationError, match="alpha"):
        adjust_ihw(np.array([0.1]), np.array([1.0]), 1.5)
    with pytest.raises(ProteoForgeValidationError, match="empty"):
        adjust_ihw(np.array([]), np.array([]), 0.1)
    with pytest.raises(ProteoForgeValidationError, match="finite"):
        adjust_ihw(np.array([np.nan]), np.array([1.0]), 0.1)
    with pytest.raises(ProteoForgeValidationError, match="\\[0, 1\\]"):
        adjust_ihw(np.array([1.5]), np.array([1.0]), 0.1)
    with pytest.raises(ProteoForgeValidationError, match="Length mismatch"):
        adjust_ihw(np.array([0.1, 0.2]), np.array([1.0]), 0.1)
    with pytest.raises(ProteoForgeValidationError, match="adjustment_type"):
        adjust_ihw(np.array([0.1]), np.array([1.0]), 0.1, adjustment_type="bad")
    with pytest.raises(ProteoForgeValidationError, match="covariate_type"):
        adjust_ihw(np.array([0.1]), np.array([1.0]), 0.1, covariate_type="bad")
    with pytest.raises(ProteoForgeValidationError, match="folds length"):
        adjust_ihw(
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            0.1,
            nbins=2,
            folds=np.array([0, 1], dtype=np.intp),
        )


def test_adjust_ihw_simulation_more_rejections_than_bh() -> None:
    rng = np.random.default_rng(42)
    m = 5000
    cov = rng.uniform(0, 3, size=m)
    signals = rng.binomial(1, 0.15, size=m)
    z = rng.normal(loc=signals * cov)
    p = 1.0 - norm.cdf(z)
    result = adjust_ihw(p, cov, 0.1, rng=np.random.default_rng(1), seed=1)
    bh_rej = int(np.sum(p_adjust(p, "fdr_bh") <= 0.1))
    ihw_rej = int(np.sum(result.adj_pvalues <= 0.1))
    assert ihw_rej >= bh_rej


def test_presorted_grenander_monotone_slopes() -> None:
    pv = np.sort(np.linspace(0.0, 1.0, 100))
    gr = presorted_grenander(pv, 100)
    if gr.slope_knots.size > 1:
        assert np.all(np.diff(gr.slope_knots) <= 1e-12)
