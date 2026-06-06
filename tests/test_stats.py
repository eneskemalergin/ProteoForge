"""Tests for the NumPy-only statistics core against scipy oracles."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._stats import betainc, gammaincc, sf_chi2, sf_f, wald_pvalue


def test_betainc_matches_scipy() -> None:
    special = pytest.importorskip("scipy.special")
    x = np.linspace(0.0, 1.0, 101)
    for a, b in [(0.5, 0.5), (2.0, 3.0), (7.0, 1.0), (1.0, 20.0)]:
        expected = special.betainc(a, b, x)
        np.testing.assert_allclose(betainc(a, b, x), expected, rtol=1e-12, atol=1e-14)


def test_gammaincc_matches_scipy() -> None:
    special = pytest.importorskip("scipy.special")
    x = np.linspace(0.0, 40.0, 201)
    for a in [0.5, 1.0, 3.0, 10.0]:
        expected = special.gammaincc(a, x)
        np.testing.assert_allclose(gammaincc(a, x), expected, rtol=1e-12, atol=1e-14)


def test_sf_f_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    x = np.linspace(0.01, 30.0, 200)
    for dfn, dfd in [(1.0, 10.0), (3.0, 25.0), (5.0, 100.0)]:
        expected = stats.f.sf(x, dfn, dfd)
        np.testing.assert_allclose(sf_f(x, dfn, dfd), expected, rtol=1e-10, atol=1e-14)


def test_sf_chi2_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    x = np.linspace(0.01, 50.0, 200)
    for dof in [1.0, 2.0, 5.0, 12.0]:
        expected = stats.chi2.sf(x, dof)
        np.testing.assert_allclose(sf_chi2(x, dof), expected, rtol=1e-10, atol=1e-14)


def test_sf_tails_are_bounded() -> None:
    assert sf_f(np.array([0.0]), 2.0, 10.0)[0] == 1.0
    assert sf_chi2(np.array([0.0]), 3.0)[0] == 1.0
    assert 0.0 <= sf_f(np.array([1e6]), 2.0, 10.0)[0] <= 1e-6


def test_wald_pvalue_degenerate_block() -> None:
    coef = np.array([1.0, 2.0])
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.isnan(wald_pvalue(coef, singular, use_f=True, df_resid=10.0))


def test_wald_pvalue_single_df_matches_f() -> None:
    coef = np.array([2.0])
    cov = np.array([[0.25]])
    # stat = 4 / 0.25 = 16, F = 16 on (1, 20)
    value = wald_pvalue(coef, cov, use_f=True, df_resid=20.0)
    expected = float(sf_f(np.array([16.0]), 1.0, 20.0)[0])
    assert value == pytest.approx(expected)
