"""Tests for :mod:`proteoforge.correction.qvalue`."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge.constants import QVALUE_LAMBDAS
from proteoforge.correction import p_adjust, p_adjust_by_group
from proteoforge.correction.qvalue import (
    adjust_qvalues,
    pi0_bootstrap,
    pi0_from_pvalues,
)

P = np.array([0.001, 0.008, 0.039, 0.041, 0.9])


def test_qvalue_monotone_and_bounded() -> None:
    q = p_adjust(P, "qvalue")
    assert np.all(q >= 0.0)
    assert np.all(q <= 1.0)


def test_qvalue_n_tests_override() -> None:
    p = np.array([0.01, 0.04, 0.2], dtype=np.float64)
    q = p_adjust(p, "qvalue", n_tests=100)
    assert q.shape == p.shape
    assert np.all(np.isfinite(q))


def test_qvalue_empty_returns_copy() -> None:
    empty = np.array([], dtype=np.float64)
    np.testing.assert_array_equal(p_adjust(empty, "qvalue"), empty)
    np.testing.assert_array_equal(adjust_qvalues(empty, n_tests=0), empty)


def _scipy_qvalue_oracle(p: np.ndarray) -> tuple[float, np.ndarray]:
    # Storey q-values with pi0 from SciPy's GCV smoothing spline evaluated at the
    # largest lambda; same algorithm as QuEStVar p_adjust(p, "qvalue")
    # (MIT, https://github.com/eneskemalergin/QuEStVar).
    from scipy.interpolate import make_smoothing_spline

    m = p.size
    counts = m - np.searchsorted(np.sort(p), QVALUE_LAMBDAS, side="right")
    pi0_lambda = counts / (m * (1.0 - QVALUE_LAMBDAS))
    pi0 = float(make_smoothing_spline(QVALUE_LAMBDAS, pi0_lambda)(QVALUE_LAMBDAS)[-1])
    pi0 = min(pi0, 1.0)
    order = np.argsort(p)[::-1]
    ranks = np.arange(m, 0, -1, dtype=np.float64)
    q_sorted = pi0 * np.minimum(1.0, np.minimum.accumulate(p[order] * m / ranks))
    q = np.empty_like(p)
    q[order] = q_sorted
    return pi0, q


def test_qvalue_matches_scipy_smoothing_spline_oracle() -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2026)
    # 70% nulls plus a signal component, so pi0 is estimated below 1 rather
    # than clipped, and the smoothing spline itself is exercised.
    p = np.concatenate([rng.uniform(0.0, 1.0, 1_400), rng.beta(0.2, 5.0, 600)])

    expected_pi0, expected_q = _scipy_qvalue_oracle(p)

    assert 0.5 < expected_pi0 < 1.0
    # The NumPy GCV spline solves a different linear system than SciPy's, so
    # agreement is about 1e-8 rather than float64 round-off.
    np.testing.assert_allclose(pi0_from_pvalues(p), expected_pi0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(p_adjust(p, "qvalue"), expected_q, rtol=0, atol=1e-7)


def test_pi0_from_pvalues_empty_returns_one() -> None:
    assert pi0_from_pvalues(np.array([], dtype=np.float64)) == 1.0


def test_pi0_bootstrap_fallback() -> None:
    p = np.array([1e-8, 1e-7, 1e-6, 1e-5], dtype=np.float64)
    pi0 = pi0_from_pvalues(p)
    assert np.isfinite(pi0)
    assert 0.0 < pi0 <= 1.0

    counts = np.ones_like(QVALUE_LAMBDAS)
    pi0_lambda = counts / (len(p) * (1.0 - QVALUE_LAMBDAS))
    boot = pi0_bootstrap(pi0_lambda, counts, QVALUE_LAMBDAS, len(p))
    assert 0.0 < boot <= 1.0


def test_qvalue_by_group() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    codes = np.array([0, 0, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "qvalue")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "qvalue")
    np.testing.assert_allclose(grouped, expected)
