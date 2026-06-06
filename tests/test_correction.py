"""Tests for the QuEStVar p_adjust port."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._correction import VALID_METHODS, p_adjust

P = np.array([0.001, 0.008, 0.039, 0.041, 0.9])


def test_bonferroni() -> None:
    np.testing.assert_allclose(p_adjust(P, "bonferroni"), np.minimum(P * 5, 1.0))


def test_none_is_identity() -> None:
    np.testing.assert_array_equal(p_adjust(P, "none"), P)
    np.testing.assert_array_equal(p_adjust(P, None), P)


def test_bh_matches_known_vector() -> None:
    # R p.adjust(c(0.001,0.008,0.039,0.041,0.9), "BH")
    expected = np.array([0.005, 0.02, 0.05125, 0.05125, 0.9])
    np.testing.assert_allclose(p_adjust(P, "fdr_bh"), expected, rtol=1e-6)


def test_hochberg_matches_known_vector() -> None:
    expected = np.array([0.005, 0.032, 0.082, 0.082, 0.9])
    np.testing.assert_allclose(p_adjust(P, "hochberg"), expected, rtol=1e-6)


def test_by_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    expected = multipletests(P, method="fdr_by")[1]
    np.testing.assert_allclose(p_adjust(P, "BY"), expected, rtol=1e-6)


def test_p_adjust_by_group_holm() -> None:
    from proteoforge._correction import p_adjust_by_group

    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "holm")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "holm")
    np.testing.assert_allclose(grouped, expected)


def test_holm_matches_known_vector() -> None:
    # R p.adjust(c(0.001,0.008,0.039,0.041,0.9), "holm")
    expected = np.array([0.005, 0.032, 0.117, 0.117, 0.9])
    np.testing.assert_allclose(p_adjust(P, "holm"), expected, rtol=1e-6)


def test_n_tests_override() -> None:
    adjusted = p_adjust(np.array([0.01]), "bonferroni", n_tests=10)
    assert adjusted[0] == pytest.approx(0.1)


def test_p_adjust_by_group_bonferroni() -> None:
    from proteoforge._correction import p_adjust_by_group

    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "bonferroni")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "bonferroni")
    np.testing.assert_allclose(grouped, expected)


def test_qvalue_is_disabled() -> None:
    assert "qvalue" not in VALID_METHODS
    with pytest.raises(ValueError, match="Unknown correction method"):
        p_adjust(P, "qvalue")
