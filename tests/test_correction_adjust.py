"""Tests for classical :mod:`proteoforge.correction` adjusters."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.correction import VALID_METHODS, p_adjust, p_adjust_by_group

P = np.array([0.001, 0.008, 0.039, 0.041, 0.9])


def test_bonferroni() -> None:
    np.testing.assert_allclose(p_adjust(P, "bonferroni"), np.minimum(P * 5, 1.0))


def test_none_is_identity() -> None:
    np.testing.assert_array_equal(p_adjust(P, "none"), P)
    np.testing.assert_array_equal(p_adjust(P, None), P)


def test_bh_matches_known_vector() -> None:
    # R p.adjust(c(0.001, 0.008, 0.039, 0.041, 0.9), "BH")
    expected = np.array([0.005, 0.02, 0.05125, 0.05125, 0.9])
    np.testing.assert_allclose(p_adjust(P, "fdr_bh"), expected, rtol=1e-6)


def test_hochberg_matches_known_vector() -> None:
    # R p.adjust(c(0.001, 0.008, 0.039, 0.041, 0.9), "hochberg")
    expected = np.array([0.005, 0.032, 0.082, 0.082, 0.9])
    np.testing.assert_allclose(p_adjust(P, "hochberg"), expected, rtol=1e-6)


def test_by_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    expected = multipletests(P, method="fdr_by")[1]
    np.testing.assert_allclose(p_adjust(P, "BY"), expected, rtol=1e-6)


def test_p_adjust_by_group_holm() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "holm")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "holm")
    np.testing.assert_allclose(grouped, expected)


def test_p_adjust_by_group_hommel() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "hommel")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "hommel")
    np.testing.assert_allclose(grouped, expected)


def test_holm_matches_known_vector() -> None:
    # R p.adjust(c(0.001, 0.008, 0.039, 0.041, 0.9), "holm")
    expected = np.array([0.005, 0.032, 0.117, 0.117, 0.9])
    np.testing.assert_allclose(p_adjust(P, "holm"), expected, rtol=1e-6)


def test_hommel_matches_known_vector() -> None:
    # R p.adjust(c(0.001,0.008,0.039,0.041,0.9), "hommel")
    expected = np.array([0.005, 0.032, 0.078, 0.082, 0.9])
    np.testing.assert_allclose(p_adjust(P, "hommel"), expected, rtol=1e-6)


def test_hommel_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    expected = multipletests(P, method="hommel")[1]
    np.testing.assert_allclose(p_adjust(P, "hommel"), expected, rtol=1e-10, atol=1e-12)


def test_hommel_n_tests_override() -> None:
    # R p.adjust(P, "hommel", n=10)
    expected = np.array([0.01, 0.072, 0.273, 0.287, 1.0])
    np.testing.assert_allclose(p_adjust(P, "hommel", n_tests=10), expected, rtol=1e-6)


def test_hommel_in_valid_methods() -> None:
    assert "hommel" in VALID_METHODS


def test_hommel_large_n() -> None:
    # statsmodels oracle, n=600
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 600)
    expected = multipletests(p, method="hommel")[1]
    np.testing.assert_allclose(p_adjust(p, "hommel"), expected, rtol=1e-10, atol=1e-12)


def test_hommel_ties_match_statsmodels() -> None:
    # statsmodels oracle, tied p-values
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.001, 0.01, 0.05, 0.05, 0.2, 1.0], dtype=np.float64)
    expected = multipletests(p, method="hommel")[1]
    np.testing.assert_allclose(p_adjust(p, "hommel"), expected, rtol=1e-10, atol=1e-12)


def test_hommel_medium_n_sorted_uniform() -> None:
    # statsmodels oracle, n=5000, ascending uniform
    from statsmodels.stats.multitest import multipletests

    p = np.sort(np.random.default_rng(7).uniform(0.0, 1.0, 5000))
    expected = multipletests(p, method="hommel")[1]
    np.testing.assert_allclose(p_adjust(p, "hommel"), expected, rtol=1e-10, atol=1e-12)


def test_hommel_empty_returns_copy() -> None:
    empty = np.array([], dtype=np.float64)
    np.testing.assert_array_equal(p_adjust(empty, "hommel"), empty)


def test_hommel_singleton() -> None:
    p = np.array([0.04], dtype=np.float64)
    np.testing.assert_allclose(p_adjust(p, "hommel"), p)


def test_p_adjust_rejects_unknown_method() -> None:
    with pytest.raises(ProteoForgeValidationError, match="Unknown correction method"):
        p_adjust(P, "not_a_method")


def test_p_adjust_by_group_rejects_unknown_method() -> None:
    p = np.array([0.01, 0.02], dtype=np.float64)
    codes = np.array([0, 1], dtype=np.intp)
    with pytest.raises(ProteoForgeValidationError, match="Unknown correction method"):
        p_adjust_by_group(p, codes, "not_a_method")


def test_p_adjust_non_positive_n_tests_returns_copy() -> None:
    p = np.array([0.01, 0.02], dtype=np.float64)
    np.testing.assert_array_equal(p_adjust(p, "holm", n_tests=0), p)
    np.testing.assert_array_equal(p_adjust(p, "holm", n_tests=-1), p)


def test_p_adjust_by_group_preserves_nan() -> None:
    p = np.array([0.01, np.nan, 0.03, 0.04], dtype=np.float64)
    codes = np.array([0, 0, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "hommel")
    assert np.isnan(grouped[1])
    finite = np.isfinite(p)
    np.testing.assert_allclose(
        grouped[finite],
        p_adjust_by_group(p[finite], codes[finite], "hommel"),
    )


def test_p_adjust_by_group_all_nan() -> None:
    p = np.array([np.nan, np.nan], dtype=np.float64)
    codes = np.array([0, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "holm")
    assert np.all(np.isnan(grouped))


def test_n_tests_override() -> None:
    adjusted = p_adjust(np.array([0.01]), "bonferroni", n_tests=10)
    assert adjusted[0] == pytest.approx(0.1)


def test_p_adjust_by_group_bonferroni() -> None:
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float64)
    codes = np.array([0, 0, 1, 1, 1], dtype=np.intp)
    grouped = p_adjust_by_group(p, codes, "bonferroni")
    expected = np.empty_like(p)
    for code in (0, 1):
        mask = codes == code
        expected[mask] = p_adjust(p[mask], "bonferroni")
    np.testing.assert_allclose(grouped, expected)


def test_qvalue_in_valid_methods() -> None:
    assert "qvalue" in VALID_METHODS


def test_top_level_api_exports_p_adjust() -> None:
    import proteoforge as pf

    assert pf.p_adjust is p_adjust
    assert pf.p_adjust_by_group is p_adjust_by_group
    assert pf.VALID_METHODS is VALID_METHODS


def test_legacy_correction_shim() -> None:
    from proteoforge._correction import p_adjust as shim_adjust

    np.testing.assert_array_equal(shim_adjust(P, "none"), P)
