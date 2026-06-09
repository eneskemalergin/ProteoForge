"""Tests for correction-related shipped constants."""

from __future__ import annotations

import pytest

from proteoforge.constants import (
    PI0_GCV_SPLINE_ASSET,
    QVALUE_LAMBDAS,
    load_pi0_gcv_spline_matrices,
)


def test_qvalue_lambda_grid() -> None:
    assert QVALUE_LAMBDAS.shape == (19,)
    assert QVALUE_LAMBDAS[0] == 0.05
    assert QVALUE_LAMBDAS[-1] == pytest.approx(0.95)


def test_pi0_gcv_spline_matrices_load() -> None:
    x_banded, w_e = load_pi0_gcv_spline_matrices()
    assert x_banded.shape == (5, len(QVALUE_LAMBDAS))
    assert w_e.shape == (5, len(QVALUE_LAMBDAS))
    assert PI0_GCV_SPLINE_ASSET.endswith(".npz")
