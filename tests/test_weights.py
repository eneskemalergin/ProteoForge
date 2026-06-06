"""Tests for mask-derived WLS weights."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from proteoforge import Config
from proteoforge._weights import imputation_weights, row_weights
from proteoforge.schema import IS_COMPLETE_MISSING, IS_REAL, WEIGHT


def test_three_tier_weights() -> None:
    is_real = np.array([True, False, False])
    is_complete_missing = np.array([False, True, False])
    weights = imputation_weights(is_real, is_complete_missing, biological_weight=0.5)
    np.testing.assert_array_equal(weights, [1.0, 0.5, 1e-5])


def test_invalid_weight_order() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        imputation_weights(
            np.array([True]),
            np.array([False]),
            biological_weight=2.0,
        )


def test_row_weights_none_for_rlm() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        model="rlm",
    )
    frame = pl.DataFrame({IS_REAL: [True], IS_COMPLETE_MISSING: [False]})
    assert row_weights(frame, config) is None


def test_row_weights_prefers_weight_column() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        model="wls",
    )
    frame = pl.DataFrame({WEIGHT: [0.3, 0.7]})
    np.testing.assert_array_equal(row_weights(frame, config), [0.3, 0.7])


def test_row_weights_from_masks() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        model="wls",
        wls_biological_weight=0.25,
    )
    frame = pl.DataFrame({IS_REAL: [True, False], IS_COMPLETE_MISSING: [False, True]})
    np.testing.assert_array_equal(row_weights(frame, config), [1.0, 0.25])
