"""Tests for per-peptide fit status codes."""

from __future__ import annotations

import numpy as np

from proteoforge.models._fit_status import (
    FIT_STATUS_ILL_CONDITIONED,
    FIT_STATUS_OK,
    FIT_STATUS_RANK_DEFICIENT,
    count_statuses,
    empty_status,
)
from proteoforge.models._wls import WLSModel


def test_wls_status_ok_on_full_rank_design() -> None:
    rng = np.random.default_rng(3)
    m, n_obs, p = 4, 24, 6
    design = rng.normal(size=(m, n_obs, p))
    design[:, :, 0] = 1.0
    response = rng.normal(size=(m, n_obs))
    pvalues, status = WLSModel().fit_pvalues_and_status(
        design, response, None, n_interaction=2
    )
    assert np.all(status == FIT_STATUS_OK)
    assert np.all(np.isfinite(pvalues))


def test_wls_status_ill_conditioned_on_singular_design() -> None:
    design = np.ones((2, 4, 3), dtype=np.float64)
    response = np.ones((2, 4), dtype=np.float64)
    _, status = WLSModel().fit_pvalues_and_status(
        design, response, None, n_interaction=1
    )
    assert np.all(status == FIT_STATUS_ILL_CONDITIONED)


def test_count_statuses_aggregates() -> None:
    status = empty_status(3)
    status[0] = FIT_STATUS_OK
    counts = count_statuses(status)
    assert counts[FIT_STATUS_OK] == 1
    assert counts[FIT_STATUS_RANK_DEFICIENT] == 2
