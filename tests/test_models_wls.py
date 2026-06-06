"""Tests for the WLS one-vs-rest backend, including statsmodels parity."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._layout import ProteinBlock
from proteoforge.models._protocol import build_design_stack
from proteoforge.models._wls import WLSModel


def _block(
    condition_code: np.ndarray,
    peptide_code: np.ndarray,
    response: np.ndarray,
    *,
    n_conditions: int,
    weight: np.ndarray | None = None,
) -> ProteinBlock:
    peptide_ids = tuple(f"PEP{i}" for i in range(int(peptide_code.max()) + 1))
    return ProteinBlock(
        protein_id="P1",
        peptide_ids=peptide_ids,
        response=response.astype(np.float64),
        condition_code=condition_code.astype(np.intp),
        peptide_code=peptide_code.astype(np.intp),
        weight=None if weight is None else weight.astype(np.float64),
        n_conditions=n_conditions,
    )


def _statsmodels_pvalue(
    y: np.ndarray,
    condition_code: np.ndarray,
    peptide_code: np.ndarray,
    target: int,
    weights: np.ndarray | None,
) -> float:
    pd = pytest.importorskip("pandas")
    smf = pytest.importorskip("statsmodels.formula.api")
    frame = pd.DataFrame(
        {
            "y": y,
            "cond": [f"c{c}" for c in condition_code],
            "allothers": np.where(peptide_code == target, "target", "rest"),
        }
    )
    if weights is None:
        fitted = smf.ols("y ~ C(cond) * C(allothers)", frame).fit()
    else:
        fitted = smf.wls("y ~ C(cond) * C(allothers)", frame, weights=weights).fit()
    return float(fitted.wald_test_terms(scalar=False).pvalues[-1])


@pytest.mark.parametrize("n_conditions", [2, 3])
def test_ols_matches_statsmodels(n_conditions: int) -> None:
    pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(0)
    reps = 4
    k = 3
    condition_code = np.tile(np.repeat(np.arange(n_conditions), reps), k)
    peptide_code = np.repeat(np.arange(k), n_conditions * reps)
    y = rng.normal(size=condition_code.size) + 0.4 * peptide_code

    block = _block(condition_code, peptide_code, y, n_conditions=n_conditions)
    design = build_design_stack(block)
    response = np.broadcast_to(y, (k, y.size))
    p_ours = WLSModel().fit_pvalues(
        design, response, None, n_interaction=n_conditions - 1
    )
    for target in range(k):
        expected = _statsmodels_pvalue(y, condition_code, peptide_code, target, None)
        assert p_ours[target] == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_wls_matches_statsmodels_with_weights() -> None:
    pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(7)
    reps, k, n_conditions = 4, 4, 2
    condition_code = np.tile(np.repeat(np.arange(n_conditions), reps), k)
    peptide_code = np.repeat(np.arange(k), n_conditions * reps)
    y = rng.normal(size=condition_code.size)
    weights = rng.uniform(0.2, 1.0, size=condition_code.size)

    block = _block(
        condition_code, peptide_code, y, n_conditions=n_conditions, weight=weights
    )
    design = build_design_stack(block)
    response = np.broadcast_to(y, (k, y.size))
    weight_stack = np.broadcast_to(weights, (k, y.size))
    p_ours = WLSModel().fit_pvalues(design, response, weight_stack, n_interaction=1)
    for target in range(k):
        expected = _statsmodels_pvalue(y, condition_code, peptide_code, target, weights)
        assert p_ours[target] == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_interaction_invariant_to_nonreference_order() -> None:
    rng = np.random.default_rng(3)
    reps, k, n_conditions = 3, 3, 3
    condition_code = np.tile(np.repeat(np.arange(n_conditions), reps), k)
    peptide_code = np.repeat(np.arange(k), n_conditions * reps)
    y = rng.normal(size=condition_code.size)

    block = _block(condition_code, peptide_code, y, n_conditions=n_conditions)
    base = WLSModel().fit_pvalues(
        build_design_stack(block),
        np.broadcast_to(y, (k, y.size)),
        None,
        n_interaction=n_conditions - 1,
    )

    swapped_code = condition_code.copy()
    swapped_code[condition_code == 1] = 2
    swapped_code[condition_code == 2] = 1
    block_swapped = _block(swapped_code, peptide_code, y, n_conditions=n_conditions)
    swapped = WLSModel().fit_pvalues(
        build_design_stack(block_swapped),
        np.broadcast_to(y, (k, y.size)),
        None,
        n_interaction=n_conditions - 1,
    )
    np.testing.assert_allclose(base, swapped, rtol=1e-10)


def test_underdetermined_returns_nan() -> None:
    condition_code = np.array([0, 1, 0, 1])
    peptide_code = np.array([0, 0, 1, 1])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    block = _block(condition_code, peptide_code, y, n_conditions=2)
    design = build_design_stack(block)
    result = WLSModel().fit_pvalues(
        design, np.broadcast_to(y, (2, 4)), None, n_interaction=1
    )
    assert np.all(np.isnan(result))


def test_rank_deficient_design_returns_nan() -> None:
    # All observations in one condition: condition dummy column is all zero.
    condition_code = np.zeros(12, dtype=np.intp)
    peptide_code = np.repeat(np.arange(3), 4)
    y = np.arange(12, dtype=np.float64)
    block = _block(condition_code, peptide_code, y, n_conditions=2)
    design = build_design_stack(block)
    result = WLSModel().fit_pvalues(
        design, np.broadcast_to(y, (3, 12)), None, n_interaction=1
    )
    assert np.all(np.isnan(result))
