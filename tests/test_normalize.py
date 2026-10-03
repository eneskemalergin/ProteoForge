"""Tests for normalization."""

from __future__ import annotations

import numpy as np
import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._normalize import (
    normalize_control_relative,
    normalize_control_relative_long,
)
from proteoforge.schema import NORMALIZED_INTENSITY


def test_normalize_rejects_non_positive_when_log_required() -> None:
    intensity = np.array([[0.0, 1.0]], dtype=np.float64)
    control_idx = np.array([0], dtype=np.intp)
    with pytest.raises(ProteoForgeValidationError, match="Non-positive"):
        normalize_control_relative(
            intensity,
            control_column_indices=control_idx,
            input_is_log2=False,
        )


def test_normalize_rejects_zero_std_column() -> None:
    intensity = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64)
    control_idx = np.array([0], dtype=np.intp)
    with pytest.raises(ProteoForgeValidationError, match="zero standard deviation"):
        normalize_control_relative(
            intensity,
            control_column_indices=control_idx,
            input_is_log2=True,
        )


def test_normalize_control_subtraction() -> None:
    rng = np.random.default_rng(0)
    intensity = rng.uniform(100, 200, size=(6, 4))
    control_idx = np.array([0, 1], dtype=np.intp)
    normalized = normalize_control_relative(
        intensity,
        control_column_indices=control_idx,
        input_is_log2=False,
    )
    control_mean = normalized[:, control_idx].mean(axis=1)
    np.testing.assert_allclose(control_mean, 0.0, atol=1e-12)


def test_long_normalize_matches_wide(minimal_peptides_frame, minimal_config) -> None:
    from proteoforge.io._design import attach_conditions

    frame = attach_conditions(minimal_peptides_frame, minimal_config.to_design_table())
    control_samples = minimal_config.to_design_table().condition_to_samples["control"]

    long_norm = normalize_control_relative_long(
        frame,
        control_sample_ids=control_samples,
        input_is_log2=False,
    )

    wide = frame.pivot(
        on="sample_id", index=["protein_id", "peptide_id"], values="intensity"
    ).sort(["protein_id", "peptide_id"])
    sample_cols = ["S1", "S2", "S3", "S4"]
    intensity = wide.select(sample_cols).to_numpy()
    wide_norm = normalize_control_relative(
        intensity,
        control_column_indices=np.array([0, 1], dtype=np.intp),
        input_is_log2=False,
    )

    long_as_wide = (
        long_norm.select(
            ["protein_id", "peptide_id", "sample_id", NORMALIZED_INTENSITY]
        )
        .pivot(
            on="sample_id",
            index=["protein_id", "peptide_id"],
            values=NORMALIZED_INTENSITY,
        )
        .sort(["protein_id", "peptide_id"])
        .select(sample_cols)
        .to_numpy()
    )
    np.testing.assert_allclose(long_as_wide, wide_norm, rtol=1e-10, atol=1e-11)


def test_long_normalize_preserves_row_order(
    minimal_peptides_frame, minimal_config
) -> None:
    from proteoforge.io._design import attach_conditions

    frame = attach_conditions(minimal_peptides_frame, minimal_config.to_design_table())
    control_samples = minimal_config.to_design_table().condition_to_samples["control"]
    keys = frame.select(["protein_id", "peptide_id", "sample_id"])

    long_norm = normalize_control_relative_long(
        frame,
        control_sample_ids=control_samples,
        input_is_log2=False,
    )

    assert keys.equals(long_norm.select(["protein_id", "peptide_id", "sample_id"]))


def test_long_normalize_is_bit_identical_across_runs() -> None:
    """Many peptide groups: the control mean must not depend on thread scheduling."""
    import polars as pl

    from proteoforge._normalize import normalize_control_relative_long

    rng = np.random.default_rng(7)
    samples = [f"S{i}" for i in range(10)]
    n_peptides = 30_000
    frame = pl.DataFrame(
        {
            "protein_id": np.repeat(
                [f"P{i // 6}" for i in range(n_peptides)], len(samples)
            ),
            "peptide_id": np.repeat([f"p{i}" for i in range(n_peptides)], len(samples)),
            "sample_id": np.tile(samples, n_peptides),
            "intensity": rng.lognormal(20.0, 1.5, size=n_peptides * len(samples)),
        }
    )
    runs = [
        normalize_control_relative_long(
            frame, control_sample_ids=tuple(samples[:5]), input_is_log2=False
        )[NORMALIZED_INTENSITY].to_numpy()
        for _ in range(4)
    ]
    for other in runs[1:]:
        np.testing.assert_array_equal(other, runs[0])
