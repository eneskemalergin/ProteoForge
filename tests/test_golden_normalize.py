"""Golden normalization tests against reference against_condition behavior."""

from __future__ import annotations

import numpy as np
import polars as pl

from proteoforge import Config, prepare
from proteoforge._normalize import normalize_control_relative
from proteoforge.schema import NORMALIZED_INTENSITY

# Golden float64 tolerances: rtol=1e-10, atol=1e-12
_GOLDEN_RTOL = 1e-10
_GOLDEN_ATOL = 1e-12


def _reference_against_condition_numpy(
    intensity: np.ndarray,
    control_indices: np.ndarray,
    *,
    input_is_log2: bool,
) -> np.ndarray:
    """Wide NumPy normalize path (pandas ``std`` ddof=1 for z-scoring)."""
    return normalize_control_relative(
        intensity,
        control_column_indices=control_indices,
        input_is_log2=input_is_log2,
    )


def _wide_from_long(
    long_frame: pl.DataFrame,
    sample_ids: tuple[str, ...],
) -> np.ndarray:
    wide = (
        long_frame.select(
            ["protein_id", "peptide_id", "sample_id", NORMALIZED_INTENSITY]
        )
        .pivot(
            on="sample_id",
            index=["protein_id", "peptide_id"],
            values=NORMALIZED_INTENSITY,
        )
        .sort(["protein_id", "peptide_id"])
    )
    return wide.select(sample_ids).to_numpy().astype(np.float64, copy=False)


def test_golden_matches_reference_algorithm(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    dataset = prepare(minimal_peptides_frame, minimal_config)

    sample_index = {sample: idx for idx, sample in enumerate(dataset.sample_ids)}
    control_indices = np.array(
        [sample_index[s] for s in ("S1", "S2")],
        dtype=np.intp,
    )

    raw = minimal_peptides_frame.pivot(
        on="sample_id",
        index=["protein_id", "peptide_id"],
        values="intensity",
    )
    sample_cols = [s for s in dataset.sample_ids if s in raw.columns]
    raw = raw.sort(["protein_id", "peptide_id"])
    intensity = raw.select(sample_cols).to_numpy()

    expected = _reference_against_condition_numpy(
        intensity,
        control_indices,
        input_is_log2=False,
    )

    actual = _wide_from_long(dataset.peptides, dataset.sample_ids)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_GOLDEN_RTOL,
        atol=_GOLDEN_ATOL,
    )


def test_golden_from_committed_fixtures(fixtures_dir) -> None:
    from proteoforge.io import read_peptides

    config = Config.from_yaml_path(fixtures_dir / "minimal_config.yaml")
    peptides = read_peptides(fixtures_dir / "minimal_long.parquet", config)
    dataset = prepare(peptides, config)

    assert not np.isnan(dataset.intensity_normalized).any()
    control = dataset.peptides.filter(pl.col("sample_id").is_in(["S1", "S2"]))
    control_means = (
        control.group_by(["protein_id", "peptide_id"])
        .agg(pl.col(NORMALIZED_INTENSITY).mean())
        .get_column(NORMALIZED_INTENSITY)
        .to_numpy()
    )
    np.testing.assert_allclose(control_means, 0.0, atol=1e-10)
