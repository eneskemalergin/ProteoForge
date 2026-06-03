"""Tests for validation and prepare pipeline."""

from __future__ import annotations

import polars as pl
import pytest

from proteoforge import Config, prepare
from proteoforge._exceptions import ProteoForgeValidationError


def test_prepare_happy_path(minimal_peptides_frame, minimal_config) -> None:
    dataset = prepare(minimal_peptides_frame, minimal_config)
    assert dataset.n_peptides == 8
    assert dataset.n_samples == 4
    assert dataset.n_proteins == 2
    assert dataset.condition_levels[0] == "control"
    assert dataset.peptides.height == 32
    assert "intensity_normalized" in dataset.peptides.columns


def test_config_rejects_control_not_in_conditions() -> None:
    with pytest.raises(ProteoForgeValidationError, match="not a key in conditions"):
        Config(
            control_condition="missing",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        )


def test_prepare_rejects_too_few_peptides(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    small = minimal_peptides_frame.filter(pl.col("peptide_id") == "PEP1")
    config = minimal_config.replace(min_peptides=4)
    with pytest.raises(ProteoForgeValidationError, match="fewer than"):
        prepare(small, config)


def test_prepare_rejects_duplicate_keys() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    frame = pl.DataFrame(
        {
            "protein_id": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "peptide_id": ["A", "A", "B", "B", "B", "B"],
            "sample_id": ["S1", "S1", "S1", "S2", "S3", "S4"],
            "intensity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="Duplicate peptide keys"):
        prepare(frame, config)


def test_prepare_filters_extra_samples(minimal_peptides_frame) -> None:
    extra = minimal_peptides_frame.filter(pl.col("sample_id") == "S1").with_columns(
        pl.lit("S9").alias("sample_id")
    )
    frame = pl.concat([minimal_peptides_frame, extra])
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    with pytest.warns(UserWarning, match="Dropping 1 sample"):
        dataset = prepare(frame, config)
    assert dataset.n_samples == 4
    assert dataset.metadata["samples_dropped"] == ("S9",)


def test_prepare_from_files(fixtures_dir) -> None:
    from proteoforge import Config, prepare_from_parquet

    config = Config.from_yaml_path(fixtures_dir / "minimal_config.yaml")
    dataset = prepare_from_parquet(fixtures_dir / "minimal_long.parquet", config)
    assert dataset.peptides.height == 32
    assert dataset.intensity_normalized.shape == (32,)


def test_prepare_read_peptides_then_prepare(fixtures_dir) -> None:
    from proteoforge import Config
    from proteoforge.io import read_peptides

    config = Config.from_yaml_path(fixtures_dir / "minimal_config.yaml")
    peptides = read_peptides(fixtures_dir / "minimal_long.parquet", config)
    dataset = prepare(peptides, config)
    assert dataset.peptides.height == 32
    assert dataset.intensity_normalized.shape == (32,)


def test_prepare_wls_requires_provenance(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    config = minimal_config.replace(model="wls")
    with pytest.raises(
        ProteoForgeValidationError,
        match="model='wls' requires provenance",
    ):
        prepare(minimal_peptides_frame, config)


def test_prepare_with_provenance_columns(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    frame = minimal_peptides_frame.with_columns(
        pl.lit(True).alias("is_real"),
        pl.lit(False).alias("is_complete_missing"),
        pl.lit(1.0).alias("weight"),
    )
    config = minimal_config.replace(model="wls")
    dataset = prepare(frame, config)
    assert dataset.is_real is not None
    assert dataset.is_complete_missing is not None
    assert dataset.weight is not None
    assert dataset.is_real.shape == (dataset.peptides.height,)


def test_prepare_skips_provenance_for_rlm(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    frame = minimal_peptides_frame.with_columns(
        pl.lit(True).alias("is_real"),
        pl.lit(False).alias("is_complete_missing"),
    )
    dataset = prepare(frame, minimal_config)
    assert dataset.is_real is None
    assert dataset.is_complete_missing is None


def test_prepare_lazy_frame_matches_eager(
    minimal_peptides_frame,
    minimal_config,
) -> None:
    eager = prepare(minimal_peptides_frame, minimal_config)
    lazy = prepare(minimal_peptides_frame.lazy(), minimal_config)
    assert eager.peptides.equals(lazy.peptides)


def test_prepare_from_parquet(fixtures_dir) -> None:
    from proteoforge import prepare_from_parquet

    config = Config.from_yaml_path(fixtures_dir / "minimal_config.yaml")
    dataset = prepare_from_parquet(fixtures_dir / "minimal_long.parquet", config)
    assert dataset.peptides.height == 32
    assert dataset.intensity_normalized.shape == (32,)
