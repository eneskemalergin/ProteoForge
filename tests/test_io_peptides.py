"""Tests for peptide matrix I/O."""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from proteoforge import Config
from proteoforge.io import materialize_peptide_table, peptides_from_frame, read_peptides
from proteoforge.io._harmonize import select_representative_protein


def test_read_generic_parquet(fixtures_dir, minimal_config) -> None:
    frame = read_peptides(fixtures_dir / "minimal_long.parquet", minimal_config)
    assert set(frame.columns) >= {
        "protein_id",
        "peptide_id",
        "sample_id",
        "intensity",
    }
    assert frame.height == 32


def test_read_mapped_csv_via_column_map(fixtures_dir) -> None:
    config = Config.from_yaml_path(fixtures_dir / "minimal_config.yaml")
    column_map = replace(
        config.column_map,
        protein_id="prot",
        peptide_id="pep",
        sample_id="run",
        intensity="quant",
    )
    cfg = config.replace(column_map=column_map)
    frame = read_peptides(fixtures_dir / "minimal_mapped.csv", cfg)
    assert frame.get_column("protein_id").to_list()[0] == "P11111"


def test_select_representative_protein() -> None:
    assert select_representative_protein("P12345") == "P12345"
    assert select_representative_protein("A0A075B6K5;Q12345") == "Q12345"


def test_resolve_protein_ids_vectorized() -> None:
    from proteoforge.io._harmonize import resolve_protein_ids

    frame = pl.DataFrame(
        {
            "protein_id": ["P11111", "A0A075B6K5;Q12345", "P22222"],
            "peptide_id": ["A", "B", "C"],
        }
    )
    resolved = resolve_protein_ids(frame)
    assert resolved.get_column("protein_id").to_list() == ["P11111", "Q12345", "P22222"]


def test_column_map_override(minimal_config) -> None:
    frame = pl.DataFrame(
        {
            "prot": ["P1"] * 4,
            "pep": ["A", "A", "A", "A"],
            "run": ["S1", "S2", "S3", "S4"],
            "quant": [1.0, 2.0, 3.0, 4.0],
        }
    )
    column_map = replace(
        minimal_config.column_map,
        protein_id="prot",
        peptide_id="pep",
        sample_id="run",
        intensity="quant",
    )
    cfg = minimal_config.replace(column_map=column_map)
    with pytest.warns(
        DeprecationWarning,
        match="peptides_from_frame\\(\\) is deprecated",
    ):
        harmonized = peptides_from_frame(frame, cfg)
    assert "protein_id" in harmonized.columns


def test_materialize_peptide_table_alias(minimal_config) -> None:
    frame = pl.DataFrame(
        {
            "prot": ["P1"] * 4,
            "pep": ["A", "A", "A", "A"],
            "run": ["S1", "S2", "S3", "S4"],
            "quant": [1.0, 2.0, 3.0, 4.0],
        }
    )
    column_map = replace(
        minimal_config.column_map,
        protein_id="prot",
        peptide_id="pep",
        sample_id="run",
        intensity="quant",
    )
    cfg = minimal_config.replace(column_map=column_map)
    harmonized = materialize_peptide_table(frame, cfg)[0]
    assert "protein_id" in harmonized.columns
