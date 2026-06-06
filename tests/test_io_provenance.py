"""Tests for provenance I/O."""

from __future__ import annotations

import polars as pl
import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.io._provenance import attach_provenance, provenance_from_frame


def test_provenance_from_frame() -> None:
    frame = pl.DataFrame(
        {
            "protein_id": ["P1"],
            "peptide_id": ["A"],
            "sample_id": ["S1"],
            "is_real": [True],
        }
    )
    result = provenance_from_frame(frame)
    assert result.get_column("is_real").to_list() == [True]


def test_read_provenance_parquet(tmp_path) -> None:
    from proteoforge.io import read_provenance

    path = tmp_path / "prov.parquet"
    pl.DataFrame(
        {
            "protein_id": ["P1"],
            "peptide_id": ["A"],
            "sample_id": ["S1"],
            "is_real": [True],
            "is_complete_missing": [False],
            "weight": [1.0],
        }
    ).write_parquet(path)
    frame = read_provenance(path)
    assert frame.height == 1
    assert frame.get_column("weight").item() == 1.0


def test_read_provenance_missing_file(tmp_path) -> None:
    from proteoforge._exceptions import ProteoForgeIOError
    from proteoforge.io import read_provenance

    with pytest.raises(ProteoForgeIOError, match="not found"):
        read_provenance(tmp_path / "missing.parquet")


def test_attach_provenance_incomplete_coverage(minimal_peptides_frame) -> None:
    peptides = minimal_peptides_frame.head(8)
    provenance = (
        peptides.head(4)
        .select(
            "protein_id",
            "peptide_id",
            "sample_id",
        )
        .with_columns(
            pl.lit(True).alias("is_real"),
            pl.lit(False).alias("is_complete_missing"),
        )
    )
    with pytest.raises(ProteoForgeValidationError, match="without matching provenance"):
        attach_provenance(peptides, provenance)


def test_attach_provenance_mismatch(minimal_peptides_frame) -> None:
    peptides = minimal_peptides_frame.head(4)
    provenance = pl.DataFrame(
        {
            "protein_id": ["P99999"],
            "peptide_id": ["PEP1"],
            "sample_id": ["S1"],
            "is_real": [True],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="not present in peptide"):
        attach_provenance(peptides, provenance)
