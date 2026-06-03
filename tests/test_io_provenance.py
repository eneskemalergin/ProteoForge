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
