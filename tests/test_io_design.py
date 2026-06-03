"""Tests for design file I/O."""

from __future__ import annotations

import polars as pl
import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.io import design_from_frame, read_design


def test_read_design_from_fixture(fixtures_dir) -> None:
    with pytest.warns(DeprecationWarning, match="read_design\\(\\) is deprecated"):
        design = read_design(fixtures_dir / "minimal_design.csv")
    assert design.sample_ids == ("S1", "S2", "S3", "S4")
    assert design.condition_to_samples["control"] == ("S1", "S2")


def test_design_rejects_duplicate_samples() -> None:
    frame = pl.DataFrame(
        {"sample_id": ["S1", "S1"], "condition": ["control", "treated"]}
    )
    with pytest.raises(ProteoForgeValidationError, match="Duplicate sample_id"):
        design_from_frame(frame)


def test_design_rejects_missing_columns() -> None:
    frame = pl.DataFrame({"sample_id": ["S1"]})
    with pytest.raises(ProteoForgeValidationError, match="missing columns"):
        design_from_frame(frame)
