"""Tests for per-protein long block construction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from proteoforge import Config, prepare
from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._layout import build_protein_blocks
from proteoforge._normalize import NORMALIZED_INTENSITY
from proteoforge.schema import CONDITION, PEPTIDE_ID, PROTEIN_ID, SAMPLE_ID
from proteoforge.types import PreparedDataset


def test_blocks_shapes_and_codes(minimal_peptides_frame, minimal_config) -> None:
    prepared = prepare(minimal_peptides_frame, minimal_config)
    blocks = build_protein_blocks(prepared)

    assert len(blocks) == 2
    block = blocks[0]
    assert block.n_conditions == 2
    assert block.n_peptides == 4
    assert block.n_obs == 16
    assert block.peptide_ids == ("PEP1", "PEP2", "PEP3", "PEP4")
    assert set(np.unique(block.condition_code).tolist()) == {0, 1}
    assert set(np.unique(block.peptide_code).tolist()) == {0, 1, 2, 3}
    assert np.all(np.isfinite(block.response))
    assert block.weight is None


def test_blocks_protein_order(minimal_peptides_frame, minimal_config) -> None:
    prepared = prepare(minimal_peptides_frame, minimal_config)
    blocks = build_protein_blocks(prepared)
    assert [b.protein_id for b in blocks] == ["P11111", "P22222"]


def test_non_finite_response_rejected() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        min_peptides=2,
    )
    rows = []
    for peptide in ("PEP1", "PEP2"):
        for sample, condition in [
            ("S1", "control"),
            ("S2", "control"),
            ("S3", "treated"),
            ("S4", "treated"),
        ]:
            rows.append(
                {
                    PROTEIN_ID: "P1",
                    PEPTIDE_ID: peptide,
                    SAMPLE_ID: sample,
                    CONDITION: condition,
                    NORMALIZED_INTENSITY: 0.5,
                }
            )
    frame = pl.DataFrame(rows)
    frame[0, NORMALIZED_INTENSITY] = float("nan")
    prepared = PreparedDataset(
        config=config,
        peptides=frame,
        sample_ids=("S1", "S2", "S3", "S4"),
        condition_levels=("control", "treated"),
        protein_index=np.zeros(frame.height, dtype=np.intp),
        metadata={},
    )
    with pytest.raises(ProteoForgeValidationError, match="non-finite"):
        build_protein_blocks(prepared)
