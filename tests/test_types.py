"""Tests for PreparedDataset and related types."""

from __future__ import annotations

import numpy as np
import polars as pl

from proteoforge import Config
from proteoforge.schema import NORMALIZED_INTENSITY
from proteoforge.types import PreparedDataset


def test_prepared_dataset_properties() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    peptides = pl.DataFrame(
        {
            "protein_id": ["P1", "P1", "P1", "P1"],
            "peptide_id": ["A", "A", "A", "A"],
            "sample_id": ["S1", "S2", "S3", "S4"],
            "condition": ["control", "control", "treated", "treated"],
            "intensity": [1.0, 2.0, 3.0, 4.0],
            NORMALIZED_INTENSITY: [0.1, 0.2, 0.3, 0.4],
        }
    )
    dataset = PreparedDataset(
        config=config,
        peptides=peptides,
        sample_ids=("S1", "S2", "S3", "S4"),
        condition_levels=("control", "treated"),
        protein_index=np.array([0, 0, 0, 0], dtype=np.intp),
        metadata={"n_proteins": 1},
    )
    assert dataset.n_peptides == 1
    assert dataset.n_samples == 4
    assert dataset.n_proteins == 1
    assert dataset.intensity_normalized.shape == (4,)
