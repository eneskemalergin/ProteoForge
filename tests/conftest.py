"""Shared pytest fixtures.

Use only small synthetic data here or under tests/fixtures/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from proteoforge import Config

FIXTURES = Path(__file__).resolve().parent / "fixtures"

MINIMAL_CONDITIONS = {
    "control": ("S1", "S2"),
    "treated": ("S3", "S4"),
}


@pytest.fixture
def minimal_config() -> Config:
    """Default config for synthetic fixtures."""
    return Config(control_condition="control", conditions=MINIMAL_CONDITIONS)


@pytest.fixture
def minimal_peptides_frame() -> pl.DataFrame:
    """Synthetic long peptide table with two proteins and four peptides each."""
    rng = np.random.default_rng(42)
    proteins = ["P11111", "P22222"]
    peptides_per_protein = [f"PEP{i}" for i in range(1, 5)]
    samples = ["S1", "S2", "S3", "S4"]
    rows: list[dict[str, object]] = []
    base = 1000.0
    for protein in proteins:
        for peptide in peptides_per_protein:
            for sample_idx, sample in enumerate(samples):
                intensity = base + rng.uniform(-50, 50) + sample_idx * 25
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": peptide,
                        "sample_id": sample,
                        "intensity": float(intensity),
                    }
                )
                base += 1.5
    return pl.DataFrame(rows)


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to committed test fixtures."""
    return FIXTURES
