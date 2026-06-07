"""Batching strategies must produce identical discordance p-values."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from proteoforge import Config, prepare
from proteoforge._discordance import run_discordance
from proteoforge.schema import ADJUSTED_P_VALUE, RAW_P_VALUE

CONDITIONS = {"control": ("S1", "S2", "S3"), "treated": ("S4", "S5", "S6")}


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(99)
    samples = {
        "S1": "control",
        "S2": "control",
        "S3": "control",
        "S4": "treated",
        "S5": "treated",
        "S6": "treated",
    }
    layout = {"P1": 4, "P2": 5, "P3": 4}
    rows: list[dict[str, object]] = []
    for protein, n_peptides in layout.items():
        for i in range(n_peptides):
            for sample, condition in samples.items():
                value = 10.0 + i + rng.normal(scale=0.3)
                if protein == "P1" and i == 0 and condition == "treated":
                    value += 2.0
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": f"{protein}_PEP{i}",
                        "sample_id": sample,
                        "intensity": value,
                        "is_real": True,
                        "is_complete_missing": False,
                    }
                )
    return pl.DataFrame(rows)


@pytest.mark.parametrize("model", ["rlm", "wls"])
def test_batching_strategies_agree(model: str) -> None:
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model=model,
        input_is_log2=True,
    )
    prepared = prepare(_frame(), config)

    scalar = run_discordance(prepared, batching="scalar").table.sort(
        ["protein_id", "peptide_id"]
    )
    protein = run_discordance(prepared, batching="protein").table.sort(
        ["protein_id", "peptide_id"]
    )
    shape = run_discordance(prepared, batching="shape").table.sort(
        ["protein_id", "peptide_id"]
    )

    for column in (RAW_P_VALUE, ADJUSTED_P_VALUE):
        base = scalar.get_column(column).to_numpy()
        np.testing.assert_allclose(
            base, protein.get_column(column).to_numpy(), rtol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            base, shape.get_column(column).to_numpy(), rtol=1e-10, equal_nan=True
        )
