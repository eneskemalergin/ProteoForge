"""Prepare to discordance handoff: keys, joins, and row order."""

from __future__ import annotations

import numpy as np
import polars as pl

from proteoforge import Config, prepare, run_discordance
from proteoforge.schema import IS_DISCORDANT


def test_discordance_join_on_peptide_keys() -> None:
    """Merge discordance results back to long peptides by key, not row index."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for protein in ("P2", "P1"):
        for pep in ("B", "A"):
            base = 8.0 + hash((protein, pep)) % 5
            for sample, _condition in (
                ("S1", "control"),
                ("S2", "control"),
                ("S3", "treated"),
                ("S4", "treated"),
            ):
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": pep,
                        "sample_id": sample,
                        "intensity": base + float(rng.normal(scale=0.2)),
                    }
                )
    frame = pl.DataFrame(rows)
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        model="rlm",
        input_is_log2=True,
        min_peptides=2,
    )
    prepared = prepare(frame, config)
    result = run_discordance(prepared)

    assert result.table.height == prepared.n_peptides
    assert result.table.get_column("protein_id").to_list()[0] == "P1"
    assert prepared.peptides.get_column("protein_id").to_list()[0] == "P2"

    joined = prepared.peptides.join(
        result.table,
        on=["protein_id", "peptide_id"],
        how="left",
    )
    assert joined.height == prepared.peptides.height
    assert IS_DISCORDANT in joined.columns
    assert joined.null_count().sum_horizontal().item() == 0
