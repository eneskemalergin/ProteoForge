"""Tests for clustering handoffs and dPF assignment."""

from __future__ import annotations

import numpy as np
import polars as pl

from proteoforge import Config
from proteoforge._cluster import run_cluster
from proteoforge._proteoform import assign_proteoforms
from proteoforge.clustering._profiles import build_profile_blocks
from proteoforge.schema import (
    ADJUSTED_P_VALUE,
    CLUSTER_ID,
    CUT_METHOD,
    DPF_ID,
    FIT_STATUS,
    IS_DISCORDANT,
    LINKAGE_METHOD,
    PEPTIDE_ID,
    PROTEIN_ID,
    RAW_P_VALUE,
    WITHIN_P_VALUE,
)
from proteoforge.types import ClusterResult, DiscordanceResult, PreparedDataset


def _config(cut: str = "fixed_height") -> Config:
    return Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        input_is_log2=True,
        min_peptides=2,
        cut=cut,  # type: ignore[arg-type]
    )


def _prepared(config: Config) -> PreparedDataset:
    rows: list[dict[str, object]] = []
    profiles = {
        "P1": {
            "A": [0.0, 0.0, 2.0, 2.0],
            "B": [0.1, 0.1, 2.1, 2.1],
            "C": [5.0, 5.0, 1.0, 1.0],
            "D": [5.1, 5.1, 1.1, 1.1],
            "E": [9.0, 9.0, 9.0, 9.0],
        },
        "P2": {
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0, 2.0],
        },
    }
    samples = ("S1", "S2", "S3", "S4")
    conditions = {
        "S1": "control",
        "S2": "control",
        "S3": "treated",
        "S4": "treated",
    }
    for protein, peptides in profiles.items():
        for peptide, values in peptides.items():
            for sample, value in zip(samples, values, strict=True):
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": peptide,
                        "sample_id": sample,
                        "condition": conditions[sample],
                        "intensity_normalized": value,
                    }
                )
    frame = pl.DataFrame(rows)
    return PreparedDataset(
        config=config,
        peptides=frame,
        sample_ids=samples,
        condition_levels=("control", "treated"),
        protein_index=np.zeros(frame.height, dtype=np.intp),
        metadata={"n_proteins": 2},
    )


def _discordance(config: Config) -> DiscordanceResult:
    n = 7
    return DiscordanceResult(
        config=config,
        table=pl.DataFrame(
            {
                PROTEIN_ID: ["P1", "P1", "P1", "P1", "P1", "P2", "P2"],
                PEPTIDE_ID: ["A", "B", "C", "D", "E", "A", "B"],
                RAW_P_VALUE: [0.01] * n,
                WITHIN_P_VALUE: [0.01] * n,
                ADJUSTED_P_VALUE: [0.01] * n,
                IS_DISCORDANT: [True, False, True, True, False, False, False],
                FIT_STATUS: ["ok"] * n,
            }
        ),
        metadata={},
    )


def test_build_profile_blocks_includes_every_protein_in_scope() -> None:
    config = _config()
    prepared = _prepared(config)
    discordance = _discordance(config)

    blocks = build_profile_blocks(prepared, discordance)

    assert [block.protein_id for block in blocks] == ["P1", "P2"]
    assert blocks[0].peptide_ids == ("A", "B", "C", "D", "E")
    np.testing.assert_array_equal(
        blocks[0].is_discordant,
        [True, False, True, True, False],
    )
    assert blocks[0].profiles.shape == (5, 2)
    assert blocks[1].peptide_ids == ("A", "B")
    np.testing.assert_array_equal(blocks[1].is_discordant, [False, False])
    assert blocks[1].profiles.shape == (2, 2)


def test_run_cluster_returns_rows_for_all_proteins_in_scope() -> None:
    config = _config(cut="fixed_height")
    prepared = _prepared(config)
    discordance = _discordance(config)

    result = run_cluster(prepared, discordance, n_jobs=1)

    assert result.table.columns == [
        PROTEIN_ID,
        PEPTIDE_ID,
        CLUSTER_ID,
        CUT_METHOD,
        LINKAGE_METHOD,
    ]
    assert result.table.height == prepared.n_peptides
    assert set(result.table.get_column(PROTEIN_ID).to_list()) == {"P1", "P2"}
    assert result.metadata["n_proteins"] == 2
    assert result.metadata["n_discordant_proteins"] == 1
    assert result.n_clustered_peptides == 7


def test_assign_proteoforms_applies_cluster_level_rule() -> None:
    config = _config()
    prepared = _prepared(config)
    discordance = _discordance(config)
    cluster = ClusterResult(
        config=config,
        table=pl.DataFrame(
            {
                PROTEIN_ID: ["P1", "P1", "P1", "P1", "P1"],
                PEPTIDE_ID: ["A", "B", "C", "D", "E"],
                CLUSTER_ID: [1, 1, 2, 3, 3],
                CUT_METHOD: ["fixed_height"] * 5,
                LINKAGE_METHOD: ["ward"] * 5,
            }
        ),
        metadata={},
    )

    mapping = assign_proteoforms(prepared, discordance, cluster)
    table = mapping.table.sort([PROTEIN_ID, PEPTIDE_ID])

    p1 = {
        row[PEPTIDE_ID]: row[DPF_ID]
        for row in table.filter(pl.col(PROTEIN_ID) == "P1").iter_rows(named=True)
    }
    assert p1 == {"A": 1, "B": 1, "C": -1, "D": 2, "E": 2}

    p2 = table.filter(pl.col(PROTEIN_ID) == "P2")
    assert p2.get_column(DPF_ID).to_list() == [0, 0]
    assert p2.get_column(CLUSTER_ID).null_count() == 2
    assert mapping.n_differential_peptides == 4
    assert mapping.n_singletons == 1
