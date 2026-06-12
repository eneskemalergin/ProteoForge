"""End-to-end clustering integration tests."""

from __future__ import annotations

import numpy as np
import polars as pl

from proteoforge import (
    Config,
    assign_proteoforms,
    prepare,
    run_cluster,
    run_discordance,
)
from proteoforge.schema import (
    CLUSTER_ID,
    CLUSTER_RESULT_COLUMNS,
    DPF_ID,
    IS_DISCORDANT,
    PROTEIN_ID,
    PROTEOFORM_MAPPING_COLUMNS,
)

CONDITIONS = {"control": ("S1", "S2", "S3"), "treated": ("S4", "S5", "S6")}
SAMPLES = {
    "S1": "control",
    "S2": "control",
    "S3": "control",
    "S4": "treated",
    "S5": "treated",
    "S6": "treated",
}


def _discordant_frame() -> pl.DataFrame:
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for i in range(4):
            base = 10.0 + i
            for sample, condition in SAMPLES.items():
                value = base + rng.normal(scale=0.05)
                if protein == "P1" and i == 0 and condition == "treated":
                    value += 4.0
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": f"PEP{i}",
                        "sample_id": sample,
                        "intensity": value,
                    }
                )
    return pl.DataFrame(rows)


def test_prepare_discordance_cluster_proteoform_pipeline() -> None:
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        fdr=0.05,
        cut="hybrid_outlier",
    )
    prepared = prepare(_discordant_frame(), config)
    discordance = run_discordance(prepared, n_jobs=1)
    cluster = run_cluster(prepared, discordance, n_jobs=1)
    mapping = assign_proteoforms(prepared, discordance, cluster)

    assert set(cluster.table.columns) == CLUSTER_RESULT_COLUMNS
    assert set(mapping.table.columns) == PROTEOFORM_MAPPING_COLUMNS
    assert mapping.table.height == prepared.n_peptides
    assert cluster.table.height == prepared.n_peptides
    assert cluster.table.get_column(PROTEIN_ID).n_unique() >= 1
    assert mapping.table.get_column(CLUSTER_ID).null_count() == 0

    discordant = mapping.table.filter(pl.col(IS_DISCORDANT))
    assert discordant.height >= 1
    assert int(discordant.select((pl.col(DPF_ID) == 0).sum()).item()) == 0

    proteins_without_discordance = (
        mapping.table.group_by(PROTEIN_ID)
        .agg(pl.col(IS_DISCORDANT).any().alias("_any_discordant"))
        .filter(~pl.col("_any_discordant"))
        .get_column(PROTEIN_ID)
        .to_list()
    )
    for protein_id in proteins_without_discordance:
        subset = mapping.table.filter(pl.col(PROTEIN_ID) == protein_id)
        assert subset.get_column(CLUSTER_ID).null_count() == 0
        assert subset.get_column(DPF_ID).unique().to_list() == [0]


def _uniform_frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for i in range(4):
            base = 10.0 + i
            for sample, _condition in SAMPLES.items():
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": f"PEP{i}",
                        "sample_id": sample,
                        "intensity": base + rng.normal(scale=0.01),
                    }
                )
    return pl.DataFrame(rows)


def test_pipeline_with_no_discordant_peptides() -> None:
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        fdr=0.05,
    )
    prepared = prepare(_uniform_frame(), config)
    discordance = run_discordance(prepared, n_jobs=1)
    assert int(discordance.table.select(pl.col(IS_DISCORDANT).sum()).item()) == 0

    cluster = run_cluster(prepared, discordance, n_jobs=1)
    mapping = assign_proteoforms(prepared, discordance, cluster)

    assert cluster.table.height == prepared.n_peptides
    assert mapping.table.get_column(DPF_ID).unique().to_list() == [0]
    assert mapping.table.get_column(CLUSTER_ID).null_count() == 0
