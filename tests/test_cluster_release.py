"""Tests for clustering contracts, IO errors, and scipy oracles."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist as scipy_pdist

from proteoforge import (
    Config,
    ProteoForgeIOError,
    ProteoForgeValidationError,
    assign_proteoforms,
    prepare_from_parquet,
    run_cluster,
)
from proteoforge.clustering._cuts import (
    DynamicTreeCut,
    FixedHeightCut,
    select_cut_strategy,
)
from proteoforge.clustering._distance import euclidean_condensed
from proteoforge.clustering._linkage import ward_linkage
from proteoforge.clustering._protocol import ProteinProfileBlock
from proteoforge.schema import (
    ADJUSTED_P_VALUE,
    CLUSTER_ID,
    CUT_METHOD,
    FIT_STATUS,
    IS_DISCORDANT,
    LINKAGE_METHOD,
    PEPTIDE_ID,
    PROTEIN_ID,
    RAW_P_VALUE,
    WITHIN_P_VALUE,
    validate_discordance_result_table,
)
from proteoforge.types import ClusterResult, DiscordanceResult, PreparedDataset


def _config(cut: str = "hybrid_outlier") -> Config:
    return Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        input_is_log2=True,
        min_peptides=2,
        cut=cut,  # type: ignore[arg-type]
    )


def test_prepare_from_parquet_missing_file_raises_io_error() -> None:
    config = _config()
    with pytest.raises(ProteoForgeIOError, match="not found"):
        prepare_from_parquet("missing-peptides.parquet", config)


def test_discordance_result_validates_table_columns() -> None:
    config = _config()
    valid = pl.DataFrame(
        {
            PROTEIN_ID: ["P1"],
            PEPTIDE_ID: ["A"],
            RAW_P_VALUE: [0.01],
            WITHIN_P_VALUE: [0.01],
            ADJUSTED_P_VALUE: [0.01],
            IS_DISCORDANT: [True],
            FIT_STATUS: ["ok"],
        }
    )
    validate_discordance_result_table(valid)
    DiscordanceResult(config=config, table=valid, metadata={})

    invalid = pl.DataFrame({PROTEIN_ID: ["P1"], PEPTIDE_ID: ["A"]})
    with pytest.raises(ProteoForgeValidationError, match="invalid columns"):
        validate_discordance_result_table(invalid)


def test_assign_proteoforms_rejects_config_mismatch() -> None:
    config_a = _config()
    config_b = _config(cut="dynamic_tree")
    prepared = PreparedDataset(
        config=config_a,
        peptides=pl.DataFrame(
            {
                PROTEIN_ID: ["P1"],
                PEPTIDE_ID: ["A"],
                "sample_id": ["S1"],
                "condition": ["control"],
                "intensity_normalized": [0.0],
            }
        ),
        sample_ids=("S1",),
        condition_levels=("control", "treated"),
        protein_index=np.zeros(1, dtype=np.intp),
        metadata={},
    )
    discordance = DiscordanceResult(
        config=config_a,
        table=pl.DataFrame(
            {
                PROTEIN_ID: ["P1"],
                PEPTIDE_ID: ["A"],
                RAW_P_VALUE: [0.01],
                WITHIN_P_VALUE: [0.01],
                ADJUSTED_P_VALUE: [0.01],
                IS_DISCORDANT: [True],
                FIT_STATUS: ["ok"],
            }
        ),
        metadata={},
    )
    cluster = ClusterResult(
        config=config_b,
        table=pl.DataFrame(
            schema={
                PROTEIN_ID: pl.String,
                PEPTIDE_ID: pl.String,
                CLUSTER_ID: pl.Int64,
                CUT_METHOD: pl.String,
                LINKAGE_METHOD: pl.String,
            }
        ),
        metadata={},
    )
    with pytest.raises(ProteoForgeValidationError, match="configs differ"):
        assign_proteoforms(prepared, discordance, cluster)


def test_run_cluster_single_peptide_assigns_one_cluster() -> None:
    config = _config()
    prepared = PreparedDataset(
        config=config,
        peptides=pl.DataFrame(
            {
                PROTEIN_ID: ["P1", "P1"],
                PEPTIDE_ID: ["A", "A"],
                "sample_id": ["S1", "S3"],
                "condition": ["control", "treated"],
                "intensity_normalized": [0.0, 1.0],
            }
        ),
        sample_ids=("S1", "S3"),
        condition_levels=("control", "treated"),
        protein_index=np.zeros(2, dtype=np.intp),
        metadata={},
    )
    discordance = DiscordanceResult(
        config=config,
        table=pl.DataFrame(
            {
                PROTEIN_ID: ["P1"],
                PEPTIDE_ID: ["A"],
                RAW_P_VALUE: [0.01],
                WITHIN_P_VALUE: [0.01],
                ADJUSTED_P_VALUE: [0.01],
                IS_DISCORDANT: [True],
                FIT_STATUS: ["ok"],
            }
        ),
        metadata={},
    )
    cluster = run_cluster(prepared, discordance, n_jobs=1)
    assert cluster.table.get_column(CLUSTER_ID).to_list() == [1]


def test_dynamic_tree_and_fixed_height_cuts() -> None:
    profiles = np.asarray([[0.0], [1.0], [2.0], [10.0]])
    distances = euclidean_condensed(profiles)
    linkage_matrix = ward_linkage(distances, n_samples=4)
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        cut="dynamic_tree",
        fixed_n_clusters=2,
    )
    dynamic = DynamicTreeCut().cut(
        linkage_matrix,
        distances,
        n_samples=4,
        config=config,
    )
    assert dynamic.shape == (4,)

    fixed = FixedHeightCut().cut(
        linkage_matrix,
        distances,
        n_samples=4,
        config=config.replace(cut="fixed_height"),
    )
    assert len(np.unique(fixed)) == 2

    assert select_cut_strategy("dynamic_tree").name == "dynamic_tree"
    assert select_cut_strategy("fixed_height").name == "fixed_height"


def test_euclidean_and_ward_match_scipy_oracles() -> None:
    rng = np.random.default_rng(7)
    profiles = rng.normal(size=(6, 3))
    ours = euclidean_condensed(profiles)
    oracle = scipy_pdist(profiles, metric="euclidean")
    np.testing.assert_allclose(ours, oracle, rtol=1e-10, atol=1e-10)

    linkage_ours = ward_linkage(ours, n_samples=profiles.shape[0])
    linkage_oracle = scipy_linkage(ours, method="ward")
    np.testing.assert_allclose(linkage_ours, linkage_oracle, rtol=1e-10, atol=1e-10)


def test_run_cluster_parallel_path() -> None:
    config = _config()
    rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for peptide in ("A", "B"):
            for sample, condition, value in (
                ("S1", "control", 0.0),
                ("S2", "control", 0.1),
                ("S3", "treated", 2.0),
                ("S4", "treated", 2.1),
            ):
                rows.append(
                    {
                        PROTEIN_ID: protein,
                        PEPTIDE_ID: peptide,
                        "sample_id": sample,
                        "condition": condition,
                        "intensity_normalized": value
                        + (5.0 if peptide == "B" else 0.0),
                    }
                )
    prepared = PreparedDataset(
        config=config,
        peptides=pl.DataFrame(rows),
        sample_ids=("S1", "S2", "S3", "S4"),
        condition_levels=("control", "treated"),
        protein_index=np.arange(len(rows), dtype=np.intp) // 4,
        metadata={},
    )
    discordance_rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for peptide in ("A", "B"):
            discordance_rows.append(
                {
                    PROTEIN_ID: protein,
                    PEPTIDE_ID: peptide,
                    RAW_P_VALUE: 0.01,
                    WITHIN_P_VALUE: 0.01,
                    ADJUSTED_P_VALUE: 0.01,
                    IS_DISCORDANT: peptide == "B",
                    FIT_STATUS: "ok",
                }
            )
    discordance = DiscordanceResult(
        config=config,
        table=pl.DataFrame(discordance_rows),
        metadata={},
    )
    cluster = run_cluster(prepared, discordance, n_jobs=2)
    assert cluster.table.height == 4
    assert cluster.metadata.get("n_jobs_effective", 1) >= 1


def test_build_profile_blocks_rejects_config_mismatch() -> None:
    from proteoforge.clustering._profiles import build_profile_blocks

    config_a = _config()
    config_b = _config(cut="dynamic_tree")
    prepared = PreparedDataset(
        config=config_a,
        peptides=pl.DataFrame(
            {
                PROTEIN_ID: ["P1"],
                PEPTIDE_ID: ["A"],
                "sample_id": ["S1"],
                "condition": ["control"],
                "intensity_normalized": [0.0],
            }
        ),
        sample_ids=("S1",),
        condition_levels=("control", "treated"),
        protein_index=np.zeros(1, dtype=np.intp),
        metadata={},
    )
    discordance = DiscordanceResult(
        config=config_b,
        table=pl.DataFrame(
            {
                PROTEIN_ID: ["P1"],
                PEPTIDE_ID: ["A"],
                RAW_P_VALUE: [0.01],
                WITHIN_P_VALUE: [0.01],
                ADJUSTED_P_VALUE: [0.01],
                IS_DISCORDANT: [True],
                FIT_STATUS: ["ok"],
            }
        ),
        metadata={},
    )
    with pytest.raises(ProteoForgeValidationError, match="different configs"):
        build_profile_blocks(prepared, discordance)


def test_public_exceptions_are_importable() -> None:
    from proteoforge import ProteoForgeIOError, ProteoForgeValidationError

    assert issubclass(ProteoForgeValidationError, Exception)
    assert issubclass(ProteoForgeIOError, Exception)


def test_degenerate_profile_block_returns_singleton_cluster() -> None:
    from proteoforge._cluster import _cluster_profile_block

    config = _config()
    block = ProteinProfileBlock(
        protein_id="P1",
        peptide_ids=("A",),
        profiles=np.asarray([[0.0, 1.0]], dtype=np.float64),
        is_discordant=np.asarray([True]),
        condition_levels=("control", "treated"),
    )
    labels = _cluster_profile_block(block, cut="hybrid_outlier", config=config)
    np.testing.assert_array_equal(labels, [1])
