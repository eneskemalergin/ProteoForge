"""Tests for clustering result table contracts."""

from __future__ import annotations

import polars as pl
import pytest

from proteoforge import Config
from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.schema import (
    CLUSTER_ID,
    CLUSTER_RESULT_COLUMNS,
    CUT_METHOD,
    DPF_ID,
    IS_DISCORDANT,
    LINKAGE_METHOD,
    PEPTIDE_ID,
    PROTEIN_ID,
    PROTEOFORM_MAPPING_COLUMNS,
    validate_cluster_result_table,
    validate_discordance_result_table,
    validate_proteoform_mapping_table,
)
from proteoforge.types import ClusterResult, DiscordanceResult, ProteoformMappingResult


def test_discordance_result_validates_columns() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    table = pl.DataFrame({PROTEIN_ID: ["P1"], PEPTIDE_ID: ["A"]})
    with pytest.raises(ProteoForgeValidationError, match="invalid columns"):
        validate_discordance_result_table(table)
    with pytest.raises(ProteoForgeValidationError, match="invalid columns"):
        DiscordanceResult(config=config, table=table, metadata={})


def test_empty_cluster_result_has_contract_columns() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    table = pl.DataFrame(
        schema={
            PROTEIN_ID: pl.String,
            PEPTIDE_ID: pl.String,
            CLUSTER_ID: pl.Int64,
            CUT_METHOD: pl.String,
            LINKAGE_METHOD: pl.String,
        }
    )
    result = ClusterResult(config=config, table=table, metadata={})
    assert set(result.table.columns) == CLUSTER_RESULT_COLUMNS


def test_cluster_result_rejects_extra_columns() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    table = pl.DataFrame({PROTEIN_ID: ["P1"], PEPTIDE_ID: ["A"], "extra": [1]})
    with pytest.raises(ProteoForgeValidationError, match="invalid columns"):
        validate_cluster_result_table(table)
    with pytest.raises(ProteoForgeValidationError, match="invalid columns"):
        ClusterResult(config=config, table=table, metadata={})


def test_proteoform_mapping_result_validates_columns() -> None:
    config = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1"],
            PEPTIDE_ID: ["A"],
            IS_DISCORDANT: [False],
            CLUSTER_ID: [None],
            DPF_ID: [0],
        }
    )
    validate_proteoform_mapping_table(table)
    result = ProteoformMappingResult(config=config, table=table, metadata={})
    assert set(result.table.columns) == PROTEOFORM_MAPPING_COLUMNS
