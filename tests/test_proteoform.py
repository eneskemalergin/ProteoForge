"""Tests for dPF assignment rules and validation."""

from __future__ import annotations

import polars as pl
import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._proteoform import _cluster_dpf_map
from proteoforge.schema import CLUSTER_ID, DPF_ID, IS_DISCORDANT, PEPTIDE_ID, PROTEIN_ID


def test_cluster_dpf_map_assigns_expected_ids() -> None:
    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1", "P1", "P1", "P1", "P1", "P1", "P1"],
            PEPTIDE_ID: ["A", "B", "C", "D", "E", "F", "G"],
            IS_DISCORDANT: [True, False, True, True, False, False, False],
            CLUSTER_ID: [1, 1, 2, 3, 3, 4, 4],
        }
    )
    result = _cluster_dpf_map(table).sort([PROTEIN_ID, CLUSTER_ID])
    assert result.to_dicts() == [
        {PROTEIN_ID: "P1", CLUSTER_ID: 1, DPF_ID: 1},
        {PROTEIN_ID: "P1", CLUSTER_ID: 2, DPF_ID: -1},
        {PROTEIN_ID: "P1", CLUSTER_ID: 3, DPF_ID: 2},
        {PROTEIN_ID: "P1", CLUSTER_ID: 4, DPF_ID: 0},
    ]


def test_validate_mapping_rejects_discordant_with_dpf_zero() -> None:
    from proteoforge._proteoform import _validate_mapping

    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1"],
            PEPTIDE_ID: ["A"],
            IS_DISCORDANT: [True],
            CLUSTER_ID: [1],
            DPF_ID: [0],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="dpf_id = 0"):
        _validate_mapping(table)


def test_validate_mapping_rejects_non_discordant_singleton_dpf() -> None:
    from proteoforge._proteoform import _validate_mapping

    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1"],
            PEPTIDE_ID: ["A"],
            IS_DISCORDANT: [False],
            CLUSTER_ID: [1],
            DPF_ID: [-1],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="dpf_id = -1"):
        _validate_mapping(table)


def test_validate_mapping_rejects_dpf_below_minus_one() -> None:
    from proteoforge._proteoform import _validate_mapping

    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1"],
            PEPTIDE_ID: ["A"],
            IS_DISCORDANT: [True],
            CLUSTER_ID: [1],
            DPF_ID: [-2],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="below -1"):
        _validate_mapping(table)


def test_validate_mapping_rejects_multi_discordant_cluster_with_dpf_zero() -> None:
    from proteoforge._proteoform import _validate_mapping

    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1", "P1"],
            PEPTIDE_ID: ["A", "B"],
            IS_DISCORDANT: [False, True],
            CLUSTER_ID: [1, 1],
            DPF_ID: [0, 1],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="positive dPF"):
        _validate_mapping(table)


def test_validate_mapping_rejects_inconsistent_dpf_within_cluster() -> None:
    from proteoforge._proteoform import _validate_mapping

    table = pl.DataFrame(
        {
            PROTEIN_ID: ["P1", "P1"],
            PEPTIDE_ID: ["A", "B"],
            IS_DISCORDANT: [True, True],
            CLUSTER_ID: [1, 1],
            DPF_ID: [1, 2],
        }
    )
    with pytest.raises(ProteoForgeValidationError, match="same dPF ID"):
        _validate_mapping(table)
