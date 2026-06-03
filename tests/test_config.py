"""Tests for Config serialization and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proteoforge import Config
from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.types import ColumnMap


def test_config_defaults() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    assert cfg.min_peptides == 4
    assert cfg.model == "rlm"
    assert cfg.fdr == 0.001


def test_config_rejects_empty_control() -> None:
    with pytest.raises(ProteoForgeValidationError, match="control_condition"):
        Config(
            control_condition="  ",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        )


def test_config_rejects_control_not_in_conditions() -> None:
    with pytest.raises(ProteoForgeValidationError, match="not a key in conditions"):
        Config(
            control_condition="day1",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        )


def test_config_rejects_duplicate_sample_across_conditions() -> None:
    with pytest.raises(ProteoForgeValidationError, match="more than one condition"):
        Config(
            control_condition="a",
            conditions={"a": ("S1", "S2"), "b": ("S2", "S3")},
        )


def test_config_rejects_invalid_fdr() -> None:
    with pytest.raises(ProteoForgeValidationError, match="fdr"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            fdr=1.0,
        )


def test_config_replace() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        min_peptides=4,
    )
    updated = cfg.replace(min_peptides=6)
    assert updated.min_peptides == 6
    assert cfg.min_peptides == 4


def test_config_to_design_table() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
    )
    design = cfg.to_design_table()
    assert design.sample_ids == ("S1", "S2", "S3", "S4")
    assert design.condition_to_samples["control"] == ("S1", "S2")


def test_config_round_trip_dict() -> None:
    cfg = Config(
        control_condition="day1",
        conditions={"day1": ("S1", "S2"), "day3": ("S3", "S4")},
        min_peptides=5,
        column_map=ColumnMap(intensity="ms1"),
    )
    restored = Config.from_dict(cfg.to_dict())
    assert restored.control_condition == "day1"
    assert restored.min_peptides == 5
    assert restored.column_map.intensity == "ms1"
    assert restored.conditions["day3"] == ("S3", "S4")


def test_config_yaml_round_trip() -> None:
    pytest.importorskip("yaml")
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        min_peptides=4,
    )
    text = cfg.to_yaml()
    restored = Config.from_yaml(text)
    assert restored.control_condition == "control"
    assert restored.min_peptides == 4


def test_config_json_schema_matches_artifact() -> None:
    schema = Config.to_json_schema()
    artifact = json.loads(
        Path("src/proteoforge/schemas/config.schema.json").read_text(encoding="utf-8")
    )
    assert schema == artifact
