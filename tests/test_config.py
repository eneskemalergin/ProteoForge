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
    assert cfg.correction_within == "bonferroni"
    assert cfg.correction_global == "fdr_bh"
    assert cfg.cut == "hybrid_outlier"
    assert cfg.cluster_min_clusters == 1
    assert cfg.cluster_max_clusters is None
    assert cfg.fixed_n_clusters == 2
    assert cfg.hybrid_outlier_threshold == 0.05
    assert cfg.cluster_min_peptides == 2


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


def test_config_rejects_ebayes() -> None:
    with pytest.raises(ProteoForgeValidationError, match="ebayes"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            model="ebayes",
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


def test_config_from_dict_requires_control_condition() -> None:
    with pytest.raises(ProteoForgeValidationError, match="control_condition"):
        Config.from_dict(
            {
                "conditions": {
                    "control": ["S1", "S2"],
                    "treated": ["S3", "S4"],
                },
            }
        )


def test_config_from_dict_requires_conditions_mapping() -> None:
    with pytest.raises(ProteoForgeValidationError, match="conditions"):
        Config.from_dict({"control_condition": "control"})


def test_config_rejects_single_condition() -> None:
    with pytest.raises(ProteoForgeValidationError, match="at least two"):
        Config(control_condition="only", conditions={"only": ("S1", "S2")})


def test_config_rejects_fewer_than_two_samples_per_condition() -> None:
    with pytest.raises(ProteoForgeValidationError, match="fewer than 2 samples"):
        Config(
            control_condition="control",
            conditions={"control": ("S1",), "treated": ("S2", "S3")},
        )


def test_config_rejects_invalid_n_jobs() -> None:
    with pytest.raises(ProteoForgeValidationError, match="n_jobs"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            n_jobs=0,
        )


def test_config_rejects_invalid_correction_method() -> None:
    with pytest.raises(ProteoForgeValidationError, match="correction_within"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            correction_within="none",
        )


def test_config_accepts_qvalue_correction() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        correction_global="qvalue",
    )
    assert cfg.correction_global == "qvalue"


def test_config_accepts_hommel_correction() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        correction_within="hommel",
    )
    assert cfg.correction_within == "hommel"


def test_config_rejects_invalid_wls_biological_weight() -> None:
    with pytest.raises(ProteoForgeValidationError, match="wls_biological_weight"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            wls_biological_weight=0.0,
        )


def test_config_rejects_min_peptides_below_two() -> None:
    with pytest.raises(ProteoForgeValidationError, match="min_peptides"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            min_peptides=1,
        )


def test_config_json_schema_matches_artifact() -> None:
    schema = Config.to_json_schema()
    artifact = json.loads(
        Path("src/proteoforge/schemas/config.schema.json").read_text(encoding="utf-8")
    )
    assert schema == artifact


def test_config_rejects_unknown_model() -> None:
    with pytest.raises(ProteoForgeValidationError, match="model 'foo'"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            model="foo",  # type: ignore[arg-type]
        )


def test_config_rejects_unknown_model_from_dict() -> None:
    with pytest.raises(ProteoForgeValidationError, match="model 'foo'"):
        Config.from_dict(
            {
                "control_condition": "control",
                "conditions": {
                    "control": ["S1", "S2"],
                    "treated": ["S3", "S4"],
                },
                "model": "foo",
            }
        )


def test_config_rejects_invalid_cut() -> None:
    with pytest.raises(ProteoForgeValidationError, match="cut 'bad'"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            cut="bad",  # type: ignore[arg-type]
        )


def test_config_rejects_invalid_linkage() -> None:
    with pytest.raises(ProteoForgeValidationError, match="linkage 'complete'"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            linkage="complete",
        )


def test_config_rejects_cluster_max_below_min() -> None:
    with pytest.raises(ProteoForgeValidationError, match="cluster_max_clusters"):
        Config(
            control_condition="control",
            conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
            cluster_min_clusters=3,
            cluster_max_clusters=2,
        )


def test_config_rejects_non_median_profile_aggregation() -> None:
    with pytest.raises(ProteoForgeValidationError, match="profile_aggregation"):
        Config.from_dict(
            {
                "control_condition": "control",
                "conditions": {
                    "control": ["S1", "S2"],
                    "treated": ["S3", "S4"],
                },
                "profile_aggregation": "mean",
            }
        )


def test_config_ignores_profile_aggregation_median() -> None:
    cfg = Config.from_dict(
        {
            "control_condition": "control",
            "conditions": {
                "control": ["S1", "S2"],
                "treated": ["S3", "S4"],
            },
            "profile_aggregation": "median",
        }
    )
    assert cfg.cut == "hybrid_outlier"


def test_config_cluster_fields_round_trip() -> None:
    cfg = Config(
        control_condition="control",
        conditions={"control": ("S1", "S2"), "treated": ("S3", "S4")},
        linkage="ward",
        cut="dynamic_tree",
        cluster_min_clusters=2,
        cluster_max_clusters=5,
        hybrid_outlier_threshold=0.1,
        cluster_min_peptides=3,
    )
    restored = Config.from_dict(cfg.to_dict())
    assert restored.cut == "dynamic_tree"
    assert restored.cluster_max_clusters == 5
    assert restored.cluster_min_peptides == 3


def test_config_from_yaml_path_missing_file(tmp_path: Path) -> None:
    from proteoforge._exceptions import ProteoForgeIOError

    with pytest.raises(ProteoForgeIOError, match="Config file not found"):
        Config.from_yaml_path(tmp_path / "missing.yaml")
