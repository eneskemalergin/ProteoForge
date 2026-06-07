"""Frozen configuration for ProteoForge pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

from proteoforge._exceptions import ProteoForgeIOError, ProteoForgeValidationError
from proteoforge.types import ColumnMap, DesignTable

ModelName = Literal["rlm", "wls", "ebayes"]
CutStrategy = Literal["hybrid_outlier", "fixed_height", "dynamic_tree"]

_LOADABLE_MODELS: frozenset[str] = frozenset({"rlm", "wls"})
_VALID_CUT_STRATEGIES: frozenset[str] = frozenset(
    {"hybrid_outlier", "fixed_height", "dynamic_tree"}
)


@dataclass(frozen=True)
class Config:
    """
    Immutable pipeline configuration.

    The experimental design is embedded in ``conditions``: each key is a
    condition label and each value lists the sample IDs to include. Only
    samples listed here are used; extra samples present in the peptide table
    are dropped. ``control_condition`` must be one of the condition keys.
    """

    control_condition: str
    conditions: dict[str, tuple[str, ...]]
    min_peptides: int = 4
    input_is_log2: bool = False
    column_map: ColumnMap = field(default_factory=ColumnMap)
    model: ModelName = "rlm"
    fdr: float = 0.001
    linkage: str = "ward"
    cut: CutStrategy = "hybrid_outlier"
    n_jobs: int = -1
    wls_biological_weight: float = 0.5
    correction_within: str = "bonferroni"
    correction_global: str = "fdr_bh"

    def __post_init__(self) -> None:
        if not self.control_condition.strip():
            msg = "control_condition must be a non-empty string."
            raise ProteoForgeValidationError(msg)
        if not self.conditions:
            msg = (
                "conditions must list at least one condition with sample IDs. "
                "Example: conditions: {day1: [S1, S2], day3: [S3, S4]}"
            )
            raise ProteoForgeValidationError(msg)
        if self.control_condition not in self.conditions:
            valid = sorted(self.conditions)
            msg = (
                f"control_condition '{self.control_condition}' is not a key in "
                f"conditions. Valid condition keys: {valid}."
            )
            raise ProteoForgeValidationError(msg)
        if len(self.conditions) < 2:
            msg = "conditions must include at least two condition keys."
            raise ProteoForgeValidationError(msg)

        seen_samples: set[str] = set()
        for condition, samples in self.conditions.items():
            if not condition.strip():
                msg = "Condition keys in conditions must be non-empty strings."
                raise ProteoForgeValidationError(msg)
            if len(samples) < 2:
                msg = (
                    f"Condition '{condition}' lists fewer than 2 samples "
                    f"({len(samples)}). Add replicates in config.conditions."
                )
                raise ProteoForgeValidationError(msg)
            for sample in samples:
                if not str(sample).strip():
                    msg = f"Condition '{condition}' contains an empty sample ID."
                    raise ProteoForgeValidationError(msg)
                if sample in seen_samples:
                    msg = (
                        f"Sample '{sample}' appears in more than one condition. "
                        "Each sample must belong to exactly one condition."
                    )
                    raise ProteoForgeValidationError(msg)
                seen_samples.add(sample)

        if self.min_peptides < 2:
            msg = "min_peptides must be at least 2."
            raise ProteoForgeValidationError(msg)
        if not 0.0 < self.fdr < 1.0:
            msg = "fdr must be between 0 and 1 (exclusive)."
            raise ProteoForgeValidationError(msg)
        if self.n_jobs == 0 or self.n_jobs < -1:
            msg = "n_jobs must be -1 or a positive integer."
            raise ProteoForgeValidationError(msg)
        if not 0.0 < self.wls_biological_weight <= 1.0:
            msg = "wls_biological_weight must be in (0, 1]."
            raise ProteoForgeValidationError(msg)
        if self.model == "ebayes":
            msg = "model='ebayes' is not implemented in v0.0.2. Use 'rlm' or 'wls'."
            raise ProteoForgeValidationError(msg)
        if self.model not in _LOADABLE_MODELS:
            msg = (
                f"model '{self.model}' is not supported. "
                "Valid loadable models: 'rlm', 'wls'."
            )
            raise ProteoForgeValidationError(msg)
        if self.cut not in _VALID_CUT_STRATEGIES:
            valid = sorted(_VALID_CUT_STRATEGIES)
            msg = f"cut '{self.cut}' is not supported. Valid: {valid}."
            raise ProteoForgeValidationError(msg)

        from proteoforge._correction import VALID_METHODS

        for field_name in ("correction_within", "correction_global"):
            method = getattr(self, field_name)
            if method not in VALID_METHODS or method in (None, "none"):
                valid = sorted(str(m) for m in VALID_METHODS if m not in (None, "none"))
                msg = (
                    f"{field_name} '{method}' is not a supported correction "
                    f"method. Valid: {valid}."
                )
                raise ProteoForgeValidationError(msg)

    @property
    def condition_levels(self) -> tuple[str, ...]:
        """Condition order with control first, then remaining config keys."""
        others = [name for name in self.conditions if name != self.control_condition]
        return (self.control_condition, *others)

    @property
    def selected_sample_ids(self) -> frozenset[str]:
        """Sample IDs listed across all configured conditions."""
        return frozenset(
            sample for samples in self.conditions.values() for sample in samples
        )

    def to_design_table(self) -> DesignTable:
        """Build a ``DesignTable`` from configured conditions and samples."""
        sample_to_condition: dict[str, str] = {}
        for condition, samples in self.conditions.items():
            for sample in samples:
                sample_to_condition[sample] = condition

        ordered_samples: list[str] = []
        condition_to_samples: dict[str, tuple[str, ...]] = {}
        for condition in self.condition_levels:
            samples = self.conditions[condition]
            condition_to_samples[condition] = samples
            ordered_samples.extend(samples)

        return DesignTable(
            sample_ids=tuple(ordered_samples),
            sample_to_condition=sample_to_condition,
            condition_to_samples=condition_to_samples,
        )

    def replace(self, **changes: Any) -> Self:
        """Return a copy with selected fields replaced."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct from a dictionary (e.g. parsed YAML)."""
        column_map_data = data.get("column_map", {})
        column_map = (
            column_map_data
            if isinstance(column_map_data, ColumnMap)
            else ColumnMap(**column_map_data)
        )
        conditions_raw = data.get("conditions")
        if conditions_raw is None:
            msg = (
                "Config must include a 'conditions' mapping of "
                "condition_name: [sample_id, ...]."
            )
            raise ProteoForgeValidationError(msg)
        if not isinstance(conditions_raw, dict):
            msg = "conditions must be a mapping of condition names to sample lists."
            raise ProteoForgeValidationError(msg)

        conditions: dict[str, tuple[str, ...]] = {}
        for condition, samples in conditions_raw.items():
            if not isinstance(samples, (list, tuple)):
                msg = (
                    f"conditions['{condition}'] must be a list of sample IDs, "
                    f"got {type(samples).__name__}."
                )
                raise ProteoForgeValidationError(msg)
            conditions[str(condition)] = tuple(str(sample) for sample in samples)

        if "control_condition" not in data:
            msg = "Config must include 'control_condition'."
            raise ProteoForgeValidationError(msg)

        return cls(
            control_condition=str(data["control_condition"]),
            conditions=conditions,
            min_peptides=int(data.get("min_peptides", 4)),
            input_is_log2=bool(data.get("input_is_log2", False)),
            column_map=column_map,
            model=data.get("model", "rlm"),
            fdr=float(data.get("fdr", 0.001)),
            linkage=str(data.get("linkage", "ward")),
            cut=data.get("cut", "hybrid_outlier"),
            n_jobs=int(data.get("n_jobs", -1)),
            wls_biological_weight=float(data.get("wls_biological_weight", 0.5)),
            correction_within=str(data.get("correction_within", "bonferroni")),
            correction_global=str(data.get("correction_global", "fdr_bh")),
        )

    def to_yaml(self) -> str:
        """Serialize to YAML."""
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required for YAML serialization."
            raise ProteoForgeValidationError(msg) from exc
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """Parse YAML configuration text."""
        try:
            import yaml
        except ImportError as exc:
            msg = "PyYAML is required for YAML parsing."
            raise ProteoForgeValidationError(msg) from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = "YAML root must be a mapping."
            raise ProteoForgeValidationError(msg)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> Self:
        """
        Load configuration from a YAML file.

        Raises
        ------
        ProteoForgeIOError
            If the file does not exist.
        ProteoForgeValidationError
            If the file cannot be parsed or validation fails.
        """
        config_path = Path(path)
        try:
            text = config_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            msg = f"Config file not found: {config_path}"
            raise ProteoForgeIOError(msg) from exc
        return cls.from_yaml(text)

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema describing ``Config``."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ProteoForge Config",
            "type": "object",
            "required": ["control_condition", "conditions"],
            "properties": {
                "control_condition": {"type": "string", "minLength": 1},
                "conditions": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "min_peptides": {"type": "integer", "minimum": 2, "default": 4},
                "input_is_log2": {"type": "boolean", "default": False},
                "column_map": {
                    "type": "object",
                    "properties": {
                        "protein_id": {"type": "string"},
                        "peptide_id": {"type": "string"},
                        "sample_id": {"type": "string"},
                        "condition": {"type": "string"},
                        "intensity": {"type": "string"},
                        "is_real": {"type": "string"},
                        "is_complete_missing": {"type": "string"},
                        "weight": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "model": {
                    "type": "string",
                    "enum": ["rlm", "wls"],
                    "default": "rlm",
                },
                "fdr": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                    "default": 0.001,
                },
                "linkage": {"type": "string", "default": "ward"},
                "cut": {
                    "type": "string",
                    "enum": ["hybrid_outlier", "fixed_height", "dynamic_tree"],
                    "default": "hybrid_outlier",
                },
                "n_jobs": {"type": "integer", "default": -1},
                "wls_biological_weight": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                    "default": 0.5,
                },
                "correction_within": {
                    "type": "string",
                    "enum": ["bonferroni", "holm", "hochberg", "fdr", "fdr_bh", "BY"],
                    "default": "bonferroni",
                },
                "correction_global": {
                    "type": "string",
                    "enum": ["bonferroni", "holm", "hochberg", "fdr", "fdr_bh", "BY"],
                    "default": "fdr_bh",
                },
            },
            "additionalProperties": False,
        }

    def write_json_schema(self, path: str | Path) -> None:
        """Write the JSON Schema artifact to ``path``."""
        schema_path = Path(path)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            json.dumps(self.to_json_schema(), indent=2) + "\n",
            encoding="utf-8",
        )
