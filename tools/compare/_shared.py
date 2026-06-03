"""Shared helpers for reference vs ProteoForge comparison."""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUTPUT_NAME = "normalized.parquet"
KEYS = ("protein_id", "peptide_id", "sample_id")
VALUE = "intensity_normalized"


def load_config_dict(config_path: Path) -> dict[str, object]:
    import yaml

    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_sample_ids(config_dict: dict[str, object]) -> list[str]:
    conditions = config_dict["conditions"]
    if not isinstance(conditions, dict):
        msg = "config.conditions must be a mapping."
        raise TypeError(msg)
    ordered: list[str] = []
    for samples in conditions.values():
        if not isinstance(samples, list):
            msg = "Each condition value must be a list of sample IDs."
            raise TypeError(msg)
        ordered.extend(str(sample) for sample in samples)
    return ordered


def apply_column_map(frame: pl.DataFrame, column_map: dict[str, str]) -> pl.DataFrame:
    rename = {
        source: canonical
        for canonical, source in column_map.items()
        if source in frame.columns and source != canonical
    }
    if rename:
        frame = frame.rename(rename)
    return frame


def read_scoped_reference(input_path: Path, config_path: Path) -> pl.DataFrame:
    """Read parquet and restrict to configured samples (reference-style I/O)."""
    config_dict = load_config_dict(config_path)
    column_map = config_dict.get("column_map", {})
    if not isinstance(column_map, dict):
        msg = "config.column_map must be a mapping."
        raise TypeError(msg)

    frame = pl.read_parquet(input_path)
    frame = apply_column_map(frame, {k: str(v) for k, v in column_map.items()})
    selected = selected_sample_ids(config_dict)
    return frame.filter(pl.col("sample_id").is_in(selected)).select(
        "protein_id",
        "peptide_id",
        "sample_id",
        "intensity",
    )


def write_normalized_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write normalized long parquet (parity checks sort on read)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
