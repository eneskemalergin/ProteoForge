"""Shared helpers for reference vs ProteoForge comparison."""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUTPUT_NAME = "normalized.parquet"
KEYS = ("protein_id", "peptide_id", "sample_id")
VALUE = "intensity_normalized"


def load_config_dict(config_path: Path) -> dict[str, object]:
    """
    Load a pipeline config YAML file as a plain dictionary.

    Parameters
    ----------
    config_path
        Path to ``config.yaml``.

    Returns
    -------
    dict
        Parsed YAML root mapping.
    """
    import yaml

    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_sample_ids(config_dict: dict[str, object]) -> list[str]:
    """
    Flatten sample IDs from a config ``conditions`` mapping.

    Parameters
    ----------
    config_dict
        Parsed config with a ``conditions`` key.

    Returns
    -------
    list[str]
        Sample IDs in YAML mapping iteration order.

    Raises
    ------
    TypeError
        If ``conditions`` is missing or malformed.
    """
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
    """
    Rename source columns to canonical names using a config-style column map.

    Parameters
    ----------
    frame
        Input table with source column names.
    column_map
        Mapping from canonical field name to source column name, as stored
        under ``column_map`` in config YAML.

    Returns
    -------
    polars.DataFrame
        Table with canonical names applied where mappings differ.
    """
    rename = {
        source: canonical
        for canonical, source in column_map.items()
        if source in frame.columns and source != canonical
    }
    if rename:
        frame = frame.rename(rename)
    return frame


def read_scoped_reference(input_path: Path, config_path: Path) -> pl.DataFrame:
    """
    Read parquet and restrict to configured samples for reference normalization.

    Parameters
    ----------
    input_path
        Long-format peptide parquet.
    config_path
        Pipeline config YAML with ``conditions`` and optional ``column_map``.

    Returns
    -------
    polars.DataFrame
        Scoped table with canonical key and intensity columns only.

    Raises
    ------
    TypeError
        If ``conditions`` or ``column_map`` has an invalid shape.
    """
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
    """
    Write a normalized long parquet for parity comparison.

    Parameters
    ----------
    frame
        Long table including ``intensity_normalized``.
    path
        Output parquet path. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
