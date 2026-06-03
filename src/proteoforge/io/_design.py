"""Design file I/O."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from proteoforge._exceptions import ProteoForgeIOError, ProteoForgeValidationError
from proteoforge.schema import DESIGN_CONDITION, DESIGN_SAMPLE_ID
from proteoforge.types import DesignTable


def read_design(path: str | Path) -> DesignTable:
    """
    Load a sample-to-condition design table.

    .. deprecated::
        Experimental design belongs in ``Config.conditions``. Prefer
        :meth:`proteoforge.Config.from_yaml_path` or pass ``conditions`` when
        constructing :class:`proteoforge.Config`. :func:`design_from_frame` remains
        available for tests and ad hoc tables.

    Parameters
    ----------
    path
        CSV or TSV with ``sample_id`` and ``condition`` columns.

    Returns
    -------
    DesignTable
        Parsed design with sample and condition lookups.

    Raises
    ------
    ProteoForgeIOError
        If the file cannot be read or parsed.
    ProteoForgeValidationError
        If required columns are missing or sample IDs are duplicated.
    """
    warnings.warn(
        "read_design() is deprecated. Put sample-to-condition mapping in "
        "Config.conditions (YAML or constructor) instead of a separate design file.",
        DeprecationWarning,
        stacklevel=2,
    )
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"Design file not found: {file_path}"
        raise ProteoForgeIOError(msg)

    suffix = file_path.suffix.lower()
    if suffix in {".csv"}:
        frame = pl.read_csv(file_path)
    elif suffix in {".tsv", ".txt"}:
        frame = pl.read_csv(file_path, separator="\t")
    else:
        msg = f"Unsupported design file extension '{suffix}'. Use .csv or .tsv."
        raise ProteoForgeIOError(msg)

    return design_from_frame(frame)


def design_from_frame(frame: pl.DataFrame) -> DesignTable:
    """
    Build a ``DesignTable`` from a Polars frame.

    Parameters
    ----------
    frame
        DataFrame with ``sample_id`` and ``condition`` columns.

    Returns
    -------
    DesignTable
        Parsed design metadata.

    Raises
    ------
    ProteoForgeValidationError
        If required columns are missing, the table is empty, or sample IDs
        are duplicated.
    """
    required = {DESIGN_SAMPLE_ID, DESIGN_CONDITION}
    missing = required - set(frame.columns)
    if missing:
        msg = (
            f"Design table missing columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}."
        )
        raise ProteoForgeValidationError(msg)

    if frame.is_empty():
        msg = "Design table is empty."
        raise ProteoForgeValidationError(msg)

    sample_series = frame.get_column(DESIGN_SAMPLE_ID).cast(pl.String)
    if sample_series.null_count() > 0:
        msg = "Design table contains null sample_id values."
        raise ProteoForgeValidationError(msg)

    sample_ids_list = sample_series.to_list()
    if len(sample_ids_list) != len(set(sample_ids_list)):
        duplicates = _find_duplicates(sample_ids_list)
        msg = (
            f"Duplicate sample_id values in design: {duplicates[:5]}. "
            "Each sample must appear once."
        )
        raise ProteoForgeValidationError(msg)

    condition_series = frame.get_column(DESIGN_CONDITION).cast(pl.String)
    if condition_series.null_count() > 0:
        msg = "Design table contains null condition values."
        raise ProteoForgeValidationError(msg)

    sample_to_condition = {
        sample: condition
        for sample, condition in zip(
            sample_ids_list,
            condition_series.to_list(),
            strict=True,
        )
    }

    condition_to_samples: dict[str, list[str]] = {}
    for sample, condition in sample_to_condition.items():
        condition_to_samples.setdefault(condition, []).append(sample)

    ordered_samples = tuple(sample_ids_list)
    condition_to_samples_tuple = {
        condition: tuple(samples) for condition, samples in condition_to_samples.items()
    }

    return DesignTable(
        sample_ids=ordered_samples,
        sample_to_condition=sample_to_condition,
        condition_to_samples=condition_to_samples_tuple,
    )


def attach_conditions(
    peptides: pl.DataFrame,
    design: DesignTable,
    *,
    sample_col: str = "sample_id",
    condition_col: str = "condition",
) -> pl.DataFrame:
    """
    Attach condition labels from a design onto a peptide table.

    Parameters
    ----------
    peptides
        Long-format peptide intensities with a sample column.
    design
        Parsed experimental design.
    sample_col
        Sample identifier column in ``peptides``.
    condition_col
        Output condition column name.

    Returns
    -------
    polars.DataFrame
        Peptide table with condition labels attached.
    """
    return peptides.with_columns(
        pl.col(sample_col).replace(design.sample_to_condition).alias(condition_col)
    )


def _find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
