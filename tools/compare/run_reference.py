#!/usr/bin/env python3
"""Reference normalize pipeline (``against_condition``) for comparison."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare._shared import (
    OUTPUT_NAME,
    load_config_dict,
    read_scoped_reference,
    write_normalized_parquet,
)

REF_NORMALIZE = ROOT / "ref" / "ProteoForge_analysis_src" / "normalize.py"


def _load_against_condition() -> Callable[..., Any]:
    if not REF_NORMALIZE.is_file():
        msg = f"Reference module not found: {REF_NORMALIZE}"
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("ref_normalize", REF_NORMALIZE)
    if spec is None or spec.loader is None:
        msg = f"Could not import {REF_NORMALIZE}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.against_condition


def run(input_path: Path, config_path: Path, output_path: Path) -> None:
    """
    Run reference ``against_condition`` and write normalized long parquet.

    Parameters
    ----------
    input_path
        Peptide parquet input.
    config_path
        Pipeline config YAML.
    output_path
        Destination parquet path.

    Raises
    ------
    FileNotFoundError
        If the reference module is not checked out under ``ref/``.
    """
    against_condition = _load_against_condition()
    config_dict = load_config_dict(config_path)
    scoped = read_scoped_reference(input_path, config_path)

    conditions = config_dict["conditions"]
    if not isinstance(conditions, dict):
        msg = "config.conditions must be a mapping."
        raise TypeError(msg)
    cond_run_dict = {str(k): [str(s) for s in v] for k, v in conditions.items()}

    pdf = pd.DataFrame(scoped.to_dict(as_series=False))
    result = against_condition(
        pdf,
        cond_run_dict=cond_run_dict,
        run_col="sample_id",
        index_cols=["protein_id", "peptide_id"],
        norm_against=str(config_dict["control_condition"]),
        intensity_col="intensity",
        is_log2=bool(config_dict.get("input_is_log2", False)),
        norm_intensity_col="intensity_normalized",
    )

    out = pl.from_dict(
        {
            "protein_id": result["protein_id"].tolist(),
            "peptide_id": result["peptide_id"].tolist(),
            "sample_id": result["sample_id"].tolist(),
            "intensity_normalized": result["intensity_normalized"].tolist(),
        }
    )
    write_normalized_parquet(out, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"Normalized long parquet (default name: {OUTPUT_NAME}).",
    )
    args = parser.parse_args()
    run(args.input, args.config, args.output)


if __name__ == "__main__":
    main()
