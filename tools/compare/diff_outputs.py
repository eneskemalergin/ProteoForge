#!/usr/bin/env python3
"""
Compare normalized long parquet outputs with strict per-key parity.

Each observation is matched on ``(protein_id, peptide_id, sample_id)`` via an
inner 1:1 join. Values are compared element-wise for that key — not via row
position after sort, and not via aggregate statistics (mean/median/sum) alone.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ["protein_id", "peptide_id", "sample_id"]
VALUE = "intensity_normalized"
PF_VALUE = f"{VALUE}_pf"


@dataclass(frozen=True)
class ParityReport:
    """Result of a strict key-aligned parity check."""

    n_rows_compared: int
    n_within_tolerance: int
    n_mismatch: int
    max_abs_diff: float
    mean_abs_diff: float
    rtol: float
    atol: float

    @property
    def passed(self) -> bool:
        return self.n_mismatch == 0

    def format_summary(self) -> str:
        lines = [
            f"Keys compared (1:1):  {self.n_rows_compared:,}",
            f"Within tolerance:      {self.n_within_tolerance:,}",
            f"Mismatched keys:       {self.n_mismatch:,}",
            f"Max abs diff:          {self.max_abs_diff:.3e}",
            f"Mean abs diff:         {self.mean_abs_diff:.3e}",
            f"Tolerance:             rtol={self.rtol:g}, atol={self.atol:g}",
        ]
        return "\n".join(lines)


def _load_values(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = [column for column in [*KEYS, VALUE] if column not in frame.columns]
    if missing:
        msg = f"{path} missing columns: {missing}"
        raise ValueError(msg)
    return frame.select([*KEYS, VALUE])


def _assert_unique_keys(frame: pl.DataFrame, *, label: str) -> None:
    n_rows = frame.height
    n_unique = frame.select(KEYS).n_unique()
    if n_rows == n_unique:
        return
    duplicates = (
        frame.group_by(KEYS)
        .len()
        .filter(pl.col("len") > 1)
        .sort("len", descending=True)
        .head(5)
    )
    examples = duplicates.select(KEYS).rows()
    msg = (
        f"{label} has duplicate primary keys: {n_rows - n_unique} extra row(s) "
        f"({n_rows:,} rows, {n_unique:,} unique keys). Examples: {examples}"
    )
    raise ValueError(msg)


def _key_only(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(KEYS)


def compare_parity(
    reference: Path | pl.DataFrame,
    proteoforge: Path | pl.DataFrame,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_examples: int = 10,
) -> ParityReport:
    """
    Compare ``intensity_normalized`` per ``(protein_id, peptide_id, sample_id)``.

    Raises
    ------
    ValueError
        Duplicate keys, unequal key sets, or join cardinality violations.
    """
    ref = _load_values(reference) if isinstance(reference, Path) else reference
    pf = _load_values(proteoforge) if isinstance(proteoforge, Path) else proteoforge

    _assert_unique_keys(ref, label="reference")
    _assert_unique_keys(pf, label="proteoforge")

    ref_keys = _key_only(ref)
    pf_keys = _key_only(pf)

    ref_only = ref_keys.join(pf_keys, on=KEYS, how="anti")
    pf_only = pf_keys.join(ref_keys, on=KEYS, how="anti")
    if ref_only.height > 0 or pf_only.height > 0:
        ref_preview = ref_only.head(max_examples).rows()
        pf_preview = pf_only.head(max_examples).rows()
        msg = (
            "Primary key sets differ. "
            f"only_in_reference={ref_only.height:,}, "
            f"only_in_proteoforge={pf_only.height:,}. "
            f"Examples only_in_reference: {ref_preview}. "
            f"Examples only_in_proteoforge: {pf_preview}."
        )
        raise ValueError(msg)

    merged = ref.join(
        pf.rename({VALUE: PF_VALUE}),
        on=KEYS,
        how="inner",
        validate="1:1",
    )
    if merged.height != ref.height:
        msg = (
            f"Inner join row count mismatch after unique keys: "
            f"reference={ref.height:,}, joined={merged.height:,}."
        )
        raise ValueError(msg)

    ref_vals = merged.get_column(VALUE).to_numpy()
    pf_vals = merged.get_column(PF_VALUE).to_numpy()
    diffs = np.abs(ref_vals - pf_vals)
    within = np.isclose(ref_vals, pf_vals, rtol=rtol, atol=atol, equal_nan=True)
    n_mismatch = int((~within).sum())

    report = ParityReport(
        n_rows_compared=merged.height,
        n_within_tolerance=int(within.sum()),
        n_mismatch=n_mismatch,
        max_abs_diff=float(diffs.max()) if diffs.size else 0.0,
        mean_abs_diff=float(diffs.mean()) if diffs.size else 0.0,
        rtol=rtol,
        atol=atol,
    )

    if n_mismatch > 0:
        worst = (
            merged.with_columns(pl.lit(diffs).alias("_abs_diff"))
            .filter(~pl.Series(within))
            .sort("_abs_diff", descending=True)
            .head(max_examples)
            .select([*KEYS, VALUE, PF_VALUE, "_abs_diff"])
        )
        msg = (
            f"{n_mismatch:,} key(s) differ beyond tolerance "
            f"(rtol={rtol:g}, atol={atol:g}). "
            f"Worst examples (reference vs proteoforge):\n{worst}"
        )
        raise AssertionError(msg)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference output parquet.")
    parser.add_argument("proteoforge", type=Path, help="ProteoForge output parquet.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance per matched key (default: 1e-10).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance per matched key (default: 1e-12).",
    )
    args = parser.parse_args()

    try:
        report = compare_parity(
            args.reference,
            args.proteoforge,
            rtol=args.rtol,
            atol=args.atol,
        )
    except (ValueError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(report.format_summary())
    print("PASS: every (protein_id, peptide_id, sample_id) matches within tolerance.")


if __name__ == "__main__":
    main()
