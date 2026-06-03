"""Self-tests for strict key-aligned parity comparison (dev-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare.diff_outputs import compare_parity


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_parity_passes_on_identical_keys_and_values(tmp_path: Path) -> None:
    data = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 0.1,
            },
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S2",
                "intensity_normalized": -0.2,
            },
        ]
    )
    ref = tmp_path / "ref.parquet"
    pf = tmp_path / "pf.parquet"
    data.write_parquet(ref)
    data.write_parquet(pf)

    report = compare_parity(ref, pf)
    assert report.passed
    assert report.n_rows_compared == 2
    assert report.n_mismatch == 0


def test_parity_fails_when_values_swapped_between_keys(tmp_path: Path) -> None:
    ref = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S2",
                "intensity_normalized": 2.0,
            },
        ]
    )
    pf = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 2.0,
            },
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S2",
                "intensity_normalized": 1.0,
            },
        ]
    )
    ref_path = tmp_path / "ref.parquet"
    pf_path = tmp_path / "pf.parquet"
    ref.write_parquet(ref_path)
    pf.write_parquet(pf_path)

    with pytest.raises(AssertionError, match="key\\(s\\) differ"):
        compare_parity(ref_path, pf_path)


def test_parity_does_not_pass_on_matching_mean_only(tmp_path: Path) -> None:
    """Same multiset of values on different keys must fail."""
    ref = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
            {
                "protein_id": "P1",
                "peptide_id": "B",
                "sample_id": "S1",
                "intensity_normalized": 3.0,
            },
        ]
    )
    pf = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 3.0,
            },
            {
                "protein_id": "P1",
                "peptide_id": "B",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
        ]
    )
    ref_path = tmp_path / "ref.parquet"
    pf_path = tmp_path / "pf.parquet"
    ref.write_parquet(ref_path)
    pf.write_parquet(pf_path)

    assert ref["intensity_normalized"].mean() == pf["intensity_normalized"].mean()
    with pytest.raises(AssertionError, match="key\\(s\\) differ"):
        compare_parity(ref_path, pf_path)


def test_parity_fails_on_missing_keys(tmp_path: Path) -> None:
    ref = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
        ]
    )
    pf = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "B",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
        ]
    )
    ref_path = tmp_path / "ref.parquet"
    pf_path = tmp_path / "pf.parquet"
    ref.write_parquet(ref_path)
    pf.write_parquet(pf_path)

    with pytest.raises(ValueError, match="Primary key sets differ"):
        compare_parity(ref_path, pf_path)


def test_parity_fails_on_duplicate_keys(tmp_path: Path) -> None:
    dup = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.1,
            },
        ]
    )
    ok = _frame(
        [
            {
                "protein_id": "P1",
                "peptide_id": "A",
                "sample_id": "S1",
                "intensity_normalized": 1.0,
            },
        ]
    )
    ref_path = tmp_path / "ref.parquet"
    pf_path = tmp_path / "pf.parquet"
    dup.write_parquet(ref_path)
    ok.write_parquet(pf_path)

    with pytest.raises(ValueError, match="duplicate primary keys"):
        compare_parity(ref_path, pf_path)
