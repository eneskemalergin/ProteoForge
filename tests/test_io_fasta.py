"""Tests for FASTA I/O."""

from __future__ import annotations

from proteoforge.io import read_fasta


def test_read_fasta_fixture(fixtures_dir) -> None:
    frame = read_fasta(fixtures_dir / "minimal.fasta")
    assert frame.height == 2
    assert frame.get_column("sequence").to_list() == ["ACDEFGHIK", "LMNPQRSTV"]
