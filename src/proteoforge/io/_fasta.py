"""Minimal FASTA loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from proteoforge._exceptions import ProteoForgeIOError

if TYPE_CHECKING:
    from collections.abc import Iterator


def read_fasta(path: str | Path) -> pl.DataFrame:
    """
    Load a FASTA file into a Polars DataFrame.

    Parses headers and sequences only. Full UniProt annotation parsing is not
    implemented in v0.0.2.

    Parameters
    ----------
    path
        Path to a FASTA file.

    Returns
    -------
    polars.DataFrame
        Table with ``entry`` (header without ``>``) and ``sequence`` columns.

    Raises
    ------
    ProteoForgeIOError
        If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        msg = f"FASTA file not found: {file_path}"
        raise ProteoForgeIOError(msg)

    entries: list[str] = []
    sequences: list[str] = []
    for header, sequence in _iter_fasta_entries(file_path):
        entries.append(header)
        sequences.append(sequence)

    return pl.DataFrame({"entry": entries, "sequence": sequences})


def _iter_fasta_entries(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence_parts: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_parts)
                header = line[1:].strip()
                sequence_parts = []
            elif header is None:
                msg = f"FASTA sequence line before header in {path}"
                raise ProteoForgeIOError(msg)
            else:
                sequence_parts.append(line)

    if header is not None:
        yield header, "".join(sequence_parts)
