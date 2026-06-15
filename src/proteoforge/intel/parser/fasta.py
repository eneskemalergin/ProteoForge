"""UniProt FASTA parsing and protein property calculation.

Diagnostics use :class:`FastaParseResult` (``return_result=True``), not stdout.
Long-running callers may add ``show_progress`` via :mod:`proteoforge._progress`
when this module is promoted into the package.
"""

from __future__ import annotations

import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

_BASE_AMINO_ACID_WEIGHTS: dict[str, float] = {
    "A": 71.03711, "C": 103.00919, "D": 115.02694, "E": 129.04259,
    "F": 147.06841, "G": 57.02146, "H": 137.05891, "I": 113.08406,
    "K": 128.09496, "L": 113.08406, "M": 131.04049, "N": 114.04293,
    "P": 97.05276, "Q": 128.05858, "R": 156.10111, "S": 87.03203,
    "T": 101.04768, "V": 99.06841, "W": 186.07931, "Y": 163.06333,
    "U": 150.948923, "O": 237.143012,
}
_STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
_AMBIGUOUS_AMINO_ACID_WEIGHTS: dict[str, float] = {
    "B": (_BASE_AMINO_ACID_WEIGHTS["D"] + _BASE_AMINO_ACID_WEIGHTS["N"]) / 2.0,
    "Z": (_BASE_AMINO_ACID_WEIGHTS["E"] + _BASE_AMINO_ACID_WEIGHTS["Q"]) / 2.0,
    "J": (_BASE_AMINO_ACID_WEIGHTS["I"] + _BASE_AMINO_ACID_WEIGHTS["L"]) / 2.0,
    "X": sum(_BASE_AMINO_ACID_WEIGHTS[aa] for aa in _STANDARD_AA) / len(_STANDARD_AA),
}
AMINO_ACID_WEIGHTS: dict[str, float] = {
    **_BASE_AMINO_ACID_WEIGHTS,
    **_AMBIGUOUS_AMINO_ACID_WEIGHTS,
}
WATER_WEIGHT: float = 18.010565
_STRICT_VALID_AA: frozenset[str] = _STANDARD_AA | frozenset("UO")
_VALID_AA: frozenset[str] = _STRICT_VALID_AA | frozenset("BZJX")

# Single schema source: column name -> Polars dtype.
COLUMN_SCHEMA: dict[str, pl.DataType] = {
    "entry": pl.Utf8,
    "entryName": pl.Utf8,
    "geneName": pl.Utf8,
    "proteinDescription": pl.Utf8,
    "reviewStatus": pl.Utf8,
    "organism": pl.Utf8,
    "taxonomyId": pl.Int64,
    "proteinExistence": pl.Int64,
    "sequenceVersion": pl.Int64,
    "isoformStatus": pl.Utf8,
    "sequenceLength": pl.Int64,
    "molecularWeight_kDa": pl.Float64,
    "sequence": pl.Utf8,
}
AVAILABLE_COLUMNS: list[str] = list(COLUMN_SCHEMA)
DEFAULT_COLUMNS: list[str] = [
    c for c in AVAILABLE_COLUMNS
    if c not in {"organism", "taxonomyId", "proteinExistence", "sequenceVersion"}
]

_HEADER_FIELDS: tuple[str, ...] = (
    "reviewStatus", "entry", "entryName", "proteinDescription", "geneName",
    "organism", "taxonomyId", "proteinExistence", "sequenceVersion",
)
_MALFORMED_HEADER: dict[str, str | int | None] = dict.fromkeys(_HEADER_FIELDS)

_HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "organism": re.compile(r"OS=([^=]+?)(?:\s+(?:OX|GN|PE|SV)=|$)"),
    "geneName": re.compile(r"GN=([^\s]+)"),
    "proteinExistence": re.compile(r"PE=(\S+)"),
    "sequenceVersion": re.compile(r"SV=(\S+)"),
    "taxonomyId": re.compile(r"OX=(\S+)"),
}
_DESC_TOKEN_BOUNDARY: re.Pattern[str] = re.compile(r"\s+(?:OS|OX|GN|PE|SV)=")


@dataclass
class FastaSkipCounts:
    """
    Per-reason skip tallies produced while parsing a FASTA file.

    Attributes
    ----------
    invalid_sequence
        Entries whose sequence contains unsupported amino acid letters.
    length_filter
        Entries outside the ``min_length`` / ``max_length`` window.
    gene_filter
        Entries skipped because ``gene_only`` is True and no ``GN=`` token exists.
    malformed_header
        Entries whose header does not match the UniProt ``db|accession|name`` shape.
    """

    invalid_sequence: int = 0
    length_filter: int = 0
    gene_filter: int = 0
    malformed_header: int = 0

    def total(self) -> int:
        """Return the sum of all skip counters."""
        return (
            self.invalid_sequence
            + self.length_filter
            + self.gene_filter
            + self.malformed_header
        )


@dataclass
class FastaParseResult:
    """
    Parsed FASTA table plus diagnostics from :func:`fasta_to_dataframe`.

    Attributes
    ----------
    dataframe
        Protein table projected to ``column_order`` (default :data:`DEFAULT_COLUMNS`).
    processed
        Number of entries retained after filtering and validation.
    skipped
        Skip counts for entries dropped during parsing.
    elapsed_s
        Wall-clock seconds for parsing, DataFrame assembly, and sorting.
    """

    dataframe: pl.DataFrame
    processed: int = 0
    skipped: FastaSkipCounts = field(default_factory=FastaSkipCounts)
    elapsed_s: float = 0.0


@dataclass
class _RowBatch:
    entry: list[str] = field(default_factory=list)
    entryName: list[str] = field(default_factory=list)
    geneName: list[str | None] = field(default_factory=list)
    proteinDescription: list[str | None] = field(default_factory=list)
    reviewStatus: list[str] = field(default_factory=list)
    organism: list[str | None] = field(default_factory=list)
    taxonomyId: list[int | None] = field(default_factory=list)
    proteinExistence: list[int | None] = field(default_factory=list)
    sequenceVersion: list[int | None] = field(default_factory=list)
    isoformStatus: list[str] = field(default_factory=list)
    sequenceLength: list[int] = field(default_factory=list)
    molecularWeight_kDa: list[float] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entry)

    def append(
        self,
        header: dict[str, str | int | None],
        sequence: str,
        molecular_weight_kda: float,
    ) -> None:
        entry = header["entry"]
        if not isinstance(entry, str):
            raise ValueError("cannot append a malformed header row")
        for name in _HEADER_FIELDS:
            getattr(self, name).append(header[name])
        self.isoformStatus.append("isoform" if "-" in entry else "canonical")
        self.sequenceLength.append(len(sequence))
        self.molecularWeight_kDa.append(molecular_weight_kda)
        self.sequence.append(sequence)

    def to_frame_data(self) -> dict[str, list]:
        return {name: getattr(self, name) for name in AVAILABLE_COLUMNS}


def _validate_fasta_path(file_path: str | Path) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return path


def read_fasta_entries(
    file_path: str | Path,
) -> Generator[tuple[str, str], None, None]:
    """
    Stream FASTA records as ``(header, sequence)`` tuples.

    Parameters
    ----------
    file_path
        Path to a FASTA file.

    Yields
    ------
    tuple[str, str]
        Header text without the leading ``>``, then the concatenated sequence in
        uppercase.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If *file_path* is not a file, the file is empty, sequence lines appear
        before the first header, or UTF-8 decoding fails.
    """
    path = _validate_fasta_path(file_path)
    try:
        with path.open(encoding="utf-8") as handle:
            header: str | None = None
            sequence_lines: list[str] = []
            line_number = 0
            for raw_line in handle:
                line_number += 1
                line = raw_line.rstrip("\n\r").removeprefix("\ufeff")
                if not line:
                    continue
                if line[0] == ">":
                    if header is not None:
                        yield header, "".join(sequence_lines)
                    header = line[1:]
                    sequence_lines = []
                elif header is not None:
                    sequence_lines.append(line.strip().upper())
                else:
                    raise ValueError(
                        f"Sequence data found before header at line {line_number}"
                    )
            if header is not None:
                yield header, "".join(sequence_lines)
            else:
                raise ValueError("Empty FASTA file")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File encoding error: {exc}") from exc


def _invalid_amino_acids(sequence: str, *, allow_ambiguous: bool = True) -> set[str]:
    if not sequence:
        return set()
    valid = _VALID_AA if allow_ambiguous else _STRICT_VALID_AA
    return set(sequence.upper()) - valid


def is_valid_sequence(
    sequence: str,
    *,
    allow_ambiguous: bool = True,
) -> bool:
    """
    Check whether a sequence contains only supported amino acid letters.

    Parameters
    ----------
    sequence
        Protein sequence to validate (uppercase from :func:`read_fasta_entries`).
    allow_ambiguous
        When True (default), accept IUPAC ambiguous codes ``B``, ``Z``, ``J``,
        and ``X`` in addition to standard and selenocysteine/pyrrolysine letters.

    Returns
    -------
    bool
        ``True`` when every character is a supported amino acid letter.
    """
    if not sequence:
        return False
    return not _invalid_amino_acids(sequence, allow_ambiguous=allow_ambiguous)


def calc_molecular_weight(sequence: str, *, allow_ambiguous: bool = True) -> float:
    """
    Calculate the monoisotopic molecular weight of a protein sequence.

    Uses ExPASy ProtParam residue masses plus one water term, returned in kDa.

    Parameters
    ----------
    sequence
        Amino acid sequence (case-insensitive).
    allow_ambiguous
        When True (default), include mean monoisotopic masses for ``B``, ``Z``,
        ``J``, and ``X``.

    Returns
    -------
    float
        Molecular weight in kilodaltons.

    Raises
    ------
    ValueError
        If *sequence* is empty or contains invalid amino acid characters.
    """
    if not sequence:
        raise ValueError("Empty sequence provided")
    seq_u = sequence.upper()
    ok, mw_kda = _validate_and_mw_kda(seq_u, allow_ambiguous=allow_ambiguous)
    if not ok:
        invalid = _invalid_amino_acids(seq_u, allow_ambiguous=allow_ambiguous)
        raise ValueError(
            f"Invalid amino acid(s) '{''.join(sorted(invalid))}' found."
        )
    return mw_kda


_WEIGHT_BY_ORD = np.zeros(128, dtype=np.float64)
for _aa, _mass in AMINO_ACID_WEIGHTS.items():
    _WEIGHT_BY_ORD[ord(_aa)] = _mass


@njit(cache=True)
def _mw_da_from_buf(
    buf: np.ndarray,
    weight_by_ord: np.ndarray,
    water_weight: float,
) -> tuple[bool, float]:
    counts = np.zeros(128, dtype=np.int64)
    for i in range(len(buf)):
        code = buf[i]
        if code >= 128 or weight_by_ord[code] == 0.0:
            return False, 0.0
        counts[code] += 1
    weight = water_weight
    for code in range(128):
        n = counts[code]
        if n:
            weight += n * weight_by_ord[code]
    return True, weight


def _validate_and_mw_kda(
    sequence: str,
    *,
    allow_ambiguous: bool = True,
) -> tuple[bool, float]:
    # allow_ambiguous=False: reject B/Z/J/X in Python (they have ord masses).
    if not sequence:
        return False, 0.0
    if not allow_ambiguous and set(sequence) - _STRICT_VALID_AA:
        return False, 0.0
    try:
        buf = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError:
        return False, 0.0
    ok, mass_da = _mw_da_from_buf(buf, _WEIGHT_BY_ORD, WATER_WEIGHT)
    return (True, mass_da / 1000.0) if ok else (False, 0.0)


_mw_da_from_buf(
    np.frombuffer(b"M", dtype=np.uint8),
    _WEIGHT_BY_ORD,
    WATER_WEIGHT,
)


def _parse_header_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _protein_description_from_desc(desc: str) -> str | None:
    # First " OS=" ends the description (decoy OS= in text is not organism).
    os_idx = desc.find(" OS=")
    if os_idx >= 0:
        return desc[:os_idx].strip() or None
    match = _DESC_TOKEN_BOUNDARY.search(desc)
    if match:
        return desc[: match.start()].strip() or None
    return desc.strip() or None


def parse_uniprot_header(
    header: str,
) -> dict[str, str | int | None]:
    """
    Parse a UniProt FASTA header into structured fields.

    Parameters
    ----------
    header
        Header string without the leading ``>`` (``sp|accession|name ...`` form).

    Returns
    -------
    dict[str, str | int | None]
        Keys: ``reviewStatus``, ``entry``, ``entryName``, ``proteinDescription``,
        ``geneName``, ``organism``, ``taxonomyId``, ``proteinExistence``,
        ``sequenceVersion``. Malformed headers set ``entry`` to ``None``.
    """
    # UniProt shape is db|accession|entry_name description... only two field
    # separators; description text may contain additional '|' (e.g. Q16408).
    parts = header.split("|", 2)
    if len(parts) < 3:
        return dict(_MALFORMED_HEADER)

    entry = parts[1].strip()
    if not entry:
        return dict(_MALFORMED_HEADER)

    name_parts = parts[2].split(" ", 1)
    entry_name = name_parts[0]
    protein_description = gene_name = organism = None
    taxonomy_id = protein_existence = sequence_version = None

    if len(name_parts) > 1:
        desc = name_parts[1]
        protein_description = _protein_description_from_desc(desc)
        if match := _HEADER_PATTERNS["organism"].search(desc):
            organism = match.group(1).strip()
        if match := _HEADER_PATTERNS["geneName"].search(desc):
            gene_name = match.group(1).strip()
        if match := _HEADER_PATTERNS["proteinExistence"].search(desc):
            protein_existence = _parse_header_int(match.group(1).strip())
        if match := _HEADER_PATTERNS["sequenceVersion"].search(desc):
            sequence_version = _parse_header_int(match.group(1).strip())
        if match := _HEADER_PATTERNS["taxonomyId"].search(desc):
            taxonomy_id = _parse_header_int(match.group(1).strip())

    return {
        "reviewStatus": "reviewed" if parts[0].strip() == "sp" else "unreviewed",
        "entry": entry,
        "entryName": entry_name,
        "proteinDescription": protein_description,
        "geneName": gene_name,
        "organism": organism,
        "taxonomyId": taxonomy_id,
        "proteinExistence": protein_existence,
        "sequenceVersion": sequence_version,
    }


def _batch_to_dataframe(
    batch: _RowBatch,
    *,
    column_order: list[str],
    sort_by: list[str],
    sort_ascending: list[bool],
) -> pl.DataFrame:
    if not batch.entry:
        return pl.DataFrame(schema={col: COLUMN_SCHEMA[col] for col in column_order})

    frame = pl.DataFrame(batch.to_frame_data())
    sort_cols = [col for col in sort_by if col in frame.columns]
    if sort_cols:
        n = len(sort_cols)
        descending = [
            not (sort_ascending[i] if i < len(sort_ascending) else True)
            for i in range(n)
        ]
        frame = frame.sort(sort_cols, descending=descending)
    return frame.select(column_order)


def fasta_to_dataframe(
    fasta_path: str | Path,
    *,
    gene_only: bool = False,
    min_length: int = 7,
    max_length: int = 10**6,
    column_order: list[str] | None = None,
    sort_by: list[str] | None = None,
    sort_ascending: list[bool] | None = None,
    allow_ambiguous: bool = True,
    include_sequence: bool = True,
    return_result: bool = False,
) -> pl.DataFrame | FastaParseResult:
    """
    Parse a UniProt FASTA file into a protein property table.

    Each entry is parsed, validated, and weighed. The returned table is a
    projection of :data:`AVAILABLE_COLUMNS`; :data:`DEFAULT_COLUMNS` is used
    when ``column_order`` is omitted.

    Parameters
    ----------
    fasta_path
        Path to a UniProt-format FASTA file.
    gene_only
        When True, keep only entries with a parsed ``GN=`` gene name.
    min_length
        Minimum sequence length in amino acids (default 7).
    max_length
        Maximum sequence length in amino acids (default 1_000_000).
    column_order
        Output columns and order. Names must be in :data:`AVAILABLE_COLUMNS`.
        Unknown names raise ``ValueError``.
    sort_by
        Columns passed to :meth:`polars.DataFrame.sort`. Defaults to
        ``["entry", "isoformStatus"]``.
    sort_ascending
        Sort direction per ``sort_by`` column. Defaults to ``[True, False]``.
    allow_ambiguous
        When True (default), keep sequences with ``B``, ``Z``, ``J``, or ``X``.
    include_sequence
        When False, omit the ``sequence`` column from the output only.
    return_result
        When True, return :class:`FastaParseResult` with skip counts and timing.

    Returns
    -------
    polars.DataFrame or FastaParseResult
        Protein table, or a result wrapper when ``return_result=True``.

    Raises
    ------
    FileNotFoundError
        If *fasta_path* does not exist.
    ValueError
        If *fasta_path* is not a file, the FASTA stream is invalid, or
        ``column_order`` or ``sort_by`` contains unknown column names.
    """
    path = _validate_fasta_path(fasta_path)
    if column_order is None:
        columns = list(
            DEFAULT_COLUMNS if include_sequence
            else [col for col in DEFAULT_COLUMNS if col != "sequence"]
        )
    else:
        columns = list(column_order)
        if not include_sequence:
            columns = [col for col in columns if col != "sequence"]

    unknown = [col for col in columns if col not in COLUMN_SCHEMA]
    if unknown:
        raise ValueError(
            f"Unknown column(s): {unknown}. "
            f"Choose from AVAILABLE_COLUMNS: {AVAILABLE_COLUMNS}"
        )

    sort_cols = list(sort_by or ["entry", "isoformStatus"])
    unknown_sort = [col for col in sort_cols if col not in COLUMN_SCHEMA]
    if unknown_sort:
        raise ValueError(
            f"Unknown sort_by column(s): {unknown_sort}. "
            f"Choose from AVAILABLE_COLUMNS: {AVAILABLE_COLUMNS}"
        )
    sort_dirs = list(sort_ascending or [True, False])
    t0 = time.perf_counter()
    batch = _RowBatch()
    skipped = FastaSkipCounts()

    for header, sequence in read_fasta_entries(path):
        if len(sequence) < min_length or len(sequence) > max_length:
            skipped.length_filter += 1
            continue

        parsed = parse_uniprot_header(header)
        if parsed["entry"] is None:
            skipped.malformed_header += 1
            continue
        if gene_only and not parsed["geneName"]:
            skipped.gene_filter += 1
            continue

        ok, mol_weight_kda = _validate_and_mw_kda(
            sequence,
            allow_ambiguous=allow_ambiguous,
        )
        if not ok:
            skipped.invalid_sequence += 1
            continue

        batch.append(parsed, sequence, mol_weight_kda)

    frame = _batch_to_dataframe(
        batch,
        column_order=columns,
        sort_by=sort_cols,
        sort_ascending=sort_dirs,
    )
    elapsed = time.perf_counter() - t0

    if return_result:
        return FastaParseResult(
            dataframe=frame,
            processed=len(batch),
            skipped=skipped,
            elapsed_s=elapsed,
        )
    return frame
