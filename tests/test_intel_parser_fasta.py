"""Tests for :mod:`proteoforge.intel.parser.fasta`."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

from proteoforge.intel.parser import fasta


@pytest.fixture
def fasta_path(tmp_path: Path) -> Path:
    return tmp_path / "sample.fasta"


_VALID_LETTERS = tuple(fasta.AMINO_ACID_WEIGHTS)


def _uniprot_header(
    accession: str,
    entry_name: str,
    *,
    description: str = "",
    organism: str = "Homo sapiens",
    taxonomy_id: int = 9606,
    gene: str | None = None,
    reviewed: bool = True,
    protein_existence: int = 1,
    sequence_version: int = 1,
) -> str:
    db = "sp" if reviewed else "tr"
    tail = description.strip()
    if tail:
        tail = f"{tail} "
    tail += f"OS={organism} OX={taxonomy_id}"
    if gene is not None:
        tail += f" GN={gene}"
    tail += f" PE={protein_existence} SV={sequence_version}"
    return f"{db}|{accession}|{entry_name} {tail}"


def _write_fasta(
    path: Path,
    entries: list[tuple[str, str | list[str]]],
    *,
    line_ending: str = "\n",
    trailing_newline: bool = True,
) -> None:
    chunks: list[str] = []
    for header, sequence in entries:
        chunks.append(f">{header}")
        if isinstance(sequence, str):
            chunks.append(sequence)
        else:
            chunks.extend(sequence)
    body = line_ending.join(chunks)
    if not trailing_newline:
        if line_ending == "\r\n":
            path.write_bytes(body.encode("utf-8"))
        else:
            path.write_text(body, encoding="utf-8")
        return
    if line_ending == "\r\n":
        path.write_bytes((body + "\r\n").encode("utf-8"))
    else:
        path.write_text(body + "\n", encoding="utf-8")


def _expected_mw_kda(sequence: str) -> float:
    weight = fasta.WATER_WEIGHT
    for letter in sequence.upper():
        weight += fasta.AMINO_ACID_WEIGHTS[letter]
    return weight / 1000.0


_RANDOM_SEQUENCES = tuple(
    "".join(random.Random(20260615 + i).choice(_VALID_LETTERS) for _ in range(length))
    for i, length in enumerate((7, 15, 31, 63, 127))
)


class TestSchema:
    def test_column_schema_derives_available_and_default_columns(self) -> None:
        assert fasta.AVAILABLE_COLUMNS == list(fasta.COLUMN_SCHEMA)
        assert set(fasta.DEFAULT_COLUMNS) <= set(fasta.AVAILABLE_COLUMNS)
        assert "organism" not in fasta.DEFAULT_COLUMNS
        assert "sequence" in fasta.DEFAULT_COLUMNS


class TestFastaContract:
    def test_output_columns_and_dtypes_match_contract(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "AAA1_HUMAN", gene="AAA1"), "ACDEFGHI")],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame.columns == fasta.DEFAULT_COLUMNS
        assert frame.schema == {
            col: fasta.COLUMN_SCHEMA[col] for col in fasta.DEFAULT_COLUMNS
        }

    def test_all_entries_filtered_returns_typed_empty_frame(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P11111", "SHORT_HUMAN"), "ACDEF"),
                (_uniprot_header("P22222", "BAD_HUMAN", gene="G1"), "ACDEF*HI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(fasta_path, min_length=7)
        assert frame.height == 0
        assert frame.columns == fasta.DEFAULT_COLUMNS
        assert frame.schema["sequenceLength"] == pl.Int64
        assert frame.schema["molecularWeight_kDa"] == pl.Float64

    def test_return_result_reports_processed_and_skip_totals(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P11111", "SHORT_HUMAN"), "ACDEF"),
                (_uniprot_header("P22222", "GOOD_HUMAN", gene="G2"), "ACDEFGHI"),
            ],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.processed == result.dataframe.height == 1
        assert result.skipped.length_filter == 1
        assert result.skipped.invalid_sequence == 0
        assert result.skipped.gene_filter == 0
        assert result.skipped.malformed_header == 0
        assert result.skipped.total() == 1
        assert result.elapsed_s >= 0.0

    def test_entry_conservation_processed_plus_skips_equals_input_records(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [
                ("bad_header", "ACDEFGHI"),
                (_uniprot_header("P11111", "SHORT_HUMAN"), "ACDEF"),
                (_uniprot_header("P22222", "NOGENE_HUMAN"), "ACDEFGHI"),
                (_uniprot_header("P33333", "BAD_HUMAN", gene="B"), "ACDEF*HI"),
                (_uniprot_header("P44444", "GOOD_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        input_records = len(list(fasta.read_fasta_entries(fasta_path)))
        result = fasta.fasta_to_dataframe(
            fasta_path,
            gene_only=True,
            return_result=True,
        )
        assert input_records == 5
        assert result.processed == result.dataframe.height == 1
        assert result.processed + result.skipped.total() == input_records

    def test_committed_minimal_fixture_parses_uniprot_fields(
        self,
        fixtures_dir: Path,
    ) -> None:
        frame = fasta.fasta_to_dataframe(
            fixtures_dir / "minimal.fasta",
            column_order=[
                "entry",
                "entryName",
                "geneName",
                "organism",
                "molecularWeight_kDa",
                "sequence",
            ],
        ).sort("entry")
        assert frame.height == 2
        assert frame["entry"].to_list() == ["P11111", "P22222"]
        assert frame["entryName"].to_list() == ["AAA1_HUMAN", "BBB2_HUMAN"]
        assert frame["geneName"].to_list() == ["AAA1", "BBB2"]
        assert frame["organism"].to_list() == ["Homo sapiens", "Homo sapiens"]
        assert frame["sequence"].to_list() == ["ACDEFGHIK", "LMNPQRSTV"]
        assert frame["molecularWeight_kDa"].to_list() == pytest.approx(
            [_expected_mw_kda(s) for s in ("ACDEFGHIK", "LMNPQRSTV")],
            rel=0,
            abs=1e-9,
        )

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="FASTA file not found"):
            fasta.fasta_to_dataframe("/no/such/proteome.fasta")

    def test_directory_path_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Path is not a file"):
            fasta.fasta_to_dataframe(tmp_path)

    def test_custom_column_order_and_sort(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P22222", "B_HUMAN", gene="B"), "ACDEFGHI"),
                (_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(
            fasta_path,
            column_order=["entry", "geneName", "sequence"],
            sort_by=["geneName"],
            sort_ascending=[False],
        )
        assert frame.columns == ["entry", "geneName", "sequence"]
        assert frame["geneName"].to_list() == ["B", "A"]

    def test_sort_by_column_omitted_from_column_order_still_applies(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P22222", "B_HUMAN", gene="B"), "ACDEFGHIJK"),
                (_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(
            fasta_path,
            column_order=["entry", "geneName"],
            sort_by=["sequenceLength"],
            sort_ascending=[False],
        )
        assert frame.columns == ["entry", "geneName"]
        assert frame["entry"].to_list() == ["P22222", "P11111"]

    def test_short_sort_ascending_defaults_missing_directions_to_ascending(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P22222", "B_HUMAN", gene="B"), "ACDEFGHI"),
                (_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(
            fasta_path,
            sort_by=["entry", "geneName"],
            sort_ascending=[False],
        )
        assert frame["entry"].to_list() == ["P22222", "P11111"]

    def test_column_order_can_include_extended_header_fields(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI")],
        )
        frame = fasta.fasta_to_dataframe(
            fasta_path,
            column_order=[
                "entry",
                "organism",
                "taxonomyId",
                "proteinExistence",
                "sequenceVersion",
                "sequence",
            ],
        )
        assert frame.columns == [
            "entry",
            "organism",
            "taxonomyId",
            "proteinExistence",
            "sequenceVersion",
            "sequence",
        ]
        assert frame["organism"][0] == "Homo sapiens"
        assert frame["taxonomyId"][0] == 9606
        assert frame["proteinExistence"][0] == 1
        assert frame["sequenceVersion"][0] == 1

    def test_unknown_column_order_raises(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI")],
        )
        with pytest.raises(ValueError, match="Unknown column"):
            fasta.fasta_to_dataframe(
                fasta_path,
                column_order=["entry", "organims"],
            )

    def test_unknown_sort_by_column_raises(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "A_HUMAN", gene="A"), "ACDEFGHI")],
        )
        with pytest.raises(ValueError, match="Unknown sort_by column"):
            fasta.fasta_to_dataframe(
                fasta_path,
                sort_by=["entry", "entriy"],
            )

    def test_include_sequence_false_default_omits_sequence(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "NOSQ_HUMAN", gene="N"), "ACDEFGHI")],
        )
        frame = fasta.fasta_to_dataframe(fasta_path, include_sequence=False)
        assert "sequence" not in frame.columns
        assert frame.columns == [c for c in fasta.DEFAULT_COLUMNS if c != "sequence"]

    def test_include_sequence_false_strips_sequence_from_column_order(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "NOSQ_HUMAN", gene="N"), "ACDEFGHI")],
        )
        frame = fasta.fasta_to_dataframe(
            fasta_path,
            column_order=["entry", "sequence", "geneName"],
            include_sequence=False,
        )
        assert frame.columns == ["entry", "geneName"]

    def test_sort_by_empty_list_uses_default_sort(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P22222-2", "ISO2_HUMAN", gene="G"), "ACDEFGHI"),
                (_uniprot_header("P22222", "CAN_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        explicit = fasta.fasta_to_dataframe(fasta_path)
        empty_sort = fasta.fasta_to_dataframe(fasta_path, sort_by=[])
        assert empty_sort["entry"].to_list() == explicit["entry"].to_list()


class TestFastaEdge:
    @pytest.mark.parametrize(
        ("sequence", "min_length", "expect_kept"),
        [
            ("ACDEFG", 7, False),
            ("ACDEFGH", 7, True),
            ("A" * 1_000_001, 7, False),
            ("A" * 1_000_000, 7, True),
        ],
    )
    def test_length_boundaries(
        self,
        fasta_path: Path,
        sequence: str,
        min_length: int,
        expect_kept: bool,
    ) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "LEN_HUMAN", gene="L"), sequence)],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            min_length=min_length,
            max_length=10**6,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert (result.dataframe.height == 1) is expect_kept
        if not expect_kept:
            assert result.skipped.length_filter == 1

    def test_max_length_filters_long_sequences(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "LONG_HUMAN", gene="L"), "ACDEFGHIJKLMN")],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            max_length=10,
            return_result=True,
        )
        assert result.dataframe.height == 0
        assert result.skipped.length_filter == 1
        assert result.processed == 0

    def test_tab_in_sequence_line_is_invalid(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "TAB_HUMAN", gene="T")
        fasta_path.write_text(
            f">{header}\nACDE\tFGHI\n",
            encoding="utf-8",
        )
        result = fasta.fasta_to_dataframe(fasta_path, return_result=True)
        assert result.dataframe.height == 0
        assert result.skipped.invalid_sequence == 1

    def test_utf8_bom_before_header_is_stripped(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "BOM_HUMAN", gene="B")
        fasta_path.write_bytes(
            b"\xef\xbb\xbf>" + header.encode("utf-8") + b"\nACDEFGHI\n"
        )
        records = list(fasta.read_fasta_entries(fasta_path))
        assert records == [(header, "ACDEFGHI")]
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame.height == 1
        assert frame["entry"][0] == "P11111"

    def test_empty_fasta_file_raises(self, fasta_path: Path) -> None:
        fasta_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty FASTA file"):
            list(fasta.read_fasta_entries(fasta_path))

    def test_whitespace_only_lines_raise_empty_fasta(self, fasta_path: Path) -> None:
        fasta_path.write_text("\n\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty FASTA file"):
            list(fasta.read_fasta_entries(fasta_path))

    def test_whitespace_without_header_raises(self, fasta_path: Path) -> None:
        fasta_path.write_text("\n\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="before header"):
            list(fasta.read_fasta_entries(fasta_path))

    def test_headers_without_sequence_are_length_filtered(self, fasta_path: Path) -> None:
        _write_fasta(fasta_path, [(_uniprot_header("P11111", "EMPTY_HUMAN", gene="E"), "")])
        result = fasta.fasta_to_dataframe(
            fasta_path,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.dataframe.height == 0
        assert result.skipped.length_filter == 1

    def test_read_fasta_entries_yields_empty_sequence_before_length_filter(
        self,
        fasta_path: Path,
    ) -> None:
        header = _uniprot_header("P11111", "EMPTY_HUMAN", gene="E")
        _write_fasta(fasta_path, [(header, "")])
        records = list(fasta.read_fasta_entries(fasta_path))
        assert records == [(header, "")]

    def test_consecutive_headers_assign_empty_sequence_to_prior(
        self,
        fasta_path: Path,
    ) -> None:
        first = _uniprot_header("P11111", "FIRST_HUMAN", gene="F")
        second = _uniprot_header("P22222", "SECOND_HUMAN", gene="S")
        _write_fasta(
            fasta_path,
            [
                (first, ""),
                (second, "ACDEFGHI"),
            ],
        )
        records = list(fasta.read_fasta_entries(fasta_path))
        assert records[0] == (first, "")
        assert records[1] == (second, "ACDEFGHI")

    def test_sequence_before_header_raises(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "LATE_HUMAN"), "ACDEFGHI")],
        )
        text = fasta_path.read_text(encoding="utf-8")
        fasta_path.write_text("ZZZZZZZ\n" + text, encoding="utf-8")
        with pytest.raises(ValueError, match="before header"):
            list(fasta.read_fasta_entries(fasta_path))

    def test_invalid_utf8_raises_encoding_error(self, fasta_path: Path) -> None:
        fasta_path.write_bytes(b">\xff\xfe\nACDEFGHI\n")
        with pytest.raises(ValueError, match="File encoding error"):
            list(fasta.read_fasta_entries(fasta_path))

    def test_non_ascii_sequence_counted_as_invalid(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "BAD_HUMAN", gene="B"), "ACDEF\u00c9GHI")],
        )
        result = fasta.fasta_to_dataframe(fasta_path, return_result=True)
        assert result.dataframe.height == 0
        assert result.skipped.invalid_sequence == 1

    def test_malformed_headers_are_counted_not_parsed(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                ("not_a_uniprot_header", "ACDEFGHI"),
                ("sp|P11111", "ACDEFGHI"),
                ("sp||NONAME_HUMAN OS=Homo GN=G PE=1 SV=1", "ACDEFGHI"),
                (_uniprot_header("P22222", "GOOD_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.dataframe.height == 1
        assert result.dataframe["entry"][0] == "P22222"
        assert result.skipped.malformed_header == 3

    def test_invalid_amino_acids_skipped_with_explicit_reason(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P11111", "BAD_HUMAN", gene="B"), "ACDEF*HI"),
                (_uniprot_header("P22222", "GOOD_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.skipped.invalid_sequence == 1
        assert result.dataframe.height == 1

    def test_gene_only_filters_entries_without_gn(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P11111", "NOGENE_HUMAN"), "ACDEFGHI"),
                (_uniprot_header("P22222", "GENE_HUMAN", gene="EGFR"), "ACDEFGHI"),
            ],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            gene_only=True,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.dataframe.height == 1
        assert result.dataframe["geneName"][0] == "EGFR"
        assert result.skipped.gene_filter == 1

    def test_gene_regex_matches_gn_token_anywhere_in_description_tail(
        self,
        fasta_path: Path,
    ) -> None:
        # Footgun: GN= substring in free-text description is treated as the gene field.
        header = (
            "sp|P11111|DECOY_HUMAN Contains GN=FAKE text OS=Homo sapiens "
            "OX=9606 PE=1 SV=1"
        )
        _write_fasta(fasta_path, [(header, "ACDEFGHI")])
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["geneName"][0] == "FAKE"

    def test_unreviewed_tr_database_code(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("A0A024QYR6", "A0A_HUMAN", reviewed=False), "ACDEFGHI")],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["reviewStatus"][0] == "unreviewed"

    def test_duplicate_accessions_are_all_retained(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "DUP_HUMAN", gene="D")
        _write_fasta(
            fasta_path,
            [
                (header, "ACDEFGHI"),
                (header, "LMNPQRST"),
            ],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame.height == 2
        assert frame["entry"].to_list() == ["P11111", "P11111"]

    def test_multiline_and_blank_line_sequence_join(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                (
                    _uniprot_header("P11111", "WRAP_HUMAN", gene="W"),
                    ["acde", "", "fghi", "KLMN"],
                ),
            ],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["sequence"][0] == "ACDEFGHIKLMN"
        assert frame["sequenceLength"][0] == 12

    def test_sequence_line_whitespace_is_stripped(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "PAD_HUMAN", gene="P")
        _write_fasta(fasta_path, [(header, ["  ACDEF  ", "  GHI  "])])
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["sequence"][0] == "ACDEFGHI"

    @pytest.mark.parametrize("letter", ["B", "X"])
    def test_ambiguous_letters_kept_or_rejected_by_allow_ambiguous(
        self,
        fasta_path: Path,
        letter: str,
    ) -> None:
        sequence = f"ACDEF{letter}GHI"
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "AMB_HUMAN", gene="A"), sequence)],
        )
        kept = fasta.fasta_to_dataframe(fasta_path)
        assert kept.height == 1
        assert kept["molecularWeight_kDa"][0] == pytest.approx(
            _expected_mw_kda(sequence),
            rel=0,
            abs=1e-9,
        )

        rejected = fasta.fasta_to_dataframe(
            fasta_path,
            allow_ambiguous=False,
            return_result=True,
        )
        assert rejected.dataframe.height == 0
        assert rejected.skipped.invalid_sequence == 1

    def test_crlf_line_endings_parse_same_as_lf(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "CRLF_HUMAN", gene="C")
        _write_fasta(fasta_path, [(header, ["ACDE", "FGHI"])], line_ending="\r\n")
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["sequence"][0] == "ACDEFGHI"

    def test_read_fasta_entries_without_trailing_newline(self, fasta_path: Path) -> None:
        header = _uniprot_header("P11111", "EOF_HUMAN", gene="E")
        _write_fasta(
            fasta_path,
            [(header, "ACDEFGHI")],
            trailing_newline=False,
        )
        records = list(fasta.read_fasta_entries(fasta_path))
        assert records == [(header, "ACDEFGHI")]
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame.height == 1
        assert frame["sequence"][0] == "ACDEFGHI"

    def test_isoform_status_uses_hyphen_in_accession_not_entry_name(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P12345", "FOO-BAR_HUMAN", gene="F"), "ACDEFGHI"),
                (_uniprot_header("P12345-2", "FOO-BAR-2_HUMAN", gene="F"), "ACDEFGHI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        by_entry = dict(zip(frame["entry"].to_list(), frame["isoformStatus"].to_list()))
        assert by_entry["P12345"] == "canonical"
        assert by_entry["P12345-2"] == "isoform"

    def test_description_truncates_at_first_os_token(self, fasta_path: Path) -> None:
        header = (
            "sp|P11111|TRICK_HUMAN Protein OS=decoy fragment OS=Homo sapiens "
            "OX=9606 GN=TRICK PE=1 SV=1"
        )
        _write_fasta(fasta_path, [(header, "ACDEFGHI")])
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["proteinDescription"][0] == "Protein"
        assert frame["geneName"][0] == "TRICK"

    def test_mixed_skip_reasons_accounting_in_one_pass(self, fasta_path: Path) -> None:
        _write_fasta(
            fasta_path,
            [
                ("bad_header", "ACDEFGHI"),
                (_uniprot_header("P11111", "SHORT_HUMAN"), "ACDEF"),
                (_uniprot_header("P22222", "NOGENE_HUMAN"), "ACDEFGHI"),
                (_uniprot_header("P33333", "BAD_HUMAN", gene="B"), "ACDEF*HI"),
                (_uniprot_header("P44444", "GOOD_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        result = fasta.fasta_to_dataframe(
            fasta_path,
            gene_only=True,
            return_result=True,
        )
        assert isinstance(result, fasta.FastaParseResult)
        assert result.dataframe.height == 1
        assert result.skipped.malformed_header == 1
        assert result.skipped.length_filter == 1
        assert result.skipped.gene_filter == 1
        assert result.skipped.invalid_sequence == 1
        assert result.skipped.total() == 4

    def test_sort_places_canonical_before_isoform_for_same_accession_prefix(
        self,
        fasta_path: Path,
    ) -> None:
        _write_fasta(
            fasta_path,
            [
                (_uniprot_header("P22222-2", "ISO2_HUMAN", gene="G"), "ACDEFGHI"),
                (_uniprot_header("P11111", "CAN1_HUMAN", gene="G"), "ACDEFGHI"),
                (_uniprot_header("P22222", "CAN2_HUMAN", gene="G"), "ACDEFGHI"),
            ],
        )
        frame = fasta.fasta_to_dataframe(fasta_path)
        assert frame["entry"].to_list() == ["P11111", "P22222", "P22222-2"]


class TestFastaEquivalence:
    def test_parse_uniprot_header_extracts_table_fields(self) -> None:
        header = _uniprot_header(
            "P00533",
            "EGFR_HUMAN",
            description="Epidermal growth factor receptor",
            gene="EGFR",
        )
        parsed = fasta.parse_uniprot_header(header)
        assert parsed["entry"] == "P00533"
        assert parsed["entryName"] == "EGFR_HUMAN"
        assert parsed["geneName"] == "EGFR"
        assert parsed["proteinDescription"] == "Epidermal growth factor receptor"
        assert parsed["reviewStatus"] == "reviewed"

    def test_description_without_os_token(self) -> None:
        header = "sp|P11111|DESC_HUMAN Some protein GN=GENE1 PE=1 SV=1"
        parsed = fasta.parse_uniprot_header(header)
        assert parsed["proteinDescription"] == "Some protein"
        assert parsed["geneName"] == "GENE1"

    def test_pipe_in_description_does_not_split_entry_fields(self) -> None:
        header = (
            "tr|Q16408|Q16408_HUMAN UDP-N-acetylglucosamine: alpha-6-D-mannoside "
            "beta-1,6-N-acetylglucosaminyltransferase V|GlcNAc transferase V protein "
            "(Fragment) OS=Homo sapiens OX=9606 PE=4 SV=1"
        )
        parsed = fasta.parse_uniprot_header(header)
        assert parsed["entry"] == "Q16408"
        assert parsed["entryName"] == "Q16408_HUMAN"
        assert parsed["reviewStatus"] == "unreviewed"
        assert parsed["organism"] == "Homo sapiens"
        assert parsed["taxonomyId"] == 9606
        assert parsed["proteinExistence"] == 4
        assert parsed["sequenceVersion"] == 1
        assert parsed["geneName"] is None
        assert "|" in (parsed["proteinDescription"] or "")

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            (
                _uniprot_header("P12345-2", "ISO2_HUMAN", gene="G1"),
                {
                    "entry": "P12345-2",
                    "entryName": "ISO2_HUMAN",
                    "geneName": "G1",
                    "reviewStatus": "reviewed",
                },
            ),
            (
                _uniprot_header("A0A024QYR6", "A0A_HUMAN", reviewed=False),
                {
                    "entry": "A0A024QYR6",
                    "reviewStatus": "unreviewed",
                },
            ),
            (
                "sp|P11111|NONAME",
                {
                    "entry": "P11111",
                    "entryName": "NONAME",
                    "geneName": None,
                    "organism": None,
                },
            ),
            (
                (
                    "sp|P22222|TRICK_HUMAN Protein OS=decoy OS=Homo sapiens "
                    "OX=9606 GN=TRICK PE=1 SV=1"
                ),
                {
                    "entry": "P22222",
                    "proteinDescription": "Protein",
                    "geneName": "TRICK",
                    "organism": "Homo sapiens",
                    "taxonomyId": 9606,
                },
            ),
            (
                "tr|Q9Y6K9|NPAP1_HUMAN Nucleolar protein OS=Homo sapiens "
                "OX=9606 GN=NPAP1 PE=1 SV=3",
                {
                    "entry": "Q9Y6K9",
                    "reviewStatus": "unreviewed",
                    "geneName": "NPAP1",
                    "proteinExistence": 1,
                    "sequenceVersion": 3,
                },
            ),
        ],
    )
    def test_parse_uniprot_header_variants(
        self,
        header: str,
        expected: dict[str, object],
    ) -> None:
        parsed = fasta.parse_uniprot_header(header)
        for key, value in expected.items():
            assert parsed[key] == value

    @pytest.mark.parametrize("header", ["", "plain text", "sp|P1", "a|b"])
    def test_malformed_headers_return_empty_entry(self, header: str) -> None:
        parsed = fasta.parse_uniprot_header(header)
        assert parsed["entry"] is None

    @pytest.mark.parametrize(
        ("header", "field", "expected"),
        [
            (
                "sp|P11111|BAD_HUMAN OS=Homo sapiens OX=notint GN=G PE=1 SV=1",
                "taxonomyId",
                None,
            ),
            (
                "sp|P11111|BAD_HUMAN OS=Homo sapiens OX=9606 GN=G PE=x SV=1",
                "proteinExistence",
                None,
            ),
            (
                "sp|P11111|BAD_HUMAN OS=Homo sapiens OX=9606 GN=G PE=1 SV=x",
                "sequenceVersion",
                None,
            ),
        ],
    )
    def test_malformed_header_integer_tokens_keep_entry_with_none(
        self,
        header: str,
        field: str,
        expected: object,
    ) -> None:
        parsed = fasta.parse_uniprot_header(header)
        assert parsed["entry"] == "P11111"
        assert parsed["entryName"] == "BAD_HUMAN"
        assert parsed[field] is expected


class TestMolecularWeight:
    @pytest.mark.parametrize("sequence", _RANDOM_SEQUENCES)
    def test_random_sequences_match_mw_oracle(
        self,
        fasta_path: Path,
        sequence: str,
    ) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "RAND_HUMAN", gene="R"), sequence)],
        )
        frame = fasta.fasta_to_dataframe(fasta_path, min_length=1)
        assert frame.height == 1
        assert frame["sequenceLength"][0] == len(sequence)
        assert frame["molecularWeight_kDa"][0] == pytest.approx(
            _expected_mw_kda(sequence),
            rel=0,
            abs=1e-9,
        )
        assert frame["molecularWeight_kDa"][0] == pytest.approx(
            fasta.calc_molecular_weight(sequence),
            rel=0,
            abs=1e-12,
        )

    @pytest.mark.parametrize(
        "sequence",
        ["U", "O", "ACDEFGHIKLMNPQRSTVWY"],
    )
    def test_nonstandard_and_full_alphabet_sequences_match_mw_oracle(
        self,
        fasta_path: Path,
        sequence: str,
    ) -> None:
        _write_fasta(
            fasta_path,
            [(_uniprot_header("P11111", "EDGE_HUMAN", gene="E"), sequence)],
        )
        frame = fasta.fasta_to_dataframe(fasta_path, min_length=1)
        assert frame.height == 1
        assert frame["molecularWeight_kDa"][0] == pytest.approx(
            _expected_mw_kda(sequence),
            rel=0,
            abs=1e-9,
        )

    def test_calc_molecular_weight_rejects_empty_and_invalid(self) -> None:
        with pytest.raises(ValueError, match="Empty sequence"):
            fasta.calc_molecular_weight("")
        with pytest.raises(ValueError, match="Invalid amino acid"):
            fasta.calc_molecular_weight("ACDEF*")
        assert fasta.calc_molecular_weight("ACDEFZ") == pytest.approx(
            _expected_mw_kda("ACDEFZ"),
            rel=0,
            abs=1e-9,
        )

    @pytest.mark.parametrize("letter", ["B", "Z", "J", "X"])
    def test_ambiguous_letters_have_mean_masses(self, letter: str) -> None:
        seq = f"ACDEF{letter}GHI"
        assert fasta.calc_molecular_weight(seq) == pytest.approx(
            _expected_mw_kda(seq),
            rel=0,
            abs=1e-9,
        )

    def test_is_valid_sequence_rules(self) -> None:
        assert fasta.is_valid_sequence("ACDEFGHI")
        assert not fasta.is_valid_sequence("")
        assert fasta.is_valid_sequence("ACDEFX")
        assert not fasta.is_valid_sequence("ACDEFX", allow_ambiguous=False)
        assert not fasta.is_valid_sequence("ACDEF*")
        assert fasta.is_valid_sequence("acdefghi")
        assert fasta.is_valid_sequence("U")
        assert fasta.is_valid_sequence("O")
        assert fasta.is_valid_sequence("BZJX")
