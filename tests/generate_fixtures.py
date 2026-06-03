"""Generate committed test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    design = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "condition": ["control", "control", "treated", "treated"],
        }
    )
    design.write_csv(FIXTURES / "minimal_design.csv")

    config_text = """\
control_condition: control

conditions:
  control:
    - S1
    - S2
  treated:
    - S3
    - S4

min_peptides: 4
input_is_log2: false

column_map:
  protein_id: protein_id
  peptide_id: peptide_id
  sample_id: sample_id
  intensity: intensity
"""
    (FIXTURES / "minimal_config.yaml").write_text(config_text, encoding="utf-8")

    rng = np.random.default_rng(42)
    proteins = ["P11111", "P22222"]
    peptides = [f"PEP{i}" for i in range(1, 5)]
    samples = ["S1", "S2", "S3", "S4"]
    rows: list[dict[str, object]] = []
    base = 1000.0
    for protein in proteins:
        for peptide in peptides:
            for sample_idx, sample in enumerate(samples):
                intensity = base + rng.uniform(-50, 50) + sample_idx * 25
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": peptide,
                        "sample_id": sample,
                        "intensity": float(intensity),
                    }
                )
                base += 1.5

    peptides_frame = pl.DataFrame(rows)
    peptides_frame.write_parquet(FIXTURES / "minimal_long.parquet")

    mapped = peptides_frame.rename(
        {
            "protein_id": "prot",
            "peptide_id": "pep",
            "sample_id": "run",
            "intensity": "quant",
        }
    )
    mapped.write_csv(FIXTURES / "minimal_mapped.csv")

    fasta_path = FIXTURES / "minimal.fasta"
    fasta_path.write_text(
        ">sp|P11111|TEST1\nACDEFGHIK\n>sp|P22222|TEST2\nLMNPQRSTV\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
