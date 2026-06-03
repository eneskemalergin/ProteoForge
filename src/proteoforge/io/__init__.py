"""
Public I/O entry points for peptide tables, provenance, design, and FASTA.
"""

from __future__ import annotations

from proteoforge.io._design import attach_conditions, design_from_frame, read_design
from proteoforge.io._fasta import read_fasta
from proteoforge.io._ingest import materialize_peptide_table
from proteoforge.io._peptides import peptides_from_frame, read_peptides
from proteoforge.io._provenance import attach_provenance, read_provenance

__all__ = [
    "attach_conditions",
    "attach_provenance",
    "design_from_frame",
    "materialize_peptide_table",
    "peptides_from_frame",
    "read_design",
    "read_fasta",
    "read_peptides",
    "read_provenance",
]
