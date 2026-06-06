"""
Per-protein long fitting blocks.

Turns the long ``PreparedDataset.peptides`` table into the arrays the
one-vs-rest model fits on: a response vector, integer condition and peptide
codes, and optional per-row WLS weights. Each block holds the full long layout
of one protein (every peptide-by-sample measurement is a row), keyed by
``(peptide_id, sample_id)`` in a stable order. This is the fitting substrate;
no wide pivot happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge._normalize import NORMALIZED_INTENSITY
from proteoforge._weights import row_weights
from proteoforge.schema import CONDITION, PEPTIDE_ID, PROTEIN_ID, SAMPLE_ID

if TYPE_CHECKING:
    from proteoforge.types import PreparedDataset

_CONDITION_CODE = "_condition_code"
_PEPTIDE_CODE = "_peptide_code"
_SAMPLE_ORDER = "_sample_order"
_ROW_WEIGHT = "_row_weight"


@dataclass(frozen=True)
class ProteinBlock:
    """
    Long fitting block for a single protein.

    Attributes
    ----------
    protein_id
        Protein identifier.
    peptide_ids
        Peptide identifiers in code order, length ``n_peptides``.
    response
        Normalized intensities, shape ``(n_obs,)``.
    condition_code
        Condition index per row (control is 0), shape ``(n_obs,)``.
    peptide_code
        Peptide index per row, ``0..n_peptides-1``, shape ``(n_obs,)``.
    weight
        Per-row WLS weights, shape ``(n_obs,)``, or ``None`` under RLM.
    n_conditions
        Number of conditions in the design.
    """

    protein_id: str
    peptide_ids: tuple[str, ...]
    response: npt.NDArray[np.float64]
    condition_code: npt.NDArray[np.intp]
    peptide_code: npt.NDArray[np.intp]
    weight: npt.NDArray[np.float64] | None
    n_conditions: int

    @property
    def n_obs(self) -> int:
        """Number of observations (rows) in the block."""
        return int(self.response.size)

    @property
    def n_peptides(self) -> int:
        """Number of distinct peptides on the protein."""
        return len(self.peptide_ids)


def build_protein_blocks(prepared: PreparedDataset) -> list[ProteinBlock]:
    """
    Build per-protein long blocks from a prepared dataset.

    Parameters
    ----------
    prepared
        Validated, normalized dataset from :func:`proteoforge.prepare`.

    Returns
    -------
    list of ProteinBlock
        Blocks in ascending protein-id order, peptides in ascending id order.

    Raises
    ------
    ProteoForgeValidationError
        If the normalized response contains non-finite values, which violates
        the prepare contract that intensities are imputed and complete.
    """
    levels = prepared.condition_levels
    condition_map = {name: index for index, name in enumerate(levels)}
    sample_map = {name: index for index, name in enumerate(prepared.sample_ids)}

    work = prepared.peptides.with_columns(
        pl.col(CONDITION)
        .replace_strict(condition_map, return_dtype=pl.Int64)
        .alias(_CONDITION_CODE),
        pl.col(SAMPLE_ID)
        .replace_strict(sample_map, return_dtype=pl.Int64)
        .alias(_SAMPLE_ORDER),
    )
    work = work.sort([PROTEIN_ID, PEPTIDE_ID, _SAMPLE_ORDER])
    work = work.with_columns(
        (pl.col(PEPTIDE_ID).rank("dense").over(PROTEIN_ID).cast(pl.Int64) - 1).alias(
            _PEPTIDE_CODE
        )
    )

    weights = row_weights(work, prepared.config)
    if weights is not None:
        work = work.with_columns(pl.Series(_ROW_WEIGHT, weights))

    n_conditions = len(levels)
    blocks: list[ProteinBlock] = []
    for sub in work.partition_by(PROTEIN_ID, maintain_order=True):
        response = sub.get_column(NORMALIZED_INTENSITY).to_numpy().astype(np.float64)
        if not np.all(np.isfinite(response)):
            protein = sub.get_column(PROTEIN_ID).item(0)
            msg = (
                f"Protein '{protein}' has non-finite normalized intensities. "
                "Prepare must deliver imputed, complete intensities."
            )
            raise ProteoForgeValidationError(msg)

        block_weight = (
            sub.get_column(_ROW_WEIGHT).to_numpy().astype(np.float64)
            if weights is not None
            else None
        )
        blocks.append(
            ProteinBlock(
                protein_id=str(sub.get_column(PROTEIN_ID).item(0)),
                peptide_ids=tuple(
                    sub.get_column(PEPTIDE_ID).unique(maintain_order=True)
                ),
                response=response,
                condition_code=sub.get_column(_CONDITION_CODE)
                .to_numpy()
                .astype(np.intp),
                peptide_code=sub.get_column(_PEPTIDE_CODE).to_numpy().astype(np.intp),
                weight=block_weight,
                n_conditions=n_conditions,
            )
        )
    return blocks
