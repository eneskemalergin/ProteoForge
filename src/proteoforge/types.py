"""Typed data contracts for pipeline handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import polars as pl

    from proteoforge._config import Config


@dataclass(frozen=True)
class ColumnMap:
    """Map source column names to canonical ProteoForge names."""

    protein_id: str = "protein_id"
    peptide_id: str = "peptide_id"
    sample_id: str = "sample_id"
    condition: str = "condition"
    intensity: str = "intensity"
    is_real: str = "is_real"
    is_complete_missing: str = "is_complete_missing"
    weight: str = "weight"

    def as_dict(self) -> dict[str, str]:
        """Return canonical-to-source mapping for renaming."""
        return {
            self.protein_id: "protein_id",
            self.peptide_id: "peptide_id",
            self.sample_id: "sample_id",
            self.condition: "condition",
            self.intensity: "intensity",
            self.is_real: "is_real",
            self.is_complete_missing: "is_complete_missing",
            self.weight: "weight",
        }


@dataclass(frozen=True)
class DesignTable:
    """Sample-to-condition map parsed from a design file."""

    sample_ids: tuple[str, ...]
    sample_to_condition: dict[str, str]
    condition_to_samples: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PreparedDataset:
    """
    Validated, normalized peptide data ready for downstream modeling.

    Long-format handoff in ``peptides``: one row per
    ``(protein_id, peptide_id, sample_id)`` with ``intensity_normalized``.
    See ``docs/prepared-dataset.md`` for the full contract.
    """

    config: Config
    peptides: pl.DataFrame
    sample_ids: tuple[str, ...]
    condition_levels: tuple[str, ...]
    protein_index: npt.NDArray[np.intp]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_peptides(self) -> int:
        """Number of unique (protein, peptide) pairs."""
        from proteoforge.schema import PEPTIDE_ID, PROTEIN_ID

        return self.peptides.select([PROTEIN_ID, PEPTIDE_ID]).n_unique()

    @property
    def n_samples(self) -> int:
        """Number of samples in the design."""
        return len(self.sample_ids)

    @property
    def n_proteins(self) -> int:
        """Number of distinct proteins."""
        return int(self.metadata.get("n_proteins", 0))

    @property
    def intensity_normalized(self) -> npt.NDArray[np.float64]:
        """Normalized intensities as a 1-D array aligned to ``peptides`` rows."""
        from proteoforge._normalize import NORMALIZED_INTENSITY

        return (
            self.peptides.get_column(NORMALIZED_INTENSITY)
            .to_numpy()
            .astype("float64", copy=False)
        )

    @property
    def is_real(self) -> npt.NDArray[np.bool_] | None:
        from proteoforge.schema import IS_REAL

        if self.config.model not in {"wls", "ebayes"}:
            return None
        if IS_REAL not in self.peptides.columns:
            return None
        return self.peptides.get_column(IS_REAL).to_numpy().astype(bool, copy=False)

    @property
    def is_complete_missing(self) -> npt.NDArray[np.bool_] | None:
        from proteoforge.schema import IS_COMPLETE_MISSING

        if self.config.model not in {"wls", "ebayes"}:
            return None
        if IS_COMPLETE_MISSING not in self.peptides.columns:
            return None
        return (
            self.peptides.get_column(IS_COMPLETE_MISSING)
            .to_numpy()
            .astype(bool, copy=False)
        )

    @property
    def weight(self) -> npt.NDArray[np.float64] | None:
        from proteoforge.schema import WEIGHT

        if self.config.model not in {"wls", "ebayes"}:
            return None
        if WEIGHT not in self.peptides.columns:
            return None
        return (
            self.peptides.get_column(WEIGHT)
            .to_numpy()
            .astype(np.float64, copy=False)
        )
