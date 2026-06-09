"""Typed data contracts for pipeline handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
    import polars as pl

    from proteoforge._config import Config


@dataclass(frozen=True)
class ColumnMap:
    """
    Map source column names in input files to canonical ProteoForge names.

    Attributes
    ----------
    protein_id, peptide_id, sample_id, condition, intensity
        Source column names. Defaults match canonical names (identity mapping).
    is_real, is_complete_missing, weight
        Optional provenance column names in the source file.
    """

    protein_id: str = "protein_id"
    peptide_id: str = "peptide_id"
    sample_id: str = "sample_id"
    condition: str = "condition"
    intensity: str = "intensity"
    is_real: str = "is_real"
    is_complete_missing: str = "is_complete_missing"
    weight: str = "weight"

    def as_dict(self) -> dict[str, str]:
        """Return source-to-canonical rename map for :func:`harmonize_columns`."""
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
    """
    Sample-to-condition map derived from ``Config.conditions``.

    Attributes
    ----------
    sample_ids
        Ordered sample identifiers.
    sample_to_condition
        Lookup from sample ID to condition label.
    condition_to_samples
        Lookup from condition label to sample ID tuple.
    """

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

    Attributes
    ----------
    config
        Frozen configuration used during prepare.
    peptides
        Long-format normalized table.
    sample_ids
        Sample IDs in condition order.
    condition_levels
        Condition names with control first.
    protein_index
        Per-row protein ordinal for grouping.
    metadata
        Run statistics and scope audit fields.
    """

    config: Config
    peptides: pl.DataFrame
    sample_ids: tuple[str, ...]
    condition_levels: tuple[str, ...]
    protein_index: npt.NDArray[np.intp]
    metadata: dict[str, object] = field(default_factory=dict)

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
        raw = self.metadata.get("n_proteins", 0)
        return raw if isinstance(raw, int) else 0

    @property
    def intensity_normalized(self) -> npt.NDArray[np.float64]:
        """Normalized intensities as a 1-D array aligned to ``peptides`` rows."""
        from proteoforge.schema import NORMALIZED_INTENSITY

        return (
            self.peptides.get_column(NORMALIZED_INTENSITY)
            .to_numpy()
            .astype("float64", copy=False)
        )

    @property
    def is_real(self) -> npt.NDArray[np.bool_] | None:
        """Measured-vs-imputed mask per row, or ``None`` for RLM."""
        from proteoforge.schema import IS_REAL

        if self.config.model not in {"wls", "ebayes"}:
            return None
        if IS_REAL not in self.peptides.columns:
            return None
        return self.peptides.get_column(IS_REAL).to_numpy().astype(bool, copy=False)

    @property
    def is_complete_missing(self) -> npt.NDArray[np.bool_] | None:
        """Condition-wide imputation flag per row, or ``None`` for RLM."""
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
        """Precomputed WLS weight per row, or ``None`` for RLM."""
        from proteoforge.schema import WEIGHT

        if self.config.model not in {"wls", "ebayes"}:
            return None
        if WEIGHT not in self.peptides.columns:
            return None
        return (
            self.peptides.get_column(WEIGHT).to_numpy().astype(np.float64, copy=False)
        )


@dataclass(frozen=True)
class DiscordanceResult:
    """
    Per-peptide discordance outcome.

    Attributes
    ----------
    config
        Frozen configuration used for the run.
    table
        Long result frame with columns ``protein_id``, ``peptide_id``,
        ``raw_p_value``, ``within_p_value``, ``adjusted_p_value``,
        ``is_discordant``, and ``fit_status``.
    metadata
        Run statistics: counts, model, correction methods, ``skip_reason_counts``,
        and parallel fitting fields (``n_jobs_requested``, ``n_jobs_effective``,
        ``parallel_fallback``, ``parallel_fallback_reason``).
    """

    config: Config
    table: pl.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from proteoforge.schema import validate_discordance_result_table

        validate_discordance_result_table(self.table)

    @property
    def n_discordant(self) -> int:
        """Number of peptides flagged discordant."""
        from proteoforge.schema import IS_DISCORDANT

        return int(self.table.get_column(IS_DISCORDANT).sum())

    @property
    def discordant(self) -> pl.DataFrame:
        """Subset of the table flagged discordant."""
        import polars as pl

        from proteoforge.schema import IS_DISCORDANT

        return self.table.filter(pl.col(IS_DISCORDANT))


@dataclass(frozen=True)
class ClusterResult:
    """
    Per-peptide clustering outcome for proteins with discordant members.

    Attributes
    ----------
    config
        Frozen configuration used for the run.
    table
        One row per clustered ``(protein_id, peptide_id)`` with ``cluster_id``,
        ``cut_method``, and ``linkage_method``. Proteins without discordance
        are omitted; those peptides receive ``dPF_0`` during assignment.
    metadata
        Run statistics and parallel execution fields.
    """

    config: Config
    table: pl.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from proteoforge.schema import validate_cluster_result_table

        validate_cluster_result_table(self.table)

    @property
    def n_clustered_peptides(self) -> int:
        """Number of peptides assigned a cluster."""
        return self.table.height

    @property
    def n_clusters(self) -> int:
        """Number of unique protein-cluster pairs."""
        from proteoforge.schema import CLUSTER_ID, PROTEIN_ID

        if self.table.is_empty():
            return 0
        return self.table.select([PROTEIN_ID, CLUSTER_ID]).n_unique()


@dataclass(frozen=True)
class ProteoformMappingResult:
    """
    Per-peptide differential proteoform assignment.

    Attributes
    ----------
    config
        Frozen configuration used for the run.
    table
        One row per peptide in the prepared scope with ``is_discordant``,
        ``cluster_id``, and ``dpf_id``.
    metadata
        Counts by dPF class and validation summary.
    """

    config: Config
    table: pl.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from proteoforge.schema import validate_proteoform_mapping_table

        validate_proteoform_mapping_table(self.table)

    @property
    def n_differential_peptides(self) -> int:
        """Number of peptides assigned to positive dPF IDs."""
        import polars as pl

        from proteoforge.schema import DPF_ID

        return int(self.table.select((pl.col(DPF_ID) > 0).sum()).item())

    @property
    def n_singletons(self) -> int:
        """Number of peptides assigned to singleton ``dPF_-1``."""
        import polars as pl

        from proteoforge.schema import DPF_ID

        return int(self.table.select((pl.col(DPF_ID) == -1).sum()).item())
