"""Protocols and shared types for clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from proteoforge._config import Config
    from proteoforge.types import DiscordanceResult, PreparedDataset


@dataclass(frozen=True)
class ProteinProfileBlock:
    """
    Condition-profile matrix for one discordant protein.

    Attributes
    ----------
    protein_id
        Protein identifier.
    peptide_ids
        Peptide identifiers in row order.
    profiles
        Median condition profiles, shape ``(n_peptides, n_conditions)``.
    is_discordant
        Discordance flags aligned to ``peptide_ids``.
    condition_levels
        Condition labels in column order.
    """

    protein_id: str
    peptide_ids: tuple[str, ...]
    profiles: npt.NDArray[np.float64]
    is_discordant: npt.NDArray[np.bool_]
    condition_levels: tuple[str, ...]

    @property
    def n_peptides(self) -> int:
        """Number of peptides in the block."""
        return len(self.peptide_ids)

    @property
    def n_conditions(self) -> int:
        """Number of profile columns."""
        return len(self.condition_levels)


class ProfileBuilder(Protocol):
    """Build per-protein condition profile blocks for clustering."""

    def build_blocks(
        self,
        prepared: PreparedDataset,
        discordance: DiscordanceResult,
    ) -> list[ProteinProfileBlock]:
        """
        Return profile blocks for proteins with discordant members.

        Parameters
        ----------
        prepared
            Normalized handoff from :func:`proteoforge.prepare.prepare`.
        discordance
            Discordance result from :func:`proteoforge.discordance.run_discordance`.

        Returns
        -------
        list of ProteinProfileBlock
            Blocks sorted by protein ID.
        """


class ClusterCut(Protocol):
    """Dendrogram cut strategy for one protein profile block."""

    name: str

    def cut(
        self,
        linkage_matrix: npt.NDArray[np.float64],
        condensed_dist: npt.NDArray[np.float64],
        *,
        n_samples: int,
        config: Config,
    ) -> npt.NDArray[np.intp]:
        """
        Return 1-based cluster labels.

        Parameters
        ----------
        linkage_matrix
            Ward linkage matrix, shape ``(n_samples - 1, 4)``.
        condensed_dist
            Condensed Euclidean distances for silhouette scoring.
        n_samples
            Number of peptides clustered.
        config
            Frozen pipeline configuration.

        Returns
        -------
        ndarray of intp
            Cluster labels, shape ``(n_samples,)``.
        """
