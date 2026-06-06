"""Protocol and shared design builder for discordance model backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from proteoforge._layout import ProteinBlock


class DiscordanceModel(Protocol):
    """
    One-vs-rest interaction backend.

    A backend fits a batch of ``m`` one-vs-rest designs that share the same
    shape and returns one interaction p-value per design. The interaction
    block is always the trailing ``n_interaction`` columns of each design.

    Attributes
    ----------
    name : str
        Backend identifier.
    use_f : bool
        ``True`` when significance uses an F test (WLS, OLS), ``False`` for the
        chi-square test (RLM).
    """

    name: str
    use_f: bool

    def fit_pvalues(
        self,
        design: npt.NDArray[np.float64],
        response: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64] | None,
        *,
        n_interaction: int,
    ) -> npt.NDArray[np.float64]:
        """
        Fit ``m`` designs and return interaction p-values, shape ``(m,)``.

        Parameters
        ----------
        design
            Design tensor, shape ``(m, n_obs, p)``.
        response
            Response per design, shape ``(m, n_obs)``.
        weight
            Per-observation weights, shape ``(m, n_obs)``, or ``None``.
        n_interaction
            Number of trailing interaction columns to test jointly.
        """
        ...

    def fit_pvalues_and_status(
        self,
        design: npt.NDArray[np.float64],
        response: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64] | None,
        *,
        n_interaction: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
        """
        Fit ``m`` designs and return p-values plus per-design status codes.

        Status strings are defined in :mod:`proteoforge.models._fit_status`.
        """
        ...


def build_design_stack(
    block: ProteinBlock,
) -> npt.NDArray[np.float64]:
    """
    Build the one-vs-rest design tensor for a protein block.

    Column order is fixed and asserted: ``[intercept, condition_1..K-1,
    allothers, (condition x allothers)_1..K-1]``. The trailing ``K-1`` columns
    are the interaction block.

    Parameters
    ----------
    block
        Per-protein long fitting block.

    Returns
    -------
    np.ndarray
        Design tensor, shape ``(n_peptides, n_obs, 2 * n_conditions)``.
    """
    n_conditions = block.n_conditions
    n_peptides = block.n_peptides
    n_obs = block.n_obs
    width = 2 * n_conditions

    condition_levels = np.arange(1, n_conditions)
    condition_dummies = (
        block.condition_code[:, None] == condition_levels[None, :]
    ).astype(np.float64)
    intercept = np.ones((n_obs, 1), dtype=np.float64)
    base = np.concatenate([intercept, condition_dummies], axis=1)

    targets = np.arange(n_peptides, dtype=np.intp)
    allothers = (block.peptide_code[None, :] == targets[:, None]).astype(np.float64)
    interaction = allothers[:, :, None] * condition_dummies[None, :, :]

    design = np.empty((n_peptides, n_obs, width), dtype=np.float64)
    design[:, :, :n_conditions] = base
    design[:, :, n_conditions : n_conditions + 1] = allothers[:, :, None]
    design[:, :, n_conditions + 1 :] = interaction
    expected_width = 2 * n_conditions
    if design.shape[2] != expected_width:
        msg = f"Design width {design.shape[2]} != expected {expected_width}."
        raise ValueError(msg)
    return design
