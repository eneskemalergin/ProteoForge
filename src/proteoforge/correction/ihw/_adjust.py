"""Public IHW adjustment API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.correction import p_adjust
from proteoforge.correction.ihw._internal import ihw_internal
from proteoforge.correction.ihw._splitting import groups_by_filter


@dataclass(frozen=True)
class IHWResult:
    """
    IHW adjustment output.

    Attributes
    ----------
    pvalues
        Input p-values (same shape as passed to :func:`adjust_ihw`).
    adj_pvalues
        Adjusted p-values after weighted testing.
    weights
        Per-hypothesis weights from cross-validated bin weights.
    weighted_pvalues
        ``pvalues / weights`` with R-style guards.
    groups
        Zero-based covariate bin index per hypothesis.
    folds
        Zero-based CV fold index per hypothesis.
    alpha
        Nominal FDR or FWER level used in the fit.
    nbins
        Number of covariate strata.
    nfolds
        Outer cross-validation fold count used.
    penalty
        ``"total_variation"`` (ordinal covariates) or ``"uniform_deviation"``.
    adjustment_type
        ``"bh"`` or ``"bonferroni"``.
    """

    pvalues: npt.NDArray[np.float64]
    adj_pvalues: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    weighted_pvalues: npt.NDArray[np.float64]
    groups: npt.NDArray[np.intp]
    folds: npt.NDArray[np.intp]
    alpha: float
    nbins: int
    nfolds: int
    penalty: str
    adjustment_type: str


def adjust_ihw(
    pvalues: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64],
    alpha: float,
    *,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas: npt.NDArray[np.float64] | str = "auto",
    adjustment_type: str = "bh",
    folds: npt.NDArray[np.intp] | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = 1,
) -> IHWResult:
    """
    Apply Independent Hypothesis Weighting to p-values.

    Parameters
    ----------
    pvalues
        Raw p-values, one per hypothesis.
    covariates
        Covariate values independent of p under H0.
    alpha
        Target FDR (``bh``) or FWER (``bonferroni``) level.
    covariate_type
        ``"ordinal"`` or ``"nominal"``.
    nbins
        Covariate strata count, or ``"auto"``.
    nfolds
        Cross-validation folds for weight learning.
    nfolds_internal
        Nested CV folds for lambda selection.
    nsplits_internal
        Random repeats for nested lambda selection.
    lambdas
        Regularization grid or ``"auto"``.
    adjustment_type
        ``"bh"`` or ``"bonferroni"``.
    folds
        Optional pre-specified zero-based fold assignment.
    rng
        Random generator; created from ``seed`` when omitted.
    seed
        Seed for covariate tie-breaking (R ``groups_by_filter``). When ``rng``
        is omitted, also seeds the CV fold generator.

    Returns
    -------
    IHWResult
        Adjusted p-values, weights, and diagnostics.

    Raises
    ------
    ProteoForgeValidationError
        If inputs are invalid.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    x = np.asarray(covariates, dtype=np.float64)
    if p.shape[0] == 0:
        msg = "pvalues must not be empty"
        raise ProteoForgeValidationError(msg)
    if np.any(~np.isfinite(p)):
        msg = "p-values must be finite"
        raise ProteoForgeValidationError(msg)
    if np.any(~np.isfinite(x)):
        msg = "covariates must be finite"
        raise ProteoForgeValidationError(msg)
    if np.any((p < 0.0) | (p > 1.0)):
        msg = "p-values must lie in [0, 1]"
        raise ProteoForgeValidationError(msg)
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ProteoForgeValidationError(msg)
    if p.shape[0] != x.shape[0]:
        msg = f"Length mismatch: {p.shape[0]} p-values vs {x.shape[0]} covariates"
        raise ProteoForgeValidationError(msg)
    if adjustment_type not in ("bh", "bonferroni"):
        msg = f"Unknown adjustment_type: {adjustment_type!r}"
        raise ProteoForgeValidationError(msg)
    if covariate_type not in ("ordinal", "nominal"):
        msg = f"Unknown covariate_type: {covariate_type!r}"
        raise ProteoForgeValidationError(msg)

    n = p.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)

    penalty = "total_variation" if covariate_type == "ordinal" else "uniform_deviation"

    if isinstance(nbins, str):
        if nbins != "auto":
            msg = f"nbins must be an integer or 'auto', got {nbins!r}"
            raise ProteoForgeValidationError(msg)
        nbins_i = max(1, min(40, n // 1500))
    else:
        nbins_i = int(nbins)

    bin_rng = np.random.default_rng(seed)
    groups = groups_by_filter(x, nbins_i, rng=bin_rng)
    m_groups = np.bincount(groups, minlength=nbins_i).astype(np.intp)

    if isinstance(lambdas, str):
        if lambdas != "auto":
            msg = f"lambdas must be an array or 'auto', got {lambdas!r}"
            raise ProteoForgeValidationError(msg)
        lam_grid = np.array(
            sorted({0.0, 1.0, nbins_i / 8, nbins_i / 4, nbins_i / 2, nbins_i, np.inf}),
            dtype=np.float64,
        )
    else:
        lam_grid = np.asarray(lambdas, dtype=np.float64)

    pad_method = "fdr_bh" if adjustment_type == "bh" else "bonferroni"

    if nbins_i == 1:
        order = np.argsort(p)
        sorted_p = p[order]
        adj_sorted = p_adjust(sorted_p, pad_method, n_tests=n)
        inv = np.argsort(order)
        return IHWResult(
            pvalues=p,
            adj_pvalues=adj_sorted[inv],
            weights=np.ones(n, dtype=np.float64),
            weighted_pvalues=p.copy(),
            groups=groups,
            folds=np.zeros(n, dtype=np.intp),
            alpha=alpha,
            nbins=1,
            nfolds=1,
            penalty=penalty,
            adjustment_type=adjustment_type,
        )

    order = np.argsort(p)
    sorted_p = p[order]
    sorted_g = groups[order]
    sorted_folds: npt.NDArray[np.intp] | None = None
    if folds is not None:
        f = np.asarray(folds, dtype=np.intp)
        if f.shape[0] != n:
            msg = f"folds length {f.shape[0]} != {n}"
            raise ProteoForgeValidationError(msg)
        sorted_folds = f[order]

    result = ihw_internal(
        sorted_g,
        sorted_p,
        alpha,
        lam_grid,
        m_groups,
        penalty=penalty,
        nfolds=nfolds,
        nfolds_internal=nfolds_internal,
        nsplits_internal=nsplits_internal,
        adjustment_type=adjustment_type,
        rng=rng,
        sorted_folds=sorted_folds,
    )

    inv = np.argsort(order)
    weights = np.asarray(result["sorted_weights"], dtype=np.float64)[inv]
    weighted = np.asarray(result["sorted_weighted_pvalues"], dtype=np.float64)[inv]
    adj = np.asarray(result["sorted_adj_p"], dtype=np.float64)[inv]
    out_folds = np.asarray(result["sorted_folds"], dtype=np.intp)[inv]
    return IHWResult(
        pvalues=p,
        adj_pvalues=adj,
        weights=weights,
        weighted_pvalues=weighted,
        groups=groups,
        folds=out_folds,
        alpha=alpha,
        nbins=nbins_i,
        nfolds=nfolds,
        penalty=penalty,
        adjustment_type=adjustment_type,
    )
