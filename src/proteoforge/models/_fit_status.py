"""Per-design fit status codes for discordance diagnostics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FIT_STATUS_OK: str = "ok"
FIT_STATUS_RANK_DEFICIENT: str = "rank_deficient"
FIT_STATUS_INSUFFICIENT_DF: str = "insufficient_df"
FIT_STATUS_ILL_CONDITIONED: str = "ill_conditioned"
FIT_STATUS_ZERO_SCALE: str = "zero_scale"
FIT_STATUS_ZERO_ROBUST_SCALE: str = "zero_robust_scale"
FIT_STATUS_WALD_FAILED: str = "wald_failed"

_ALL_STATUSES: frozenset[str] = frozenset(
    {
        FIT_STATUS_OK,
        FIT_STATUS_RANK_DEFICIENT,
        FIT_STATUS_INSUFFICIENT_DF,
        FIT_STATUS_ILL_CONDITIONED,
        FIT_STATUS_ZERO_SCALE,
        FIT_STATUS_ZERO_ROBUST_SCALE,
        FIT_STATUS_WALD_FAILED,
    }
)


def empty_status(m: int) -> npt.NDArray[np.object_]:
    """Allocate a status vector prefilled with rank deficiency."""
    out = np.empty(m, dtype=object)
    out[:] = FIT_STATUS_RANK_DEFICIENT
    return out


def count_statuses(status: npt.NDArray[np.object_]) -> dict[str, int]:
    """Aggregate per-peptide fit status counts for run metadata."""
    counts: dict[str, int] = {code: 0 for code in _ALL_STATUSES}
    for value in status:
        key = str(value)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts) if counts[key] > 0}
