"""Shipped numerical constants (fixed grids, precomputed spline geometry)."""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# Storey / Bioconductor qvalue lambda grid (step 0.05 on (0, 1)).
QVALUE_LAMBDAS = np.arange(0.05, 0.96, 0.05, dtype=np.float64)

PI0_GCV_SPLINE_ASSET = "pi0_gcv_spline.npz"


def load_pi0_gcv_spline_matrices() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Load banded B-spline design (``X``) and penalty (``wE``) for the pi0 GCV spline.

    Matrices match ``scipy.interpolate.make_smoothing_spline`` on
    :data:`QVALUE_LAMBDAS`. Regenerate with ``scripts/build_pi0_gcv_spline_constants.py``.
    """
    ref = resources.files(__package__).joinpath(PI0_GCV_SPLINE_ASSET)
    with resources.as_file(ref) as path:
        data = np.load(path)
    return (
        np.ascontiguousarray(data["X"], dtype=np.float64),
        np.ascontiguousarray(data["wE"], dtype=np.float64),
    )
