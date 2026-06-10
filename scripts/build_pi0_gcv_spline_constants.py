#!/usr/bin/env python3
"""Build banded design matrices for the Storey pi0 GCV spline grid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.interpolate import _bsplines as bs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proteoforge.constants import QVALUE_LAMBDAS  # noqa: E402

OUT = ROOT / "src" / "proteoforge" / "constants" / "pi0_gcv_spline.npz"


def main() -> None:
    x = QVALUE_LAMBDAS
    n = len(x)
    w = np.ones(n)
    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    x_bspl = BSpline.design_matrix(x, t, 3)
    x_banded = np.zeros((5, n))
    for i in range(1, 4):
        x_banded[i, 2:-2] = x_bspl[i : i - 4, 3:-3][np.diag_indices(n - 4)]
    x_banded[1, 1] = x_bspl[0, 0]
    x_banded[2, :2] = (
        (x[2] + x[1] - 2 * x[0]) * x_bspl[0, 0],
        x_bspl[1, 1] + x_bspl[1, 2],
    )
    x_banded[3, :2] = ((x[2] - x[0]) * x_bspl[1, 1], x_bspl[2, 2])
    x_banded[1, -2:] = (x_bspl[-3, -3], (x[-1] - x[-3]) * x_bspl[-2, -2])
    x_banded[2, -2:] = (
        x_bspl[-2, -3] + x_bspl[-2, -2],
        (2 * x[-1] - x[-2] - x[-3]) * x_bspl[-1, -1],
    )
    x_banded[3, -2] = x_bspl[-1, -1]

    w_e = np.zeros((5, n))
    w_e[2:, 0] = bs._coeff_of_divided_diff(x[:3]) / w[:3]
    w_e[1:, 1] = bs._coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        w_e[:, j] = (
            (x[j + 2] - x[j - 2])
            * bs._coeff_of_divided_diff(x[j - 2 : j + 3])
            / w[j - 2 : j + 3]
        )
    w_e[:-1, -2] = -bs._coeff_of_divided_diff(x[-4:]) / w[-4:]
    w_e[:-2, -1] = bs._coeff_of_divided_diff(x[-3:]) / w[-3:]
    w_e *= 6

    np.savez(OUT, x=x, X=x_banded, wE=w_e)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
