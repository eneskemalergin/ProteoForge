"""Tests that runtime imports avoid scipy, sklearn, and statsmodels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_runtime_imports_stay_numpy_only() -> None:
    # Fresh subprocess: numba is core; sklearn/statsmodels must not load
    code = (
        "import sys\n"
        "import proteoforge\n"
        "import proteoforge.cluster\n"
        "import proteoforge.correction\n"
        "import proteoforge.discordance\n"
        "from proteoforge import p_adjust\n"
        "assert p_adjust([0.1], 'bonferroni')[0] == 0.1\n"
        "from proteoforge.cluster import run_cluster, assign_proteoforms\n"
        "from proteoforge.discordance import run_discordance\n"
        "assert run_cluster and assign_proteoforms and run_discordance\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] "
        "in {'dynamicTreeCut', 'sklearn', 'sknetwork', 'statsmodels'})\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_src_does_not_import_scipy_or_sklearn() -> None:
    # SciPy/sklearn imports belong in tests/, not src/
    root = Path(__file__).resolve().parents[1] / "src" / "proteoforge"
    forbidden = ("scipy", "sklearn", "statsmodels", "sknetwork", "dynamicTreeCut")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.relative_to(root)}: {name}")
    assert not offenders, offenders
