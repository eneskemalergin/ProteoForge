"""Guard: the runtime package must not import scipy or statsmodels."""

from __future__ import annotations

import subprocess
import sys


def test_runtime_imports_stay_numpy_only() -> None:
    code = (
        "import sys\n"
        "import proteoforge\n"
        "import proteoforge.discordance\n"
        "from proteoforge.discordance import run_discordance\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] "
        "in {'scipy', 'statsmodels'})\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
