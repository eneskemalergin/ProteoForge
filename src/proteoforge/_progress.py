"""
Shared progress reporting for long-running package tasks.

Uses ``tqdm.auto`` (terminal or notebook). Disabled when the user opts out,
when ``TQDM_DISABLE`` is set, or when neither an interactive terminal nor a
notebook frontend is detected (piped logs, CI capture).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any


def format_elapsed(seconds: float) -> str:
    """
    Format a duration for progress display.

    Sub-second values use milliseconds; under one minute uses second decimals;
    longer values use ``M:SS``.
    """
    if seconds < 0:
        return "0ms"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.3f}s"
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def progress_enabled(*, requested: bool) -> bool:
    """
    Return whether a progress bar should be shown.

    Parameters
    ----------
    requested
        Caller flag (e.g. ``show_progress`` on ``run_discordance``).

    Returns
    -------
    bool
        ``True`` only when ``requested`` is set and the environment supports
        interactive display.
    """
    if not requested:
        return False
    disabled = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return False
    return _interactive_display_available()


def _interactive_display_available() -> bool:
    if sys.stderr.isatty() or sys.stdout.isatty():
        return True
    return _notebook_frontend_active()


def _notebook_frontend_active() -> bool:
    try:
        ipython = importlib.import_module("IPython")
        get_ipython = ipython.get_ipython
    except (ImportError, AttributeError):
        return False
    shell = get_ipython()
    if shell is None:
        return False
    name = shell.__class__.__name__
    return name in {"ZMQInteractiveShell", "GoogleColabShell", "Shell"}


def _precise_tqdm_class() -> Any:
    from tqdm.auto import tqdm as auto_tqdm

    class PreciseTqdm(auto_tqdm):  # type: ignore[type-arg]
        @staticmethod
        def format_interval(seconds: float) -> str:
            return format_elapsed(seconds)

    return PreciseTqdm


class WeightedProgress:
    """
    Weighted unit progress bar (context manager).

    Parameters
    ----------
    enabled
        Whether the caller wants progress (combined with :func:`progress_enabled`).
    total
        Total weighted units (e.g. peptide count).
    desc
        Short label shown left of the bar.
    unit
        Unit name for rate display (default ``peptide``).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        total: int,
        desc: str,
        unit: str = "peptide",
    ) -> None:
        self._bar: Any = None
        if progress_enabled(requested=enabled) and total > 0:
            precise_tqdm = _precise_tqdm_class()
            self._bar = precise_tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                smoothing=0.05,
                mininterval=0.05,
            )

    def update(self, n: int) -> None:
        if self._bar is not None and n > 0:
            self._bar.update(n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> WeightedProgress:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
