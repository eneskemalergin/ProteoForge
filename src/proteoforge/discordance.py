"""
Public discordance entry point.

Exposes :func:`run_discordance` and the result type for Module 2. Phase 4 may
wrap this inside ``discover()``.
"""

from __future__ import annotations

from proteoforge._discordance import run_discordance
from proteoforge.types import DiscordanceResult

__all__ = ["DiscordanceResult", "run_discordance"]
