"""Smoke tests for package metadata."""

from __future__ import annotations

import re

from proteoforge import __version__


def test_version_is_non_empty_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_version_matches_pep440_or_unknown() -> None:
    pep440 = re.compile(r"^\d+\.\d+")
    assert pep440.match(__version__) or __version__ == "0.0.0+unknown"
