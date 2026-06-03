"""ProteoForge exception hierarchy."""

from __future__ import annotations


class ProteoForgeError(Exception):
    """Base exception for ProteoForge errors."""


class ProteoForgeValidationError(ProteoForgeError):
    """Data contract violation."""


class ProteoForgeIOError(ProteoForgeError):
    """File format or parse failure."""
