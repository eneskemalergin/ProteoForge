"""ProteoForge exception hierarchy."""

from __future__ import annotations


class ProteoForgeError(Exception):
    """Base exception for ProteoForge errors."""


class ProteoForgeValidationError(ProteoForgeError):
    """Raised when input data or configuration violates the package contract."""


class ProteoForgeIOError(ProteoForgeError):
    """Raised when a file cannot be read or parsed."""


class ProteoForgeParallelFallbackWarning(UserWarning):
    """Parallel discordance fitting fell back to the serial path."""
