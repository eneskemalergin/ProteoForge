"""Backward-compatible re-exports; prefer :mod:`proteoforge.correction`."""

from proteoforge.correction import VALID_METHODS, p_adjust, p_adjust_by_group

__all__ = ["VALID_METHODS", "p_adjust", "p_adjust_by_group"]
