"""Multiple-testing correction (R ``p.adjust`` parity and extensions)."""

from proteoforge.correction._adjust import VALID_METHODS, p_adjust, p_adjust_by_group

__all__ = [
    "VALID_METHODS",
    "p_adjust",
    "p_adjust_by_group",
]
