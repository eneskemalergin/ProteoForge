"""Discordance model backends and :func:`select_model`."""

from __future__ import annotations

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.models._protocol import DiscordanceModel
from proteoforge.models._rlm import RLMModel
from proteoforge.models._wls import WLSModel

__all__ = [
    "DiscordanceModel",
    "RLMModel",
    "WLSModel",
    "select_model",
]


def select_model(name: str) -> DiscordanceModel:
    """
    Return the model backend for a config model name.

    Parameters
    ----------
    name
        Model name from ``Config.model``: ``"rlm"`` or ``"wls"``.

    Returns
    -------
    DiscordanceModel
        The matching model backend.

    Raises
    ------
    ProteoForgeValidationError
        If the name is unknown or ``ebayes`` (deferred).
    """
    if name == "rlm":
        return RLMModel()
    if name == "wls":
        return WLSModel()
    if name == "ebayes":
        msg = "model='ebayes' is not implemented. Use 'rlm' or 'wls'."
        raise ProteoForgeValidationError(msg)
    msg = f"Unknown model '{name}'. Valid models: 'rlm', 'wls'."
    raise ProteoForgeValidationError(msg)
