"""Tests for discordance model selection."""

from __future__ import annotations

import pytest

from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.models import select_model


def test_select_model_rlm() -> None:
    model = select_model("rlm")
    assert model.name == "rlm"


def test_select_model_wls() -> None:
    model = select_model("wls")
    assert model.name == "wls"


def test_select_model_rejects_ebayes() -> None:
    with pytest.raises(ProteoForgeValidationError, match="ebayes"):
        select_model("ebayes")


def test_select_model_rejects_unknown() -> None:
    with pytest.raises(ProteoForgeValidationError, match="Unknown model"):
        select_model("foo")
