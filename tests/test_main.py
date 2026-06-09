"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from proteoforge.__main__ import main


def test_main_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "proteoforge" in captured.out


def test_main_prints_help_without_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "Library API" in captured.out
    assert "clustering" in captured.out
