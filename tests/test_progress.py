"""Tests for shared progress helpers."""

from __future__ import annotations

import sys

import pytest

from proteoforge._progress import WeightedProgress, format_elapsed, progress_enabled


def test_format_elapsed_milliseconds() -> None:
    assert format_elapsed(0.042) == "42ms"


def test_format_elapsed_subsecond_seconds() -> None:
    assert format_elapsed(3.5) == "3.500s"


def test_format_elapsed_minutes() -> None:
    assert format_elapsed(125.0) == "2:05"


def test_progress_enabled_respects_requested() -> None:
    assert progress_enabled(requested=False) is False


def test_progress_enabled_respects_tqdm_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setenv("TQDM_DISABLE", "1")
    assert progress_enabled(requested=True) is False


def test_progress_enabled_with_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(
        "proteoforge._progress._notebook_frontend_active",
        lambda: False,
    )
    assert progress_enabled(requested=True) is True


def test_progress_enabled_without_tty_or_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(
        "proteoforge._progress._notebook_frontend_active",
        lambda: False,
    )
    assert progress_enabled(requested=True) is False


def test_weighted_progress_updates_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[int] = []

    class _FakeBar:
        def update(self, n: int) -> None:
            updates.append(n)

        def close(self) -> None:
            updates.append(-1)

    def _fake_tqdm(**kwargs: object) -> _FakeBar:
        del kwargs
        return _FakeBar()

    monkeypatch.setattr(
        "proteoforge._progress.progress_enabled",
        lambda *, requested: requested,
    )
    monkeypatch.setattr(
        "proteoforge._progress._precise_tqdm_class",
        lambda: _fake_tqdm,
    )
    with WeightedProgress(enabled=True, total=5, desc="test", unit="item") as bar:
        bar.update(2)
        bar.update(3)

    assert updates == [2, 3, -1]


def test_weighted_progress_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tqdm.auto as tqdm_auto

    calls: list[int] = []

    def _fake_tqdm(**kwargs: object) -> object:
        calls.append(1)
        del kwargs
        return object()

    monkeypatch.setattr(
        "proteoforge._progress.progress_enabled",
        lambda *, requested: False,
    )
    monkeypatch.setattr(tqdm_auto, "tqdm", _fake_tqdm)
    with WeightedProgress(enabled=True, total=5, desc="test") as bar:
        bar.update(1)

    assert calls == []
