"""Nox sessions for lint, typecheck, and tests."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests"]

@nox.session
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.run("mypy")


@nox.session(python="3.12")
def tests(session: nox.Session) -> None:
    session.run(
        "pytest",
        "--cov=proteoforge",
        "--cov-report=term-missing",
        "tests",
        external=True,
    )
