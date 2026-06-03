"""Nox sessions for lint, typecheck, and tests."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests"]

PYTHON_VERSIONS = ["3.12", "3.13", "3.14", "3.15"]


@nox.session
def lint(session: nox.Session) -> None:
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.run("mypy")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.run(
        "pytest",
        "--cov=proteoforge",
        "--cov-report=term-missing",
        "tests",
        external=True,
    )
