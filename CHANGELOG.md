# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Python support: minimum **3.12**, test target **3.12 to 3.15** (dropped 3.10/3.11).
- Runtime dependencies added: NumPy 2.2+, Polars 1.26+, SciPy 1.15+.
- Optional `cli` extra (Typer, Rich, PyYAML); bumped optional dependency floors.

### Added

- Initial project scaffolding: `uv` workflow, `hatchling` + `hatch-vcs`, CI, docs, and PyPI publish pipeline.
- CLI entry point (`proteoforge --version`).
- Smoke tests, benchmarks, and development tooling (`ruff`, `mypy`, `pytest`, `nox`, `pre-commit`).
