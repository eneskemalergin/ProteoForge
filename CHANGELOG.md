<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CI: single ubuntu job on Python 3.12 until v0.1.0; full OS and version matrix deferred. Python 3.15 excluded while still in beta.

## [0.0.2] - 2026-06-06

Module 2 discordance: core WLS and RLM backends with unit-test coverage.

### Added

- `run_discordance()` and `DiscordanceResult` (exported from `proteoforge` and `proteoforge.discordance`)
- `models/`: `DiscordanceModel` protocol, batched WLS, vectorized Huber RLM (IRLS)
- `_correction.py`: NumPy-only `p_adjust` (BH, Bonferroni, Holm, Hochberg, BY)
- `_weights.py`: mask-derived WLS weights (`is_real`, `is_complete_missing`, `wls_biological_weight`)
- `_discordance.py`: shape-group batching, spawn pool parallelism, `fit_status` / `skip_reason_counts`
- `_progress.py`: shared `WeightedProgress` and `progress_enabled()` for tqdm-backed bars (terminal, notebook, piped/CI-safe)
- Test coverage: config `from_dict` validation, prepare/provenance failure paths, correction methods, WLS and RLM discordance, fit-status metadata wiring, `load_fixture_bundle`, and `_resolve_n_jobs(-1)`
- [Discordance](docs/discordance.md) documentation page

### Changed

- `Config`: `fdr`, `n_jobs`, and correction fields consumed by discordance (not prepare-only stubs)
- `wls_biological_weight` drives mask-derived weight construction for WLS
- `pyproject.toml`: `testpaths = ["tests"]` only
- `run_discordance(show_progress=True)`: peptide-weighted bar with parallel `as_completed` updates; ms/sub-second elapsed formatting; auto-suppressed when output is not interactive or `TQDM_DISABLE` is set
- Parallel shape pool: explicit `shutdown(cancel_futures=True)` on errors or interrupts to avoid leaving worker processes behind
- Test suite: removed duplicate and self-referential tests; strengthened prepare and discordance integration assertions

### Fixed

- Default correction: Bonferroni within protein, BH globally (`bonferroni` + `fdr_bh`)
- WLS `prepare()` now requires both mask columns or `weight` (no silent unit weights)
- `model='ebayes'` rejected at `Config` construction
- Docs: `is_discordant` column name, join-on-keys guidance, runtime deps, provenance wording
- `Config.from_dict`: missing `control_condition` now raises `ProteoForgeValidationError` instead of `KeyError`

## [0.0.1] - 2026-06-03

First release: peptide I/O, validation, and control-relative normalization.

### Added

- `Config` with YAML loading; experimental design and sample scope from `conditions`.
- Long-format peptide I/O (Parquet, CSV, TSV) with column harmonization and lazy parquet scans.
- `prepare()`, `prepare_from_parquet()`, and `PreparedDataset` long-format handoff ([PreparedDataset](docs/prepared-dataset.md)).
- Control-relative normalization (`intensity_normalized`) via Polars window expressions.
- Provenance column ingest (`is_real`, `is_complete_missing`, `weight`) for imputation-aware models.
- Validation: duplicate-key rejection, minimum peptides per protein, scope and design checks.
- Unit and regression normalization tests; CLI entry point (`proteoforge --version`).

### Changed

- Python support: minimum **3.12**, test target **3.12 to 3.15**.
- Runtime dependencies: NumPy 2.2+, Polars 1.26+, PyYAML.
