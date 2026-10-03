<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Clustering runs on every protein in the prepared scope; repository reorganized ahead of v0.1.0.

### Added

- `proteoforge.intel.parser.fasta`: UniProt FASTA to Polars table (header parse, sequence validation, molecular weight, skip accounting via `FastaParseResult`)
- `CITATION.cff` and a README citation for the article in the *Journal of Proteome Research* (2026, doi:10.1021/acs.jproteome.5c01235)
- README rewritten and fact-checked against the code: light and dark header (outlined Inter wordmark), the four method stages from the article, what "imputation-aware" means for RLM and WLS, verified install and quick-start steps, and links to the analysis repository and its Zenodo snapshot
- CI with Python 3.12 on Linux (x64, arm64) and macOS (arm64), Python 3.13 and 3.14 on Linux x64, plus coverage, dependency-floor, packaging, and runtime-audit jobs; runs on `main` and `dev`

### Changed

- `run_cluster()` clusters all proteins in scope, not only those with discordant peptides. `ClusterResult.table` and `ProteoformMappingResult.table` now include `cluster_id` on every peptide; canonical proteins still receive `dpf_id = 0`.
- `ClusterResult.metadata`: `n_proteins` added; `n_discordant_proteins` counts discordant proteins only.
- Clustering docs updated for full-scope behavior (`docs/clustering.md`, `docs/discordance.md`, `docs/index.md`, README).
- Minimum numba is 0.61.2 (0.61.0 does not support NumPy 2.2). All runtime floors are tested in CI.
- `run_discordance()` with the default RLM model is 33% to 54% faster: the Huber scale update runs in a numba kernel and the weighted least-squares step makes one pass fewer over the design. Discordance calls are unchanged; p-values match the previous implementation to within the IRLS convergence tolerance.
- `run_cluster()` clusters proteins in-process. A process pool was slower than serial clustering on every tested dataset.
- `n_jobs=-1` counts only the CPUs the process may use (for example under `taskset` or a SLURM allocation) before taking half, capped at 8.
- The q-value test compares against a SciPy smoothing-spline oracle instead of skipping.

### Fixed

- Discordance worker processes now start with one BLAS, OpenMP, numba, and Polars thread each. The limit was previously set inside the workers after NumPy had loaded, so each worker could start one BLAS thread per CPU.

### Removed

- The `n_jobs` argument of `run_cluster()` and the clustering `n_jobs_*` and `parallel_fallback*` metadata fields.
- MkDocs site, `docs/` pages, and the `docs` extra. The user guide is being rewritten.
- Placeholder extras `intel`, `plots`, `interactive`, `cli`, and `ref`; they installed packages that ProteoForge does not import yet.
- Python 3.15 classifier; issue and pull request templates; the deferred CodSpeed workflow.

## [0.0.4] - 2026-06-10

Correction subpackage (q-value, Hommel) and multiple-testing user documentation.

### Added

- Storey q-value correction via `p_adjust(..., "qvalue")`, with GCV pi0 on the shipped lambda grid (`proteoforge.correction.qvalue`)
- Hommel adjustment via `p_adjust(..., "hommel")`, R `p.adjust` parity including `n_tests` padding
- `proteoforge.correction` subpackage with `p_adjust`, `p_adjust_by_group`, and `VALID_METHODS` (also exported from `proteoforge`)
- `correction_within` / `correction_global` config values: `hommel`, `qvalue`
- [Multiple-testing correction](docs/correction.md); mkdocs nav entries for correction and clustering

### Changed

- Hommel adjustment uses an O(n) kernel in `correction._hommel`. R `p.adjust` parity unchanged.
- Correction implementation moved from `_correction.py` into `proteoforge.correction` (`_methods`, `qvalue/`); `_correction.py` remains a backward-compatible re-export
- User docs: discordance batching and `metadata`, WLS weight tiers, config YAML correction fields, index shipped/planned list, cross-links across guides
- README and docs index: pipeline diagrams and correction method summary aligned with shipped Modules 1 to 3

### Removed

- Unreleased experimental `proteoforge.correction.ihw` subpackage (never shipped on PyPI)

## [0.0.3] - 2026-06-08

Module 3 clustering and dPF assignment.

### Added

- `run_cluster()`, `assign_proteoforms()`, and `ClusterResult` / `ProteoformMappingResult` for the prepare -> discordance -> cluster -> dPF pipeline
- Numba clustering geometry under `proteoforge.clustering` (Euclidean pdist, Ward linkage, hybrid outlier cut)
- [Clustering](docs/clustering.md) user documentation
- `ProteoForgeValidationError` and `ProteoForgeIOError` on the top-level `proteoforge` import
- `DiscordanceResult.table` contract validation at construction

### Changed

- CI: single ubuntu job on Python 3.12 until v0.1.0; full OS and version matrix deferred. Python 3.15 excluded while still in beta.

### Fixed

- `prepare_from_parquet`: missing paths raise `ProteoForgeIOError`

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
