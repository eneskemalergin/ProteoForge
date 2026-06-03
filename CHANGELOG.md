<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Simple md-based documentation added to keep track of things, and clean explanation of the package as far as it is implemented. It includes: [configuration](docs/config.md), [I/O](docs/io.md), [prepare](docs/prepare.md), [normalization](docs/normalization.md), and updated [PreparedDataset](docs/prepared-dataset.md) guide.

## [0.0.1] - 2026-06-03

First release: peptide I/O, validation, and control-relative normalization.

### Added

- `Config` with YAML loading; experimental design and sample scope from `conditions`.
- Long-format peptide I/O (Parquet, CSV, TSV) with column harmonization and lazy parquet scans.
- `prepare()`, `prepare_from_parquet()`, and `PreparedDataset` long-format handoff ([PreparedDataset](docs/prepared-dataset.md)).
- Control-relative normalization (`intensity_normalized`) via Polars window expressions.
- Provenance column ingest (`is_real`, `is_complete_missing`, `weight`) for imputation-aware models.
- Validation: duplicate-key rejection, minimum peptides per protein, scope and design checks.
- Unit and golden normalization tests; CLI entry point (`proteoforge --version`).

### Changed

- Python support: minimum **3.12**, test target **3.12 to 3.15**.
- Runtime dependencies: NumPy 2.2+, Polars 1.26+, SciPy 1.15+.
