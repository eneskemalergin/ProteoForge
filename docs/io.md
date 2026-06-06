# Input and output

ProteoForge expects a **long-format** peptide intensity table: one row per `(protein_id, peptide_id, sample_id)`. The package does not impute, search, or quantify. Upstream imputation is required.

## Supported file formats

- **`read_peptides()`:** Parquet, CSV, TSV
- **`prepare_from_parquet()`:** Parquet only
- **`read_provenance()`:** Parquet, CSV, TSV
- **`read_fasta()`:** FASTA only

Parquet is preferred for large tables. `prepare_from_parquet()` lazy-scans, projects columns from `column_map`, and filters to configured samples before materialization.

```python
from proteoforge import Config
from proteoforge.io import read_peptides, read_provenance, read_fasta

config = Config.from_yaml_path("config.yaml")

peptides = read_peptides("peptides.parquet", config)
masks = read_provenance("provenance.parquet")
fasta = read_fasta("proteins.fasta")
```

For file-based prepare, pass a parquet path directly:

```python
from proteoforge import prepare_from_parquet

dataset = prepare_from_parquet("peptides.parquet", config)
```

## Required columns

After harmonization, every peptide table must contain:

- **`protein_id`**: protein accession or group (see protein resolution below)
- **`peptide_id`**: peptide sequence or stable identifier
- **`sample_id`**: run or sample name matching `Config.conditions`
- **`intensity`**: numeric, finite (see intensity rules below)

`condition` is not required in input files. `prepare()` attaches it from `Config.conditions`.

## Optional provenance columns

For imputation-aware modeling (WLS and empirical-Bayes paths):

- **`is_real`**: `true` if the intensity was measured; `false` if imputed
- **`is_complete_missing`**: `true` if the value stands in for condition-wide missingness
- **`weight`**: precomputed per-observation weight

These may live in the main peptide file or in a separate table joined via `prepare(..., provenance=...)`. When `model="wls"`, provide either:

- a precomputed **`weight`** column, or
- **both** **`is_real`** and **`is_complete_missing`** (mask-derived tiered weights)

A single mask column alone is not sufficient and `prepare()` will reject the input.

## Column harmonization

On ingest, ProteoForge:

1. Selects source columns named in `config.column_map` (when present in the file)
2. Renames to canonical names
3. Casts types (`String` for IDs, `Float64` for intensity, `Boolean` for masks)
4. Resolves multi-accession protein groups (below)
5. Filters to samples listed in `config.conditions` (lazy path only, before full collect)

`materialize_peptide_table()` is the shared entry for both `read_peptides()` and `prepare()`. Behavior is identical whether input is a lazy scan or an eager frame.

### Protein group resolution

When `protein_id` contains semicolon-separated accessions (e.g. `A0A075B6K5;Q12345`), ProteoForge collapses to one representative ID:

- Prefer a 6-character accession (canonical UniProt length) when present
- Otherwise use the first accession in the list

This runs automatically during ingest when any row contains `;`.

## Intensity values

Before normalization, validation requires:

- Numeric `intensity` column
- No `NaN` or infinite values (error: impute upstream)
- Null values trigger a warning; treat nulls as missing data and impute before production use

Each `(protein_id, peptide_id)` must not be all-NaN across samples after scoping.

## Primary key

Observations are keyed by `(protein_id, peptide_id, sample_id)`. Duplicate keys raise `ProteoForgeValidationError` during `prepare()`.

Row order in the file is preserved through prepare. Downstream code must join on key columns, not assume sort order.

## Provenance I/O

`read_provenance()` loads a long table with keys `protein_id`, `peptide_id`, `sample_id` and at least one of `is_real`, `is_complete_missing`, `weight`.

`attach_provenance()` (called inside `prepare()` when `provenance=` is passed) left-joins masks onto the peptide table. Rules:

- Every provenance key must exist in the peptide table
- When provenance is non-empty, every peptide key must have a matching provenance row (full coverage for WLS)

```python
from proteoforge import prepare

dataset = prepare(peptides, config, provenance=masks)
```

## FASTA

`read_fasta()` returns a minimal Polars table with columns `entry` (header without `>`) and `sequence`. Full sequence annotation is not part of v0.0.2; this loader exists for early integration tests and future sequence modules.

## Deprecated helpers

- **`peptides_from_frame()`**: use `materialize_peptide_table()` instead
- **`read_design()`**: put design in `Config.conditions`

## Exceptions

- **`ProteoForgeIOError`**: missing file, unsupported extension, parse failure
- **`ProteoForgeValidationError`**: missing columns after harmonization, provenance key mismatch, empty table

See [Prepare](prepare.md) for the full validation checklist applied after I/O.
