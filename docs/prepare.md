# Prepare

`prepare()` validates peptide input, applies control-relative normalization, and returns a `PreparedDataset`. Pass the result to `run_discordance()`, then `run_cluster()` and `assign_proteoforms()` for the full pipeline through dPF assignment.

Public API:

```python
from proteoforge import Config, prepare, prepare_from_parquet

config = Config.from_yaml_path("config.yaml")

# From an in-memory or lazy Polars table
dataset = prepare(peptides, config)

# From parquet (lazy scan, recommended for files)
dataset = prepare_from_parquet("peptides.parquet", config)

# With separate provenance masks
dataset = prepare(peptides, config, provenance=masks)
```

`validate_and_prepare()` is an alias exported from `proteoforge` for the same function.

## Which entry point to use

**`prepare_from_parquet(path, config)`** when starting from a parquet file. Column projection and sample filtering happen on the lazy scan before collect.

**`prepare(frame, config)`** when the table is already in memory or when wrapping a custom `LazyFrame` scan.

**`read_peptides()` + `prepare()`** when inspection of harmonized rows is needed before normalization, or when input is CSV/TSV (not supported by `prepare_from_parquet`).

```python
from proteoforge.io import read_peptides

peptides = read_peptides("peptides.csv", config)
dataset = prepare(peptides, config)
```

If `read_peptides()` output is already canonical, scoped, and protein-resolved, `prepare()` skips redundant harmonization.

## Pipeline steps

`validate_and_prepare()` runs these stages in order:

1. **Ingest**: harmonize columns, resolve protein groups, scope to configured samples
2. **Design attach**: drop any input `condition` column; map `sample_id` to condition from `Config.conditions`
3. **Provenance attach** (optional): join masks or weights by primary key
4. **Column select**: keep keys, intensity, condition, and relevant provenance fields
5. **Structure validation**: keys, design coverage, duplicates, intensity finiteness, WLS provenance rule
6. **Coverage validation**: minimum peptides per protein; reject all-NaN peptide rows
7. **Normalize**: `normalize_control_relative_long()` adds `intensity_normalized`
8. **Pack**: build `PreparedDataset` with metadata

Output shape: long table with one row per `(protein_id, peptide_id, sample_id)`. See [PreparedDataset](prepared-dataset.md).

## Validation rules

Failures raise `ProteoForgeValidationError` unless noted as warnings.

- **Design and scope**
    - Every sample in `config.conditions` must appear in the peptide table after scoping
    - Samples in the table but not in config are dropped (warning lists up to 8 IDs)
    - At least two conditions, each with at least two samples (also enforced on `Config`)
- **Table structure**
    - Required columns: `protein_id`, `peptide_id`, `sample_id`, `condition`, `intensity`
    - Unique primary key `(protein_id, peptide_id, sample_id)`
    - `intensity` must be numeric with no NaN or infinity
- **Coverage**
    - Each protein must have at least `min_peptides` unique peptides
    - No peptide row set may be entirely NaN on intensity
- **WLS**
    - When `model="wls"`, the table must include `weight` or both `is_real` and `is_complete_missing` after attach

**Warnings (non-fatal)**

- Null intensity values (imputed data expected; nulls should be rare)
- Samples dropped because they are absent from `config.conditions`

## Metadata on `PreparedDataset`

`dataset.metadata` includes counts and scope audit fields:

- `n_proteins`, `n_peptides`, `n_samples`
- `nan_fraction` on raw intensity after validation
- `control_condition`, `conditions_used`
- `samples_used`, `samples_dropped`

## Model-specific handoff

`PreparedDataset` properties `is_real`, `is_complete_missing`, and `weight` return NumPy arrays aligned to `peptides` rows only when `config.model` is `wls` and the columns exist. For `rlm`, these properties return `None` even if provenance columns are retained on `peptides`.

For `model="wls"`, `prepare()` requires a `weight` column or **both** `is_real` and `is_complete_missing`.

Normalization itself does not branch on `model`. The field gates provenance validation and selects the discordance backend in `run_discordance()` (RLM and WLS).

## WLS observation weights

`prepare()` retains a `weight` column when present. When `model="wls"` and only masks are supplied, `run_discordance()` derives per-observation weights at fit time from `is_real` and `is_complete_missing`:

- **Measured** (`is_real=true`): weight `1.0`
- **Condition-wide imputed** (`is_real=false`, `is_complete_missing=true`): `config.wls_biological_weight` (default `0.5`)
- **Sparse imputed** (all other imputed entries): `1e-5`

Pass a precomputed `weight` column on the peptide table to skip mask derivation. RLM ignores weights at fit time even if provenance columns remain on `peptides`.

## Errors and exceptions

- **`ProteoForgeValidationError`:** failed design, duplicate keys, missing samples, WLS without sufficient provenance, `model='ebayes'`, normalization preconditions
- **`ProteoForgeIOError`:** bad file path or format (file entry points only)

Message text includes offending keys, sample IDs, or protein examples where applicable.

## Related pages

- [Configuration](config.md): YAML and design rules
- [Input and output](io.md): formats and harmonization
- [Normalization](normalization.md): transform details
- [PreparedDataset](prepared-dataset.md): output schema
- [Discordance](discordance.md): Module 2 entry point
- [Multiple-testing correction](correction.md): applied inside discordance
