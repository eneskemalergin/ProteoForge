# PreparedDataset

Output of `prepare()` and `prepare_from_parquet()`. Validated, control-relative normalized peptide intensities in long format. Downstream modeling should treat this object as the single source of truth after prepare: no re-parsing of raw files required.

## Object fields

**`config`**
The `Config` instance used during prepare (frozen copy of design and options).

**`peptides`**
Polars `DataFrame`, one row per `(protein_id, peptide_id, sample_id)`. Column contract below.

**`sample_ids`**
Tuple of sample IDs in condition order: control condition first, then remaining conditions in config order, samples in YAML list order within each condition.

**`condition_levels`**
Tuple of condition names with control first.

**`protein_index`**
Length `peptides.height`. Integer protein ordinal per row for grouping (same protein shares one index value across peptides and samples).

**`metadata`**
Dict with run statistics: `n_proteins`, `n_peptides`, `n_samples`, `nan_fraction`, `control_condition`, `conditions_used`, `samples_used`, `samples_dropped`.

## `peptides` columns

- **`protein_id`, `peptide_id`, `sample_id`** (required): primary key; unique after validation
- **`condition`** (required): from `Config.conditions`, not from raw input
- **`intensity`** (required): raw input intensity (log2 or linear per config)
- **`intensity_normalized`** (required): control-relative normalized value
- **`is_real`, `is_complete_missing`, `weight`** (optional): retained when supplied; `PreparedDataset` array properties expose them only for `wls` (not `rlm`)

## Keys and row order

Identity is `(protein_id, peptide_id, sample_id)`, not row index.

Row order matches the validated input (not sorted by key). Join and group on key columns.

`run_discordance()` returns a separate `DiscordanceResult.table` sorted by `(protein_id, peptide_id)`. Merge back to `peptides` on those keys, never by row index. See [Discordance](discordance.md).

Only samples listed under `config.conditions` are kept. Others are dropped with a warning recorded in `metadata["samples_dropped"]`.

## Properties

**`n_peptides`**
Count of unique `(protein_id, peptide_id)` pairs.

**`n_samples`**
Length of `sample_ids`.

**`n_proteins`**
From `metadata["n_proteins"]`.

**`intensity_normalized`**
1-D `float64` NumPy array aligned to `peptides` rows (same order as the DataFrame).

**`is_real`**, **`is_complete_missing`**, **`weight`**
Return aligned NumPy arrays when `config.model` is `wls` and the column exists on `peptides`. Return `None` for `rlm` (columns may still be present on `peptides` for later WLS runs).

## Normalization

Values in `intensity_normalized` come from [control-relative normalization](normalization.md): optional log2, per-sample z-score across peptides, subtract peptide mean in the control condition.

## Downstream use

Pass a `PreparedDataset` to `run_discordance()` for peptide-level interaction testing (Module 2, v0.0.2). See [Discordance](discordance.md).

## Related pages

- [Prepare](prepare.md): how `PreparedDataset` is built
- [Discordance](discordance.md): Module 2 entry point
- [Configuration](config.md): design and model options
- [Input and output](io.md): input columns and provenance
