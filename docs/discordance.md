# Discordance (Module 2)

Peptide-level interaction testing: for each protein, fit one-vs-rest models (`Intensity ~ Condition × Peptide`) and test whether a target peptide diverges from its siblings across conditions.

## Entry point

```python
from proteoforge import Config, prepare_from_parquet, run_discordance

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)
result = run_discordance(dataset)

result.table       # one row per peptide
result.discordant  # rows where is_discordant is true
result.metadata    # counts, parallelism, skip_reason_counts
```

`run_discordance()` also accepts an in-memory `PreparedDataset` from `prepare()`.

Optional arguments:

- `n_jobs`: overrides `config.n_jobs` for this call
- `show_progress=True`: tqdm progress bar in an interactive terminal or notebook (auto-suppressed in CI or when `TQDM_DISABLE` is set)

## Models

- `rlm` (default): Huber IRLS. Provenance columns optional; retained on `peptides` when present.
- `wls`: mask-derived or precomputed weights. Requires `weight` or both `is_real` and `is_complete_missing` at `prepare()`.
- `ebayes`: not implemented. Rejected at `Config` construction.

Set `model` in [Configuration](config.md).

## Multiple testing

Correction is two-step: within protein, then global.

Defaults:

- `correction_within`: Bonferroni
- `correction_global`: Benjamini-Hochberg (`fdr_bh`)

`fdr` (default `0.001`) sets the global adjusted p-value cutoff for `is_discordant`. Other methods (`holm`, `hochberg`, `fdr_bh`, `BY`) are available via config.

## Parallelism

Discordance groups proteins by design shape (same condition layout and peptide count), then fits each group in parallel when `n_jobs > 1`.

- `n_jobs=-1` (default) maps to `min(8, cpu_count // 2)`
- `n_jobs=1` runs serially (useful for debugging)
- Worker pools shut down cleanly on errors or interrupts

## Output

`DiscordanceResult.table` columns:

- `protein_id`, `peptide_id`: peptide keys
- `raw_p_value`: uncorrected interaction p-value
- `within_p_value`: after within-protein correction
- `adjusted_p_value`: after global correction
- `is_discordant`: passes global FDR threshold
- `fit_status`: skip reason when a fit is not usable

`result.table` is sorted by `(protein_id, peptide_id)`. `PreparedDataset.peptides` keeps input row order. **Join on keys, not row index:**

```python
joined = dataset.peptides.join(
    result.table,
    on=["protein_id", "peptide_id"],
    how="left",
)
```

## Related pages

- [PreparedDataset](prepared-dataset.md): input to discordance
- [Configuration](config.md): `model`, `fdr`, `n_jobs`, correction fields
- [Prepare](prepare.md): builds the handoff object
