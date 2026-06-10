# Discordance (Module 2)

Peptide-level interaction testing: for each protein, fit one-vs-rest models (`Intensity ~ Condition × Peptide`) and test whether a target peptide diverges from its siblings across conditions. Raw p-values pass through two-step correction (within protein, then global) before `is_discordant` is set. See [Multiple-testing correction](correction.md) for method names and defaults.

## Entry point

```python
from proteoforge import Config, prepare_from_parquet, run_discordance

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)
result = run_discordance(dataset)

result.table          # one row per peptide
result.discordant     # rows where is_discordant is true
result.n_discordant   # count of discordant peptides
result.metadata       # model, correction pair, batching, fit counts
```

`run_discordance()` accepts a `PreparedDataset` from `prepare()` or `prepare_from_parquet()`. The `config` on that object must match the config you intend for correction fields (`fdr`, `correction_within`, `correction_global`, `model`).

Optional arguments:

- `batching`: fitting layout (`"shape"` default, `"protein"`, `"scalar"`). All three return the same p-values; `shape` enables shape-group parallelism when `n_jobs > 1`.
- `n_jobs`: overrides `config.n_jobs` for this call (shape batching only)
- `show_progress=True`: tqdm progress bar in an interactive terminal or notebook (auto-suppressed in CI or when `TQDM_DISABLE` is set)

## Models

- `rlm` (default): Huber IRLS. Provenance columns optional; retained on `peptides` when present.
- `wls`: mask-derived or precomputed weights. Requires `weight` or both `is_real` and `is_complete_missing` at `prepare()`. Weight tiers are documented under [Prepare](prepare.md#wls-observation-weights).
- `ebayes`: not implemented. Rejected at `Config` construction.

Set `model` in [Configuration](config.md).

## Multiple testing

Correction is two-step: within protein, then global. Defaults are `correction_within: bonferroni` and `correction_global: fdr_bh`. `fdr` (default `0.001`) sets the cutoff on `adjusted_p_value` for `is_discordant`.

Supported config methods include `holm`, `hommel`, `hochberg`, `fdr_bh`, `BY`, and `qvalue` at either step. Method semantics are in [Multiple-testing correction](correction.md).

## Parallelism

With `batching="shape"` (default), proteins that share the same design shape (condition layout and peptide count) are fitted together. When `n_jobs > 1`, shape groups run in parallel.

- `n_jobs=-1` (default) maps to `min(8, cpu_count // 2)`
- `n_jobs=1` runs serially (useful for debugging)
- Worker pools shut down cleanly on errors or interrupts
- In notebooks or non-interactive stdin, parallelism may fall back to serial; see `metadata["parallel_fallback"]`

`batching="protein"` or `"scalar"` always fits serially regardless of `n_jobs`.

## Output

`DiscordanceResult.table` columns:

- `protein_id`, `peptide_id`: peptide keys
- `raw_p_value`: uncorrected interaction p-value
- `within_p_value`: after within-protein correction
- `adjusted_p_value`: after global correction
- `is_discordant`: `adjusted_p_value <= config.fdr` and finite
- `fit_status`: per-peptide fit outcome (see below)

`result.table` is sorted by `(protein_id, peptide_id)`. `PreparedDataset.peptides` keeps input row order. Join on keys, not row index:

```python
joined = dataset.peptides.join(
    result.table,
    on=["protein_id", "peptide_id"],
    how="left",
)
```

### `fit_status` values

- `ok`: interaction p-value computed
- `rank_deficient`: design rank too low for the interaction test
- `insufficient_df`: not enough residual degrees of freedom
- `ill_conditioned`: numerically unstable design
- `zero_scale`: zero residual scale in the weighted path
- `zero_robust_scale`: zero robust scale in RLM
- `wald_failed`: Wald test could not be formed

Non-`ok` rows may carry `NaN` p-values and are excluded from finite correction pools.

### `metadata` fields

Common keys on `result.metadata`:

- `model`, `correction_within`, `correction_global`, `fdr`, `batching`
- `n_proteins`, `n_peptides_tested`, `n_peptides_flagged`, `n_peptides_skipped`
- `skip_reason_counts`: dict of `fit_status` to count
- `n_jobs_requested`, `n_jobs_effective`, `parallel_applicable`, `n_shape_groups`
- `parallel_fallback`, `parallel_fallback_reason` (when shape parallelism was requested but serial fitting ran)

## Downstream

`run_cluster()` uses `is_discordant` to choose proteins for linkage. Stricter correction (for example `bonferroni` within and low `fdr`) reduces the discordant set and therefore Module 3 scope. See [Clustering](clustering.md).

## Related pages

- [PreparedDataset](prepared-dataset.md): input to discordance
- [Multiple-testing correction](correction.md): methods, defaults, two-step flow
- [Clustering](clustering.md): downstream Module 3 on discordant proteins
- [Configuration](config.md): `model`, `fdr`, `n_jobs`, correction fields
- [Prepare](prepare.md): builds the handoff object
- [Normalization](normalization.md): upstream `intensity_normalized` values
