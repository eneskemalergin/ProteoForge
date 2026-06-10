# Normalization

Module 1 in the ProteoForge method: control-relative normalization of peptide intensities before discordance modeling. There is no separate `run_normalize()` call. The production path is `normalize_control_relative_long()` inside `prepare()`, which writes `intensity_normalized` on `PreparedDataset.peptides`.

Input: long-format table with `protein_id`, `peptide_id`, `sample_id`, `intensity`, and control samples identified by config.

Output: same rows plus `intensity_normalized`.

## Transform steps

Applied in order for each observation:

1. **Log2** (optional): when `input_is_log2=false`, compute `log2(intensity)`. Non-positive values raise `ProteoForgeValidationError`.
2. **Per-sample z-score**: for each `sample_id`, subtract the sample mean and divide by the sample standard deviation across all peptides (population std, ddof=1, matching reference `against_condition`).
3. **Control baseline removal**: for each `(protein_id, peptide_id)`, subtract the mean z-score across control-condition samples.

After step 3, control samples for a peptide sum to zero in the normalized scale (up to floating-point tolerance). The transform prepares data for `Intensity ~ Condition × Peptide` by removing run effects and centering each peptide on its control baseline.

## Configuration knobs

Only two config fields affect normalization today:

- **`control_condition`**: which samples in `conditions` define the baseline in step 3
- **`input_is_log2`**: skip log2 when intensities are already log-transformed

`model`, `fdr`, clustering options, and parallel settings do not change normalization. They apply in `run_discordance()`, `run_cluster()`, and `assign_proteoforms()`.

## Implementation notes

The long-format implementation uses Polars window expressions and a grouped join for control means. It does not pivot to wide layout during `prepare()`. Row order matches the validated input; identity is `(protein_id, peptide_id, sample_id)`.

A wide NumPy implementation (`normalize_control_relative()`) exists for unit tests and regression checks. It is not used in the prepare pipeline.

## Failure modes

- **Empty control sample list:** error
- **Non-positive intensity with `input_is_log2=false`:** error
- **Zero standard deviation for any sample after log2:** error (constant or degenerate sample column)

## Numerical reference

Normalization matches `against_condition` in the [ProteoForge analysis repository](https://github.com/LangeLab/ProteoForge_Analysis/blob/main/ProteoForge/normalize.py) (v0.7.0). Small regression tests in `tests/test_golden_normalize.py` lock the algorithm on committed fixtures.

## Output column

**`intensity_normalized`**: `float64`, one value per row, appended to the peptide table inside `PreparedDataset.peptides`. Access as a column or via the `PreparedDataset.intensity_normalized` property (1-D NumPy array aligned to row order).

See [PreparedDataset](prepared-dataset.md) for the full handoff contract. Module 2 reads `intensity_normalized` in [Discordance](discordance.md). Multiple-testing correction runs after model fitting; see [Multiple-testing correction](correction.md).
