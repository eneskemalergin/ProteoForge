# Multiple-testing correction

ProteoForge adjusts peptide interaction p-values in two steps during `run_discordance()`: within each protein, then across all peptides globally. The correction layer lives in `proteoforge.correction` (`p_adjust`, `p_adjust_by_group`). Config fields `correction_within` and `correction_global` select the method at each step.

## Pipeline integration

`run_discordance()` applies correction after one-vs-rest model fitting:

1. **Within protein:** `p_adjust_by_group(raw_p_value, protein_code, correction_within)` on finite p-values per protein.
2. **Global:** `p_adjust(within_p_value, correction_global, n_tests=<finite count>)` on the pooled within-protein values.

`is_discordant` is `True` when `adjusted_p_value <= config.fdr` (default `0.001`) and the value is finite.

Defaults match the reference analysis pair:

- `correction_within`: `bonferroni`
- `correction_global`: `fdr_bh`

See [Discordance](discordance.md) for table columns and [Configuration](config.md) for YAML fields.

## Methods available in config

Both `correction_within` and `correction_global` accept the same string labels (validated against `proteoforge.correction.VALID_METHODS` at `Config` construction):

- **`bonferroni`:** FWER (per step). Within default; most conservative unweighted step-up.
- **`holm`:** FWER. Step-down; less conservative than Bonferroni.
- **`hommel`:** FWER. Closed testing; R `p.adjust(..., "hommel")` parity.
- **`hochberg`:** FWER. Step-up; assumes positive dependence.
- **`fdr_bh`:** FDR. Benjamini-Hochberg; global default. (alias `fdr`)
- **`BY`:** FDR. Benjamini-Yekutieli; conservative under arbitrary dependence (case-sensitive in config).
- **`qvalue`:** FDR. Storey q-values with GCV pi0 (`proteoforge.correction.qvalue`).

`none` is valid for `p_adjust()` in library code but is **not** allowed in `Config`; discordance always runs both correction steps.

## Library API

For scripts and tests outside the discordance pipeline, imports work from the package root or from `proteoforge.correction`:

```python
import numpy as np
from proteoforge import VALID_METHODS, p_adjust, p_adjust_by_group
# equivalent: from proteoforge.correction import p_adjust, p_adjust_by_group

p = np.array([0.001, 0.04, 0.2, 0.8])
p_adjust(p, "fdr_bh")
p_adjust(p, "hommel", n_tests=10)  # pad to n_tests with p=1 when larger than len(p)

codes = np.array([0, 0, 1, 1], dtype=np.intp)
p_adjust_by_group(p, codes, "holm")
```

- **`p_adjust`:** vectorized adjustment; optional `n_tests` for Hommel padding (R `n` argument).
- **`p_adjust_by_group`:** independent adjustment per integer group code; non-finite inputs stay `NaN`.
- **`qvalue`:** no SciPy at runtime; pi0 from a shipped GCV spline grid.

Import path `proteoforge._correction` remains a backward-compatible re-export of `proteoforge.correction`.

## Choosing within vs global methods

Within-protein families are small (minimum four peptides per protein) and tests share the same design matrix, so p-values are positively dependent. Bonferroni within is strict but matches reference golden configs. Hommel or Holm within can reduce conservatism while staying FWER-minded.

The global pool has thousands of peptides. `fdr_bh` at `fdr=0.001` is already a tight discovery bar. `qvalue` global can increase power when many nulls are expected. `BY` global is appropriate when you want a more conservative FDR statement across weakly dependent proteins.

The two-step procedure is the shipped contract; it is not identical to a single pooled BH pass on raw p-values.

## Independent hypothesis weighting (IHW)

`proteoforge.correction.ihw.adjust_ihw()` implements Bioconductor-style IHW: covariate strata, cross-validated weight learning, and weighted Benjamini-Hochberg (or Bonferroni). It is **not** wired into `Config` or `run_discordance()` yet. Call it from library code when you have a covariate aligned with each p-value.

```python
import numpy as np
from proteoforge.correction.ihw import adjust_ihw

rng = np.random.default_rng(1)
result = adjust_ihw(pvalues, covariates, alpha=0.1, seed=1, rng=rng)
result.adj_pvalues
result.weights
result.folds
```

The covariate must be independent of the p-value under the null (for example mean normalized intensity per peptide). Plan on a global pool with hundreds or thousands of hypotheses; within-protein families are too small for stable bin weights.

`IHWResult` holds `adj_pvalues`, `weights`, `weighted_pvalues`, `groups` (bin index per hypothesis), `folds`, and run settings (`alpha`, `nbins`, `nfolds`, `penalty`, `adjustment_type`). With fixed `seed`, `rng`, and optional `folds`, Python results are reproducible on the same machine.

### Regularization grid (`lambdas`)

IHW learns a weight for each covariate bin, then smooths that weight profile so it does not overfit noise. The smoothing strength is controlled by a regularization parameter λ (lambda): lower λ allows sharper weight changes across bins; higher λ favors smoother profiles; λ = ∞ uses the maximum-flexibility convex fit used in most parity checks.

**`lambdas="auto"` (default)**

:   Build the same candidate grid as Bioconductor IHW from `nbins` and fixed anchors (including 0 and ∞). On each outer CV fold, nested cross-validation picks the candidate with the most rejections on held-out data. All candidates on one random split share the same internal fold assignment.

**Explicit array**

:   Pass your own candidates, for example `np.array([np.inf])` for a single strength or `[0.0, 1.0, np.inf]` for a custom grid. With one value, inner λ selection is skipped.

For a fixed, fast run, use one explicit λ (often `np.inf`), plus `seed` and optional `folds`. For the full grid search, keep `lambdas="auto"`. Defaults are `nfolds=5`, `nfolds_internal=5`, `nsplits_internal=1`, and `nbins="auto"`.

### Random folds and adjusted p-values

!!! warning "IHW adjusted p-values depend on random cross-validation"

    `adjust_ihw()` uses randomness in three places:

    - tie-breaking when binning ordinal covariates
    - assignment of hypotheses to CV folds (default `nfolds=5`)
    - nested splits when choosing λ from the grid

    **Same method, same data:** fixed `seed`, `rng`, and optional pre-specified `folds` give reproducible Python results. Different seeds or fold assignments change weights and adjusted p-values slightly. That variability is part of the estimator, not a bug.

    **Python vs Bioconductor:** adjusted p-values match at machine precision when folds and binning match (for example `nfolds=1` with the same `seed`). With default multi-fold CV, R and Python draw different random folds, so full outputs will not match bit-for-bit even when the LP core agrees. Compare with prespecified `folds`, or treat small differences as expected.

    **Different correction methods:** BH, q-value, and IHW target different procedures. Adjusted p-values agree only in degenerate cases (for example uniform IHW weights reducing to BH). Do not expect `fdr_bh` and `adjust_ihw()` to return the same numbers on the same raw p-values.

    For production runs, record `seed`, `nfolds`, and returned `folds` / `weights` in metadata when IHW is enabled. `nfolds=1` learns weights without proper pre-validation; R documents it for weight exploration only, not for final testing.

## Parity notes

Classical methods (`holm`, `hommel`, `hochberg`, `fdr_bh`, `BY`) follow R `p.adjust` where applicable. The shipped unit tests cover IHW weights, the convex LP, and standard adjustment paths; full Bioconductor cross-checks are a maintainer workflow outside the default test suite.

## Related pages

- [Discordance](discordance.md): where correction runs in Module 2
- [Configuration](config.md): `correction_within`, `correction_global`, `fdr`
- [PreparedDataset](prepared-dataset.md): handoff before p-values exist
- [Clustering](clustering.md): discordance stringency sets Module 3 scope
- [Prepare](prepare.md): upstream of raw p-values
