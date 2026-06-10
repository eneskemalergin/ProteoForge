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

## Parity notes

All config-selectable methods are validated for production use:

- **R `p.adjust` parity:** `bonferroni`, `holm`, `hommel`, `hochberg`, `fdr_bh`, `BY` (unit tests; `hommel` honors the R `n` padding argument via `n_tests`).
- **`qvalue`:** Storey step-up with GCV pi0 on the shipped lambda grid (no SciPy at runtime). Matches Bioconductor `qvalue` on rejection counts across synthetic and real DE p-value fixtures; bit-identical on airway-scale data when pi0 clips to 1.

Dev tests with the optional `ref` dependency group also assert `qvalue` against a scipy smoothing-spline reference implementation.

## Related pages

- [Discordance](discordance.md): where correction runs in Module 2
- [Configuration](config.md): `correction_within`, `correction_global`, `fdr`
- [PreparedDataset](prepared-dataset.md): handoff before p-values exist
- [Clustering](clustering.md): discordance stringency sets Module 3 scope
- [Prepare](prepare.md): upstream of raw p-values
