# Reference vs ProteoForge comparison

Local harness for normalization parity and end-to-end performance against the manuscript reference implementation. Uses [zebrac](https://github.com/eneskemalergin/zebrac) for wall time and peak RSS.

**Reference base:** [`against_condition`](https://github.com/LangeLab/ProteoForge_Analysis/blob/main/ProteoForge/normalize.py) from [LangeLab/ProteoForge_Analysis](https://github.com/LangeLab/ProteoForge_Analysis) (v0.7.0). Check out locally as `ref/ProteoForge_analysis_src/normalize.py`.

## Why

ProteoForge reimplements control-relative normalization in Polars/NumPy. Unit tests and golden fixtures check internal consistency, but they do not prove equivalence to the original manuscript code.

This harness exists to:

- Verify numerical parity on every `(protein_id, peptide_id, sample_id)`, not aggregate statistics alone.
- Profile both sides as standalone end-to-end scripts (read, normalize, write parquet) under the same zebrac conditions.
- Keep reference imports out of `src/proteoforge/` and `tests/`.

## What

Both runners take the same long-format input parquet and config YAML and write a long parquet with `intensity_normalized`:

- **Reference** (`run_reference.py`): scope samples, pandas, `against_condition()`, parquet.
- **ProteoForge** (`run_proteoforge.py`): `prepare_from_parquet()` (validate + `normalize_control_relative_long`), parquet.

`zebrac.sh` runs both commands, checks parity with `diff_outputs.py`, and prints a timing summary.

Default fixture: `benchmarks/fixtures/complete/complete-real.parquet` (~467k rows, 2,217 peptides × 21 samples, three conditions). Synthetic tiers (`medium`, `large`, `xlarge`) live under `benchmarks/fixtures/synthetic/` after generation.

## How

```text
input.parquet + config.yaml
        │
        ├─► run_reference.py  ──► reference/normalized.parquet ─┐
        │                                                        ├─► diff_outputs.py
        └─► run_proteoforge.py ──► proteoforge/normalized.parquet ─┘
                    ▲
              zebrac profiles each runner
```

**Requirements:** `tools/zebrac`, local `ref/ProteoForge_analysis_src/normalize.py`, fixture parquet + config.

```bash
tools/compare/validate.sh   # parity checker self-tests (no ref/ needed)
tools/compare/zebrac.sh     # full ref vs PF (needs ref/ + fixtures)
```

```bash
# Complete-real fixture (default)
tools/compare/zebrac.sh

# Synthetic scaling tiers
uv run python benchmarks/synthetic/generate.py
tools/compare/zebrac.sh --cases medium large xlarge

# Custom input
tools/compare/zebrac.sh --input data.parquet --config config.yaml --case my-run
```

Each case writes under `tmp/compare/<case>/`: reference and ProteoForge normalized parquets, plus `zebrac.json`.

## Validation history

Exploration done through this harness before adopting the current production path:

- **End-to-end vs step-mixed profiling:** early profiles compared unlike workloads; fixed by matching full pipelines in both runners.
- **Wide pivot path** (pivot, NumPy, melt): correct but slow; kept as a numerical baseline during layout exploration.
- **Long Polars windows** (`normalize_control_relative_long`): same math as wide/NumPy; adopted as production normalize.
- **Join-based long normalize:** rejected. Hash joins on ~465k+ rows were slower and used more RSS than window ops at ~21 samples.
- **NumPy long kernel:** rejected. Much slower than Polars windows at all synthetic tiers.
- **Streaming collect:** rejected. Marginal speed gain, significantly higher RSS.
- **Unsorted parquet write:** accepted. About 8 to 11% faster write with identical per-key values (identity is key columns, not row order).
- **Long handoff in `PreparedDataset`:** production `prepare()` keeps a long `peptides` table with no pivot/melt detour.

Production normalize: Polars window z-score per sample, join control mean per `(protein_id, peptide_id)`, subtract. Implemented in `normalize_control_relative_long()`.

## Parity validation

Parity is enforced by `compare_parity()` in `diff_outputs.py`. `zebrac.sh` calls it with `--atol 1e-11`.

Rules, in order:

1. Reject duplicate keys on either side.
2. Anti-join: fail if any key exists only in reference or only in ProteoForge.
3. Inner join on `(protein_id, peptide_id, sample_id)` with `validate="1:1"`.
4. Compare `intensity_normalized` element-wise with `np.isclose` (default `rtol=1e-10`, `atol=1e-12`; zebrac uses `atol=1e-11`).
5. On failure: report worst keys with reference vs ProteoForge values.

Matching means, medians, or row order after sort does not count as parity. Self-tests in `validation/test_parity.py` cover swapped keys, duplicate keys, and missing keys.

**Current result on `complete-real`** (467,187 observations):

- Keys compared (1:1): 467,187
- Mismatched keys: 0
- Max abs diff: 1.17×10⁻¹³
- Mean abs diff: 3.59×10⁻¹⁴
- Tolerance: `rtol=1e-10`, `atol=1e-12`

**PASS:** every `(protein_id, peptide_id, sample_id)` matches within tolerance.

## Performance

zebrac reports median wall time and median peak RSS over repeated runs. Both runners include subprocess and `uv run` overhead so numbers reflect real invocations, not in-process micro-benchmarks.

### End-to-end on `complete-real` (467k rows)

| | Reference | ProteoForge | Δ |
| --- | --- | --- | --- |
| Wall time | 1,411 ms | 794 ms | ~44% faster |
| Peak RSS | 370 MB | 412 MB | +11% |

ProteoForge trades modest memory for avoiding the reference pandas pivot and melt path while running validation and long-format normalize in one pass.

### Normalize layout exploration on synthetic tiers

Candidates were compared at the normalize step on the same scoped long input. Median wall ms / peak RSS MB:

| Method | medium (~466k rows) | large (~931k) | xlarge (~1.86M) |
| ------ | ------------------- | ------------- | --------------- |
| **long_window** (production) | 494 / 395 | 594 / 606 | 848 / 941 |
| pivot, NumPy, melt (baseline) | 734 / 308 | 1,129 / 490 | 1,903 / 736 |
| long_numpy | 1,632 / 329 | 3,253 / 540 | 6,559 / 884 |

All long-window candidates passed parity vs the pivot-melt baseline at `atol=1e-11` on every tier. **long_window** was chosen as production (best balance of speed and RSS among parity-preserving long paths).

Reproduce end-to-end numbers:

```bash
tools/compare/zebrac.sh --case complete
tools/compare/zebrac.sh --cases medium large xlarge
```

Results land in `tmp/compare/<case>/zebrac.json`. `summarize.py` prints the reference vs ProteoForge table.
