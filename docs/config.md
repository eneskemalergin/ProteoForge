# Configuration

All pipeline inputs are driven by a frozen `Config` dataclass. Load from YAML, construct in Python, or round-trip through JSON-compatible dictionaries. Validation runs at construction time: invalid designs fail before any file I/O.

## Experimental design in `conditions`

The design lives in `Config.conditions`, not a separate design file. Each key is a condition name; each value is a list of sample IDs (replicates) for that condition.

Rules enforced at construction:

- At least **two** condition keys.
- At least **two** samples per condition.
- Each sample ID appears in **exactly one** condition.
- `control_condition` must match one of the condition keys.

`control_condition` is listed first in `condition_levels`. Sample order in outputs follows condition order, then sample order within each condition as written in YAML.

### Example YAML

```yaml
control_condition: control

conditions:
  control:
    - S1
    - S2
  treated:
    - S3
    - S4

min_peptides: 4
input_is_log2: false
model: rlm
fdr: 0.001
correction_within: bonferroni
correction_global: fdr_bh
```

### Python construction

```python
from proteoforge import Config

config = Config(
    control_condition="control",
    conditions={
        "control": ("S1", "S2"),
        "treated": ("S3", "S4"),
    },
    min_peptides=4,
)
```

Only samples listed under `conditions` are used. Extra samples in the peptide table are dropped with a warning. Missing configured samples raise `ProteoForgeValidationError`.

## Column mapping

Use `column_map` when source files use non-canonical column names. Each field names the **source** column in the file; ProteoForge renames to the canonical name on ingest.

Canonical names:

- `protein_id`, `peptide_id`, `sample_id`, `intensity` (required for I/O)
- `condition` (optional in files; always attached from `conditions` during `prepare()`)
- `is_real`, `is_complete_missing`, `weight` (optional provenance)

Example for a vendor export:

```yaml
column_map:
  protein_id: prot
  peptide_id: pep
  sample_id: run
  intensity: quant
```

Defaults match canonical names (identity mapping). Omitted `column_map` fields use the default from `ColumnMap`.

## Fields reference

### Used by `prepare()`

- `control_condition` (required): control condition key in `conditions`. Normalization subtracts each peptide's mean intensity in these samples.
- `conditions` (required): mapping of condition name to sample ID list. Defines scope and replicate structure.
- `min_peptides` (default `4`, minimum `2`): each protein must have at least this many unique `(protein_id, peptide_id)` pairs after scoping.
- `input_is_log2` (default `false`): when `false`, intensities are log2-transformed during normalization. Set `true` when the input is already log2-scaled.
- `column_map` (optional): source-to-canonical column renames for peptide tables.
- `model` (default `"rlm"`, one of `rlm`, `wls`, `ebayes`): affects validation and which optional columns are retained on `PreparedDataset`. `rlm` keeps provenance columns on `peptides` when present; `PreparedDataset.is_real` and `weight` return `None`. `wls` requires a `weight` column or both `is_real` and `is_complete_missing` before `prepare()` completes. `ebayes` is rejected at `Config` construction (not implemented).
- `wls_biological_weight` (default `0.5`, range `(0, 1]`): weight for condition-wide imputed entries when `run_discordance()` derives WLS weights from masks (see [Prepare](prepare.md#wls-observation-weights)).

### Used by `run_discordance()`

- `fdr` (default `0.001`): global adjusted p-value threshold for `is_discordant`.
- `correction_within` (default `bonferroni`): within-protein method passed to `p_adjust_by_group`.
- `correction_global` (default `fdr_bh`): global method passed to `p_adjust` on within-protein adjusted values.
- `n_jobs` (default `-1`): parallel worker count for discordance shape groups and clustering (`-1` maps to `min(8, cpu_count // 2)`).

Allowed values for both correction fields: `bonferroni`, `holm`, `hommel`, `hochberg`, `fdr`, `fdr_bh`, `BY`, `qvalue`. See [Multiple-testing correction](correction.md) for control targets and usage notes. Independent hypothesis weighting (`ihw`) is implemented under `proteoforge.correction.ihw` but is not a config option yet.

These fields do not change normalization in `prepare()`.

### Used by `run_cluster()` and `assign_proteoforms()`

- `linkage` (default `"ward"`): hierarchical clustering linkage. Only `ward` is supported at runtime.
- `cut` (default `"hybrid_outlier"`): dendrogram cut strategy (`hybrid_outlier`, `fixed_height`, `dynamic_tree`). See [Clustering](clustering.md).
- `cluster_min_clusters` (default `1`): lower bound on cluster count from dendrogram k-selection.
- `cluster_max_clusters` (default `null`): upper bound on cluster count; `null` means no cap beyond peptide count.
- `fixed_n_clusters` (default `2`): target cluster count for `fixed_height` cut (combined with `cluster_min_clusters`).
- `hybrid_outlier_threshold` (default `0.05`): silhouette cutoff for singleton relabel in `hybrid_outlier` cut.
- `cluster_min_peptides` (default `2`): minimum peptides on a protein to run linkage; below this, assign a single cluster without Ward.

## Loading and serialization

```python
config = Config.from_yaml_path("config.yaml")
config = Config.from_yaml("control_condition: control\n...")
config = Config.from_dict({"control_condition": "control", "conditions": {...}})

yaml_text = config.to_yaml()
data = config.to_dict()
updated = config.replace(min_peptides=3)
```

JSON Schema for tooling and editors:

```python
schema = Config.to_json_schema()
config.write_json_schema("config.schema.json")
```

## Related pages

- [Multiple-testing correction](correction.md): `correction_within`, `correction_global`, `fdr`
- [Prepare](prepare.md): fields consumed at `prepare()`
- [Discordance](discordance.md): `model`, `fdr`, `n_jobs`
- [Clustering](clustering.md): linkage and cut fields
