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
- `model` (default `"rlm"`, one of `rlm`, `wls`, `ebayes`): affects validation and which optional columns are retained on `PreparedDataset`. `rlm` keeps provenance columns on `peptides` when present; `PreparedDataset.is_real` and `weight` return `None`. `wls` requires a `weight` column or both `is_real` and `is_complete_missing` before `prepare()` completes. `ebayes` is rejected at `Config` construction (not implemented in v0.0.2).
- `wls_biological_weight` (default `0.5`, range `(0, 1]`): weight for condition-wide imputed entries when deriving WLS weights from masks.

### Used by `run_discordance()` (v0.0.2)

- `fdr` (default `0.001`): global adjusted p-value threshold for `is_discordant`.
- `correction_within` (default `bonferroni`): within-protein correction method.
- `correction_global` (default `fdr_bh`): global correction method.
- `n_jobs` (default `-1`): parallel worker count for shape-group discordance (`-1` maps to `min(8, cpu_count // 2)`).

These fields do not change normalization in `prepare()`.

### Reserved (not yet implemented)

Validated in config but not consumed by the current release:

- `linkage` (default `"ward"`): hierarchical clustering linkage.
- `cut` (default `"hybrid_outlier"`): cluster count strategy (`hybrid_outlier`, `fixed_height`, `dynamic_tree`).

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
