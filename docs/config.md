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

### Used by `prepare()` today

**`control_condition`** (required)  
Control condition key in `conditions`. Normalization subtracts each peptide's mean intensity in these samples.

**`conditions`** (required)  
Mapping of condition name to sample ID list. Defines scope and replicate structure.

**`min_peptides`** (default `4`, minimum `2`)  
Each protein must have at least this many unique `(protein_id, peptide_id)` pairs after scoping.

**`input_is_log2`** (default `false`)  
When `false`, intensities are log2-transformed during normalization. Set `true` when the input is already log2-scaled.

**`column_map`** (optional)  
Source-to-canonical column renames for peptide tables.

**`model`** (default `"rlm"`, one of `rlm`, `wls`, `ebayes`)  
Affects validation and which optional columns are retained on `PreparedDataset`:

- `rlm`: provenance columns are dropped from the handoff unless present in input (not exposed as array properties).
- `wls`: requires provenance (`is_real` / `is_complete_missing`) or `weight` on the peptide table before prepare completes.
- `ebayes`: same provenance column retention as WLS for properties; weight derivation is not implemented in v0.0.1.

**`wls_biological_weight`** (default `0.5`, range `(0, 1]`)  
Reserved for WLS weight construction in a future release. Stored and validated only.

### Reserved for v0.1.0 (validated now, not consumed by `prepare()`)

These fields are part of the shared config schema for the full discovery pipeline. Setting them has no effect on normalization in v0.0.1:

- **`fdr`** (default `0.001`): FDR threshold for discordance testing
- **`linkage`** (default `"ward"`): hierarchical clustering linkage
- **`cut`** (default `"hybrid_outlier"`): cluster count strategy (`hybrid_outlier`, `fixed_height`, `dynamic_tree`)
- **`n_jobs`** (default `-1`): parallel worker count (`-1` means all cores)

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

PyYAML is a core dependency (`pyyaml` in project requirements).

## Deprecated: standalone design files

`read_design()` and separate CSV design files are deprecated. Put the sample-to-condition map in `Config.conditions`. `design_from_frame()` remains for tests and ad hoc Polars tables.

## Errors

Invalid config raises `ProteoForgeValidationError` with the failing field named in the message (unknown control key, duplicate sample across conditions, too few replicates, and similar).
