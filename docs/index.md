# ProteoForge documentation

ProteoForge discovers differential proteoforms from an imputed peptide matrix and a condition design with a control. The installable package covers configuration, long-format peptide I/O, validation, control-relative normalization inside `prepare()` (Module 1), peptide discordance with RLM and WLS backends (Module 2), two-step multiple-testing correction, Ward clustering, and dPF assignment (Module 3). The unified `discover()` API and HTML report are not implemented yet.

## Shipped today vs planned

**Shipped**

- `Config`, YAML/JSON loading, column mapping
- `read_peptides()`, `prepare()`, `prepare_from_parquet()`
- `run_discordance()` with RLM and WLS, two-step correction (`bonferroni` / `fdr_bh` defaults, plus `holm`, `hommel`, `hochberg`, `BY`, `qvalue`)
- `p_adjust()`, `p_adjust_by_group()`, `VALID_METHODS` on the top-level `proteoforge` import
- `proteoforge.correction.ihw.adjust_ihw()` (library only, not wired into `Config` yet)
- `run_cluster()`, `assign_proteoforms()`, Numba clustering geometry
- `load_fixture_bundle()` for committed test fixtures

**Planned**

- `discover()`, `ProteoformResults`, HTML report, Typer CLI (`proteoforge discover`)
- `model="ebayes"`
- IHW as a `correction_global` (or weighted) config option
- Vendor-wide ingest, plotting extras beyond optional `plots` dependency

See [Changelog](https://github.com/eneskemalergin/ProteoForge/blob/main/CHANGELOG.md) for release history.

## Pipeline

```mermaid
flowchart LR
  IN["Imputed peptide matrix\n+ condition design"]
  N["1. Normalize"]
  D["2. Discordance"]
  C["3. Cluster"]
  P["4. dPF assign"]
  OUT["ProteoformResults"]

  IN --> N
  N --> D
  D --> C
  C --> P
  P --> OUT

  style N fill:#059669,color:#fff
  style D fill:#059669,color:#fff
  style C fill:#059669,color:#fff
  style P fill:#059669,color:#fff
  style OUT fill:#64748b,color:#fff
```

Modules 1 to 3 (green) run through `prepare()`, `run_discordance()`, `run_cluster()`, and `assign_proteoforms()`. Module 4 (grey) is not available yet.

Stage map (function, primary output):

- **Ingest and normalize:** `prepare()` / `prepare_from_parquet()` → `PreparedDataset` with `intensity_normalized` on each row
- **Discordance:** `run_discordance()` → `DiscordanceResult.table` with `raw_p_value`, `within_p_value`, `adjusted_p_value`, `is_discordant`
- **Cluster:** `run_cluster()` → `ClusterResult.table` with `cluster_id` on discordant proteins
- **dPF assign:** `assign_proteoforms()` → `ProteoformMappingResult.table` with `dpf_id`

Correction runs inside `run_discordance()`; see [Multiple-testing correction](correction.md).

## Reading order

1. [Configuration](config.md): experimental design, column mapping, correction and clustering fields
2. [Input and output](io.md): supported formats, canonical columns, provenance
3. [Prepare](prepare.md): `prepare()` and `prepare_from_parquet()`
4. [PreparedDataset](prepared-dataset.md): handoff contract before Module 2
5. [Normalization](normalization.md): control-relative transform inside prepare (Module 1 detail)
6. [Discordance](discordance.md): `run_discordance()`, models, batching, outputs
7. [Multiple-testing correction](correction.md): method list, two-step logic, IHW library notes
8. [Clustering](clustering.md): `run_cluster()` and `assign_proteoforms()` (Module 3)

## Quick example

```python
from proteoforge import (
    Config,
    assign_proteoforms,
    prepare_from_parquet,
    run_cluster,
    run_discordance,
)

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)
discordance = run_discordance(dataset)
clusters = run_cluster(dataset, discordance)
mapping = assign_proteoforms(dataset, discordance, clusters)

mapping.n_differential_peptides
```

Pass the same `Config` (or matching frozen values) on `PreparedDataset`, `DiscordanceResult`, and downstream results. `run_cluster()` and `assign_proteoforms()` compare configs and raise on mismatch.

## Project links

- [Repository README](https://github.com/eneskemalergin/ProteoForge)
- [Changelog](https://github.com/eneskemalergin/ProteoForge/blob/main/CHANGELOG.md)
- [License](https://github.com/eneskemalergin/ProteoForge/blob/main/LICENSE) (MIT)
