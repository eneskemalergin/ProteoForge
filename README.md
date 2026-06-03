<!-- markdownlint-disable MD033 MD036 MD041 MD045 -->
<p align="center">
    <!-- <img src="assets/icon.jpg" alt="ProteoForge logo" /> -->
    <strong>ProteoForge</strong>
</p>

<p align="center">
    <strong>Differential proteoform discovery for bottom-up proteomics</strong>
</p>

<p align="center">
    <a href="#references"><img src="https://img.shields.io/badge/paper-bioRxiv%202025-7c3aed?style=for-the-badge" alt="Paper" /></a>
    <a href="docs/index.md"><img src="https://img.shields.io/badge/docs-0891b2?style=for-the-badge" alt="Documentation" /></a>
    <a href="https://github.com/eneskemalergin/ProteoForge/actions"><img src="https://img.shields.io/github/actions/workflow/status/eneskemalergin/ProteoForge/ci.yml?branch=main&style=for-the-badge&logo=github&label=CI" alt="CI" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-059669?style=for-the-badge" alt="MIT License" /></a>
    <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-64748b?style=for-the-badge" alt="Changelog" /></a>
</p>
<p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12--3.15-f59e0b?style=for-the-badge" alt="Python 3.12-3.15" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/numpy-2.2%2B-2563eb?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy 2.2+" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/polars-1.26%2B-cd7c2f?style=for-the-badge&logo=polars&logoColor=white" alt="Polars 1.26+" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/scipy-1.15%2B-8b5cf6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy 1.15+" /></a>
</p>
<p align="center">
    <a href="https://github.com/eneskemalergin/ProteoForge"><img src="https://img.shields.io/badge/status-in%20development-0f766e?style=for-the-badge" alt="In development" /></a>
</p>

ProteoForge discovers differential proteoforms from an imputed peptide matrix and a condition design with a control. Peptides that break rank with their siblings are grouped into dPF units: canonical signal (`dPF_0`), multi-peptide proteoforms (`dPF_1+`), and singleton discordants (`dPF_-1`).

The package does not impute, search, or quantify. Upstream imputation is required.

**Available now (v0.0.1):** long-format peptide I/O, validation, and control-relative normalization via `prepare()`.

**In progress toward v0.1.0:** discordance modeling, clustering, dPF assignment, `discover()`, CLI, and PyPI release.

## Pipeline

Four modules from the ProteoForge method. Module 1 is implemented; the rest are planned for v0.1.0.

```mermaid
flowchart LR
  IN["Imputed peptide matrix\n+ condition design"]
  N["1. Normalize\ncontrol-relative intensities"]
  D["2. Discordance\nIntensity ~ Condition × Peptide"]
  C["3. Cluster\ndiscordant peptides"]
  P["4. dPF assign\nmapping + quantities"]
  OUT["ProteoformResults\nmapping, summary, exports"]

  IN --> N
  N --> D
  D --> C
  C --> P
  P --> OUT
```

**Planned capabilities for v0.1.0**

- RLM default; WLS (imputation-aware weights) and empirical-Bayes backends
- Dynamic hierarchical clustering (`hybrid_outlier_cut`)
- Peptide-to-dPF mapping and collapsed dPF quantity matrix for downstream tools such as [QuEStVar](https://github.com/eneskemalergin/QuEStVar)
- FASTA, peptide positions, UniProt and iPTMnet overlays; HTML report and core plots
- Python API, YAML config, and Typer CLI

## Installation

Python 3.12 to 3.15. Runtime: NumPy 2.2+, Polars 1.26+, SciPy 1.15+.

PyPI follows v0.1.0. Until then, install from source:

```bash
git clone https://github.com/eneskemalergin/ProteoForge.git
cd ProteoForge
uv sync
```

Optional extras: `plots`, `interactive`, `accel`, `cli`, `docs`.

```bash
pip install -e ".[cli,plots]"
```

## Quick start

### Available now

Load a long-format peptide parquet, validate, and normalize in one call. Experimental design and sample scope live in the config YAML, not a separate design file.

```python
from proteoforge import Config, prepare_from_parquet

config = Config.from_yaml_path("config.yaml")
dataset = prepare_from_parquet("peptides.parquet", config)

dataset.peptides.height  # n_peptides * n_samples long rows
dataset.intensity_normalized  # 1-D, aligned to peptides rows
```

Example `config.yaml`:

```yaml
control_condition: control
conditions:
  control: [S1, S2]
  treated: [S3, S4]
min_peptides: 4
model: rlm
```

For in-memory tables use `prepare(df, config)` or `prepare(lazy_frame, config)`. Prefer `prepare_from_parquet` when starting from a file: it lazy-scans, projects columns, and filters to configured samples before materialization.

To inspect harmonized long-format rows without normalizing, use `read_peptides(path, config)`.

### Planned discovery API

```python
import proteoforge as pf

config = pf.Config.from_yaml_path("config.yaml")
result = pf.discover(data="peptides.parquet", config=config)

result.summary()
result.mapping
result.dpf_quantities()
result.save("result.pfg")
```

```bash
proteoforge discover peptides.parquet --config config.yaml -o results/
```

## Documentation

Full docs for v0.0.1 (configuration, I/O, prepare, normalization):

- [Documentation home](docs/index.md)
- [Configuration](docs/config.md)
- [Input and output](docs/io.md)
- [Prepare](docs/prepare.md)
- [Normalization](docs/normalization.md)
- [PreparedDataset](docs/prepared-dataset.md)
- [Changelog](CHANGELOG.md)

Local reference comparison against the [manuscript analysis code](https://github.com/LangeLab/ProteoForge_Analysis) lives under `tools/compare/` (not required for package use).

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

Tag `vX.Y.Z` to trigger trusted PyPI publish (`hatch-vcs` versioning).

## References

- ProteoForge manuscript (bioRxiv 2025)
- [PeCorA](https://doi.org/10.1021/acs.jproteome.0c00602), [COPF](https://doi.org/10.1038/s41467-021-24030-x)
- [ProteoForge analysis repository](https://github.com/LangeLab/ProteoForge_Analysis) (manuscript scripts; reference for numerical parity)

## License

MIT License. See [LICENSE](LICENSE) for details.

<p align="center">
    <em>Frost thins the thick stem,</em><br />
    <em>Peptides break their silent bond,</em><br />
    <em>New forms now emerge.</em>
</p>
