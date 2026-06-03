<!-- markdownlint-disable MD033 MD036 MD041 MD045 -->
<p align="center">
    <strong>ProteoForge</strong>
</p>

<p align="center">
    <strong>Differential proteoform discovery for bottom-up proteomics</strong>
</p>

<p align="center">
    <a href="https://github.com/eneskemalergin/ProteoForge"><img src="https://img.shields.io/badge/status-in%20development-0f766e?style=for-the-badge" alt="In development" /></a>
    <a href="#references"><img src="https://img.shields.io/badge/paper-bioRxiv%202025-7c3aed?style=for-the-badge" alt="Paper" /></a>
    <a href="https://github.com/eneskemalergin/ProteoForge"><img src="https://img.shields.io/badge/docs-readme-0891b2?style=for-the-badge" alt="Documentation" /></a>
    <a href="https://github.com/eneskemalergin/ProteoForge/actions"><img src="https://img.shields.io/github/actions/workflow/status/eneskemalergin/ProteoForge/ci.yml?branch=main&style=for-the-badge&logo=github&label=CI" alt="CI" /></a>
</p>
<p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12--3.15-f59e0b?style=for-the-badge" alt="Python 3.12-3.15" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/numpy-2.2%2B-2563eb?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy 2.2+" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/polars-1.26%2B-cd7c2f?style=for-the-badge&logo=polars&logoColor=white" alt="Polars 1.26+" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/scipy-1.15%2B-8b5cf6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy 1.15+" /></a>
</p>

ProteoForge discovers differential proteoforms from an **imputed peptide matrix** and a **condition design with a control**. Peptides that break rank with their siblings are grouped into dPF units: canonical signal (`dPF_0`), multi-peptide proteoforms (`dPF_1+`), and singleton discordants (`dPF_-1`).

**v0.1.0 is in development.** Scaffolding is done; the discovery workflow, CLI, and PyPI release are next.

Does not impute, search, or quantify. Upstream imputation is required.

## Features

Target for v0.1.0:

- Four-module pipeline: normalize → discordance (`Intensity ~ Condition * Peptide`) → cluster → dPF assign
- RLM default; WLS (imputation-aware weights) and empirical-Bayes backends
- Dynamic hierarchical clustering (`hybrid_outlier_cut`)
- Peptide-to-dPF mapping and collapsed dPF quantity matrix (downstream handoff to [QuEStVar](https://github.com/eneskemalergin/QuEStVar))
- FASTA, peptide positions, UniProt / iPTMnet overlays; HTML report and core plots
- Python API, YAML config, and Typer CLI

## Installation

Python 3.12 to 3.15. Runtime: NumPy 2.2+, Polars 1.26+, SciPy 1.15+.

PyPI follows v0.1.0. Until then, install from source:

```bash
git clone https://github.com/eneskemalergin/ProteoForge.git
cd ProteoForge
uv sync
```

Contributors: [uv](https://docs.astral.sh/uv/) is recommended. Optional extras: `plots`, `interactive`, `accel`, `cli`, `docs`.

```bash
pip install -e ".[cli,plots]"
```

## Quick start

Planned v0.1.0 interface (not implemented yet):

```python
import proteoforge as pf

result = pf.discover(
    data="peptides.parquet",
    design="design.csv",
    config=pf.Config(model="rlm", fdr=0.001, min_peptides=4, n_jobs=-1),
)

result.summary()
result.mapping
result.dpf_quantities()
result.save("result.pfg")
```

```bash
proteoforge discover peptides.parquet --design design.csv -o results/
```

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

Tag `vX.Y.Z` to trigger trusted PyPI publish (`hatch-vcs` versioning).

## Roadmap

Path to v0.1.0:

- Input: `Config`, Polars I/O, validation, control-relative normalization
- Engine: batched NumPy discordance models, two-step FDR, WLS masks
- Grouping: hierarchical cluster, dPF rules
- Surface: `discover()`, `ProteoformResults`, CLI, sequence/annotation, report
- Release: golden tests, docs site, PyPI

## References

- ProteoForge manuscript (bioRxiv 2025)
- [PeCorA](https://doi.org/10.1021/acs.jproteome.0c00602), [COPF](https://doi.org/10.1038/s41467-021-24030-x)

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
    <em>Frost thins the thick stem,</em><br />
    <em>Peptides break their silent bond,</em><br />
    <em>New forms now emerge.</em>
</p>
