# ProteoForge

Proteomics analysis toolkit.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv add proteoforge
```

Optional extras: `plots`, `interactive`, `accel`, `docs`.

```bash
uv add "proteoforge[plots,interactive]"
```

## Development

```bash
git clone https://github.com/eneskemalergin/ProteoForge.git
cd ProteoForge
uv sync
uv run pre-commit install
```

Common commands:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run nox
uv run mkdocs serve   # requires: uv sync --extra docs
```

## Versioning

Versions are derived from git tags via `hatch-vcs`. Tag releases as `vX.Y.Z` and push to trigger the PyPI publish workflow.

## License

MIT. See [LICENSE](LICENSE).
