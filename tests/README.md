# Tests

Fast, committed, CI-safe checks for the installable package.

## Data policy

- **`tests/fixtures/`**: small synthetic inputs (minimal parquet, YAML, CSV). In CI.
- **`tests/conftest.py`**: inline synthetic frames (few rows). In CI.

**Rule:** If a test needs data that is not in `tests/fixtures/` or a few lines of synthetic Polars, it does not belong here.

## What belongs here

- API contracts (`prepare`, `run_discordance`, config validation)
- Numerical checks on **tiny** designs (hand-computed or statsmodels on ≤10 peptides)
- Regression tests using committed minimal fixtures (`test_golden_normalize.py`)

## Commands

CI and default local run (same as `.github/workflows/ci.yml`):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest tests --cov=proteoforge --cov-report=term-missing
```

`pyproject.toml` sets `testpaths = ["tests"]`, so `uv run pytest` only collects this tree.
