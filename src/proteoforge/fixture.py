"""Fixture bundle loading for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from proteoforge._config import Config
from proteoforge._exceptions import ProteoForgeValidationError
from proteoforge.io import read_peptides
from proteoforge.prepare import prepare_from_parquet

if TYPE_CHECKING:
    from proteoforge.types import PreparedDataset


@dataclass(frozen=True)
class FixtureBundle:
    """
    Resolved paths for a versioned fixture directory.

    Experimental design lives in ``config.yaml`` (``control_condition`` and
    ``conditions``). The manifest only indexes data files.

    Parameters
    ----------
    root
        Directory containing ``manifest.yaml``.
    name
        Fixture identifier (e.g. ``complete``).
    peptides_path
        ProteoForge input parquet/csv (pipeline columns only).
    config_path
        ``Config`` YAML including embedded design.
    benchmark_path
        Optional parquet with evaluation columns (not passed to ``prepare()``).
    description
        Human-readable summary from the manifest.
    """

    root: Path
    name: str
    peptides_path: Path
    config_path: Path
    benchmark_path: Path | None = None
    description: str = ""

    def load_config(self) -> Config:
        """Load pipeline configuration including experimental design."""
        return Config.from_yaml_path(self.config_path)

    def load_peptides(self, config: Config | None = None) -> pl.DataFrame:
        """Load harmonized peptide input table."""
        cfg = config or self.load_config()
        return read_peptides(self.peptides_path, cfg)

    def load_benchmark_table(self) -> pl.DataFrame:
        """
        Load the benchmark table including ground-truth columns.

        Raises
        ------
        ProteoForgeValidationError
            If this fixture has no benchmark file.
        """
        if self.benchmark_path is None:
            msg = f"Fixture '{self.name}' has no benchmark table."
            raise ProteoForgeValidationError(msg)
        return pl.read_parquet(self.benchmark_path)

    def prepare(self, config: Config | None = None) -> PreparedDataset:
        """Validate and normalize this fixture's peptide table."""
        cfg = config or self.load_config()
        return prepare_from_parquet(self.peptides_path, cfg)


def load_fixture_bundle(manifest_path: str | Path) -> FixtureBundle:
    """
    Load a fixture bundle from ``manifest.yaml``.

    Manifest layout::

        name: complete
        description: ...
        files:
          peptides: complete-real.parquet
          benchmark: eval-labels.parquet  # optional ground-truth columns
          config: config.yaml

    Parameters
    ----------
    manifest_path
        Path to ``manifest.yaml`` or its parent directory.

    Returns
    -------
    FixtureBundle
        Resolved fixture paths.

    Raises
    ------
    ProteoForgeValidationError
        If the manifest or referenced files are missing or invalid.
    """
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.yaml"
    if not path.is_file():
        msg = f"Fixture manifest not found: {path}"
        raise ProteoForgeValidationError(msg)

    root = path.parent
    data = _load_yaml(path)
    if not isinstance(data, dict):
        msg = "Fixture manifest root must be a mapping."
        raise ProteoForgeValidationError(msg)

    name = str(data.get("name", root.name))
    description = str(data.get("description", "")).strip()
    files = data.get("files")
    if not isinstance(files, dict):
        msg = "Fixture manifest must include a 'files' mapping."
        raise ProteoForgeValidationError(msg)

    required_keys = ("peptides", "config")
    missing_keys = [key for key in required_keys if key not in files]
    if missing_keys:
        msg = f"Fixture manifest missing file keys: {missing_keys}."
        raise ProteoForgeValidationError(msg)

    peptides_path = _resolve_file(root, str(files["peptides"]), label="peptides")
    config_path = _resolve_file(root, str(files["config"]), label="config")

    benchmark_path: Path | None = None
    if "benchmark" in files and files["benchmark"] is not None:
        benchmark_path = _resolve_file(root, str(files["benchmark"]), label="benchmark")

    return FixtureBundle(
        root=root,
        name=name,
        peptides_path=peptides_path,
        config_path=config_path,
        benchmark_path=benchmark_path,
        description=description,
    )


def _resolve_file(root: Path, relative: str, *, label: str) -> Path:
    resolved = (root / relative).resolve()
    if not resolved.is_file():
        msg = (
            f"Fixture {label} file not found: {resolved}. "
            f"Check paths in {root / 'manifest.yaml'}."
        )
        raise ProteoForgeValidationError(msg)
    return resolved


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        msg = "PyYAML is required to load fixture manifests."
        raise ProteoForgeValidationError(msg) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))
