"""Fixture bundle loading for examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from proteoforge._config import Config
from proteoforge._exceptions import ProteoForgeIOError, ProteoForgeValidationError
from proteoforge.io import read_peptides
from proteoforge.prepare import prepare_from_parquet

if TYPE_CHECKING:
    import polars as pl

    from proteoforge.types import PreparedDataset


@dataclass(frozen=True)
class FixtureBundle:
    """
    Resolved paths for a versioned fixture directory.

    Attributes
    ----------
    root
        Directory containing ``manifest.yaml``.
    name
        Fixture identifier (e.g. ``complete``).
    peptides_path
        Input parquet or csv with pipeline columns only.
    config_path
        ``Config`` YAML including embedded design.
    description
        Human-readable summary from the manifest.
    """

    root: Path
    name: str
    peptides_path: Path
    config_path: Path
    description: str = ""

    def load_config(self) -> Config:
        """Load pipeline configuration including experimental design."""
        return Config.from_yaml_path(self.config_path)

    def load_peptides(self, config: Config | None = None) -> pl.DataFrame:
        """Load harmonized peptide input table."""
        cfg = config or self.load_config()
        return read_peptides(self.peptides_path, cfg)

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
    ProteoForgeIOError
        If a manifest YAML file cannot be read.
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

    return FixtureBundle(
        root=root,
        name=name,
        peptides_path=peptides_path,
        config_path=config_path,
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


def _load_yaml(path: Path) -> object:
    try:
        import yaml
    except ImportError as exc:
        msg = "PyYAML is required to load fixture manifests."
        raise ProteoForgeValidationError(msg) from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"Fixture manifest not found: {path}"
        raise ProteoForgeIOError(msg) from exc
    except OSError as exc:
        msg = f"Could not read fixture manifest: {path}"
        raise ProteoForgeIOError(msg) from exc
