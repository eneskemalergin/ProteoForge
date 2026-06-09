"""ProteoForge public API."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from proteoforge._config import Config
from proteoforge._exceptions import ProteoForgeIOError, ProteoForgeValidationError
from proteoforge.cluster import (
    ClusterResult,
    ProteoformMappingResult,
    assign_proteoforms,
    run_cluster,
)
from proteoforge.discordance import DiscordanceResult, run_discordance
from proteoforge.fixture import FixtureBundle, load_fixture_bundle
from proteoforge.prepare import prepare, prepare_from_parquet, validate_and_prepare
from proteoforge.types import ColumnMap, DesignTable, PreparedDataset

try:
    __version__ = version("proteoforge")
except PackageNotFoundError:
    __version__ = "0.0.3"

__all__ = [
    "ClusterResult",
    "ColumnMap",
    "Config",
    "DesignTable",
    "DiscordanceResult",
    "FixtureBundle",
    "PreparedDataset",
    "ProteoForgeIOError",
    "ProteoForgeValidationError",
    "ProteoformMappingResult",
    "__version__",
    "assign_proteoforms",
    "load_fixture_bundle",
    "prepare",
    "prepare_from_parquet",
    "run_cluster",
    "run_discordance",
    "validate_and_prepare",
]
