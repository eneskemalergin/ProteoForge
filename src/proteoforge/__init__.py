"""ProteoForge: differential proteoform discovery for bottom-up proteomics."""

from importlib.metadata import PackageNotFoundError, version

from proteoforge._config import Config
from proteoforge.fixture import FixtureBundle, load_fixture_bundle
from proteoforge.prepare import prepare, prepare_from_parquet, validate_and_prepare
from proteoforge.types import ColumnMap, DesignTable, PreparedDataset

try:
    __version__ = version("proteoforge")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ColumnMap",
    "Config",
    "DesignTable",
    "FixtureBundle",
    "PreparedDataset",
    "__version__",
    "load_fixture_bundle",
    "prepare",
    "prepare_from_parquet",
    "validate_and_prepare",
]
