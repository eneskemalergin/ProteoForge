"""ProteoForge: proteomics analysis toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("proteoforge")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
