"""Command-line entry point for ProteoForge."""

from __future__ import annotations

import argparse
import sys

from proteoforge import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser for ``proteoforge``."""
    parser = argparse.ArgumentParser(prog="proteoforge")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run the ProteoForge CLI.

    Parameters
    ----------
    argv
        Optional argument list. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    parser.parse_args(argv if argv is not None else sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
