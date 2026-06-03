#!/usr/bin/env python3
"""ProteoForge prepare pipeline for comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteoforge import Config, prepare_from_parquet  # noqa: E402
from tools.compare._shared import OUTPUT_NAME, write_normalized_parquet  # noqa: E402


def run(input_path: Path, config_path: Path, output_path: Path) -> None:
    config = Config.from_yaml_path(config_path)
    dataset = prepare_from_parquet(input_path, config)
    write_normalized_parquet(dataset.peptides, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"Normalized long parquet (default name: {OUTPUT_NAME}).",
    )
    args = parser.parse_args()
    run(args.input, args.config, args.output)


if __name__ == "__main__":
    main()