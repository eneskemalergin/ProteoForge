"""Smoke tests for fixture bundle loading."""

from __future__ import annotations

import pytest

from proteoforge import load_fixture_bundle
from proteoforge._exceptions import ProteoForgeValidationError


def test_load_fixture_bundle_from_manifest(fixtures_dir, tmp_path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "name: minimal",
                "description: Unit-test bundle",
                "files:",
                "  peptides: minimal_long.parquet",
                "  config: minimal_config.yaml",
            ]
        ),
        encoding="utf-8",
    )
    for name in ("minimal_long.parquet", "minimal_config.yaml"):
        (tmp_path / name).symlink_to(fixtures_dir / name)

    bundle = load_fixture_bundle(manifest)
    assert bundle.name == "minimal"
    assert bundle.peptides_path.is_file()
    config = bundle.load_config()
    prepared = bundle.prepare(config)
    assert prepared.n_peptides >= config.min_peptides


def test_load_fixture_bundle_rejects_missing_manifest(tmp_path) -> None:
    with pytest.raises(ProteoForgeValidationError, match="manifest not found"):
        load_fixture_bundle(tmp_path / "missing.yaml")
