"""CodSpeed smoke benchmarks for Phase 1 prepare."""

from __future__ import annotations

import pytest

from conftest import FIXTURE_AVAILABLE, FIXTURE_ROOT
from proteoforge import Config, load_fixture_bundle

pytestmark = pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="local complete fixture not present under benchmarks/fixtures/",
)


@pytest.mark.benchmark
def test_prepare_complete_subset(benchmark: object) -> None:
    bundle = load_fixture_bundle(FIXTURE_ROOT)
    subset = Config.from_yaml_path(bundle.root / "config-subset-day1-day3.yaml")

    def run() -> int:
        dataset = bundle.prepare(subset)
        return dataset.n_peptides

    result = benchmark(run)
    assert result == 22_247
