"""CodSpeed smoke benchmarks."""

from __future__ import annotations

import pytest


@pytest.mark.benchmark
def test_sum_range(benchmark: object) -> None:
    def run() -> int:
        return sum(range(10_000))

    result = benchmark(run)
    assert result == 49_995_000
