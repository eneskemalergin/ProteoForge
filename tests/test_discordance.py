"""End-to-end discordance orchestration tests."""

from __future__ import annotations

import types

import numpy as np
import polars as pl
import pytest

from proteoforge import Config, prepare
from proteoforge._discordance import (
    ADJUSTED_P_VALUE,
    IS_DISCORDANT,
    RAW_P_VALUE,
    WITHIN_P_VALUE,
    _spawn_main_usable,
    run_discordance,
)
from proteoforge._exceptions import (
    ProteoForgeParallelFallbackWarning,
    ProteoForgeValidationError,
)
from proteoforge.schema import FIT_STATUS, PEPTIDE_ID, PROTEIN_ID

CONDITIONS = {"control": ("S1", "S2", "S3"), "treated": ("S4", "S5", "S6")}
SAMPLES = {
    "S1": "control",
    "S2": "control",
    "S3": "control",
    "S4": "treated",
    "S5": "treated",
    "S6": "treated",
}


def _discordant_frame() -> pl.DataFrame:
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for i in range(4):
            base = 10.0 + i
            for sample, condition in SAMPLES.items():
                value = base + rng.normal(scale=0.05)
                if protein == "P1" and i == 0 and condition == "treated":
                    value += 4.0
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": f"PEP{i}",
                        "sample_id": sample,
                        "intensity": value,
                    }
                )
    return pl.DataFrame(rows)


def test_run_discordance_flags_divergent_peptide() -> None:
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        fdr=0.05,
    )
    prepared = prepare(_discordant_frame(), config)
    result = run_discordance(prepared)
    table = result.table
    assert table.columns == [
        PROTEIN_ID,
        PEPTIDE_ID,
        RAW_P_VALUE,
        WITHIN_P_VALUE,
        ADJUSTED_P_VALUE,
        IS_DISCORDANT,
        FIT_STATUS,
    ]
    assert table.height == 8

    target = table.filter((pl.col(PROTEIN_ID) == "P1") & (pl.col(PEPTIDE_ID) == "PEP0"))
    assert bool(target.get_column(IS_DISCORDANT).item(0)) is True

    protein_one = table.filter(pl.col(PROTEIN_ID) == "P1")
    min_adjusted = protein_one.get_column(ADJUSTED_P_VALUE).min()
    assert target.get_column(ADJUSTED_P_VALUE).item(0) == min_adjusted
    assert result.metadata["n_peptides_tested"] == 8
    assert result.n_discordant == int(table.get_column(IS_DISCORDANT).sum())


def test_wls_discordance_end_to_end() -> None:
    rng = np.random.default_rng(31)
    rows: list[dict[str, object]] = []
    for protein in ("P1", "P2"):
        for pep in ("PEP0", "PEP1", "PEP2", "PEP3"):
            for sample in SAMPLES:
                rows.append(
                    {
                        "protein_id": protein,
                        "peptide_id": pep,
                        "sample_id": sample,
                        "intensity": 10.0 + rng.normal(scale=0.02),
                        "is_real": True,
                        "is_complete_missing": False,
                    }
                )
    frame = pl.DataFrame(rows)
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="wls",
        input_is_log2=True,
        fdr=0.05,
    )
    result = run_discordance(prepare(frame, config))
    assert result.metadata["model"] == "wls"
    assert result.table.height == 8
    assert "ok" in result.metadata["skip_reason_counts"]


def test_discordance_surfaces_non_ok_fit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model skip reasons must flow into the table and skip_reason_counts metadata."""
    import proteoforge._discordance as disc
    from proteoforge.models._fit_status import FIT_STATUS_OK, FIT_STATUS_RANK_DEFICIENT

    class _StubModel:
        name = "rlm"
        use_f = False

        def fit_pvalues(
            self,
            design: object,
            response: object,
            weight: object,
            *,
            n_interaction: int,
        ) -> np.ndarray:
            pvalues, _ = self.fit_pvalues_and_status(
                design, response, weight, n_interaction=n_interaction
            )
            return pvalues

        def fit_pvalues_and_status(
            self,
            design: np.ndarray,
            response: object,
            weight: object,
            *,
            n_interaction: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            del response, weight, n_interaction
            m = design.shape[0]
            pvalues = np.full(m, np.nan, dtype=np.float64)
            status = np.full(m, FIT_STATUS_RANK_DEFICIENT, dtype=object)
            pvalues[0] = 0.25
            status[0] = FIT_STATUS_OK
            return pvalues, status

    monkeypatch.setattr(disc, "select_model", lambda _name: _StubModel())

    rng = np.random.default_rng(17)
    rows: list[dict[str, object]] = []
    for pep in ("PEP0", "PEP1"):
        for sample in SAMPLES:
            rows.append(
                {
                    "protein_id": "P1",
                    "peptide_id": pep,
                    "sample_id": sample,
                    "intensity": 10.0 + rng.normal(scale=0.05),
                }
            )
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        min_peptides=2,
    )
    result = run_discordance(prepare(pl.DataFrame(rows), config))
    statuses = set(result.table.get_column(FIT_STATUS).to_list())
    assert FIT_STATUS_RANK_DEFICIENT in statuses
    assert result.metadata["skip_reason_counts"][FIT_STATUS_RANK_DEFICIENT] == 1
    assert result.metadata["n_peptides_skipped"] == 1
    assert np.isfinite(result.table.get_column(RAW_P_VALUE).item(0))
    assert not np.isfinite(result.table.get_column(RAW_P_VALUE).item(1))


def test_is_discordant_respects_fdr_threshold() -> None:
    frame = _discordant_frame()
    strict = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        fdr=1e-12,
    )
    loose = strict.replace(fdr=0.5)
    strict_result = run_discordance(prepare(frame, strict))
    loose_result = run_discordance(prepare(frame, loose))
    assert int(strict_result.table.get_column(IS_DISCORDANT).sum()) <= int(
        loose_result.table.get_column(IS_DISCORDANT).sum()
    )
    assert int(loose_result.table.get_column(IS_DISCORDANT).sum()) >= 1


def test_correction_within_holm_changes_within_p_values() -> None:
    frame = _discordant_frame()
    bonf = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        correction_within="bonferroni",
        fdr=0.05,
    )
    holm = bonf.replace(correction_within="holm")
    bonf_table = run_discordance(prepare(frame, bonf)).table.sort(
        [PROTEIN_ID, PEPTIDE_ID]
    )
    holm_table = run_discordance(prepare(frame, holm)).table.sort(
        [PROTEIN_ID, PEPTIDE_ID]
    )
    assert not np.allclose(
        bonf_table.get_column(WITHIN_P_VALUE).to_numpy(),
        holm_table.get_column(WITHIN_P_VALUE).to_numpy(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        bonf_table.get_column(RAW_P_VALUE).to_numpy(),
        holm_table.get_column(RAW_P_VALUE).to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_resolve_n_jobs_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``n_jobs=-1`` caps workers and avoids oversubscribing BLAS threads."""
    import proteoforge._discordance as disc

    monkeypatch.setattr(disc.os, "cpu_count", lambda: 16)
    assert disc._resolve_n_jobs(-1) == 8
    monkeypatch.setattr(disc.os, "cpu_count", lambda: 4)
    assert disc._resolve_n_jobs(-1) == 2
    assert disc._resolve_n_jobs(0) == 1
    assert disc._resolve_n_jobs(3) == 3


def test_spawn_main_usable_rejects_stdin() -> None:
    import sys

    original = sys.modules["__main__"]
    fake = types.ModuleType("__main__")
    fake.__file__ = "<stdin>"
    sys.modules["__main__"] = fake
    try:
        assert _spawn_main_usable() is False
    finally:
        sys.modules["__main__"] = original


def test_parallel_falls_back_when_spawn_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stdin/notebook entry points must not raise BrokenProcessPool."""
    import proteoforge._discordance as disc

    serial_calls: list[int] = []

    def _fake_serial(
        model: object,
        blocks: list[object],
        offsets: list[int],
        group_items: list[tuple[object, list[int]]],
        *,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        del model, blocks, offsets, group_items, show_progress
        serial_calls.append(1)
        return np.array([0.5, 0.6]), np.array(["ok", "ok"], dtype=object)

    monkeypatch.setattr(disc, "_spawn_main_usable", lambda: False)
    monkeypatch.setattr(disc, "_resolve_n_jobs", lambda n: 4)
    monkeypatch.setattr(
        disc,
        "_shape_group_items",
        lambda blocks: ([0], [((6, 4, 1, False), [0]), ((8, 4, 1, False), [0])]),
    )
    monkeypatch.setattr(disc, "_fit_shape_serial", _fake_serial)

    class _Model:
        name = "rlm"

    with pytest.warns(ProteoForgeParallelFallbackWarning, match="script file"):
        out, _, meta = disc._fit_shape_streaming(_Model(), [object()], n_jobs=4)
    assert serial_calls == [1]
    np.testing.assert_allclose(out, [0.5, 0.6])
    assert meta["parallel_fallback"] is True
    assert meta["n_jobs_effective"] == 1
    assert meta["n_jobs_requested"] == 4
    assert "script file" in str(meta["parallel_fallback_reason"])


def test_parallel_falls_back_on_broken_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed worker pool must recover via serial fitting."""
    from concurrent.futures.process import BrokenProcessPool

    import proteoforge._discordance as disc

    serial_calls: list[int] = []

    def _fake_serial(
        model: object,
        blocks: list[object],
        offsets: list[int],
        group_items: list[tuple[object, list[int]]],
        *,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        del model, blocks, offsets, group_items, show_progress
        serial_calls.append(1)
        return np.array([0.1]), np.array(["ok"], dtype=object)

    def _broken_parallel(
        *args: object, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        del args, kwargs
        raise BrokenProcessPool("synthetic pool failure", None)

    monkeypatch.setattr(disc, "_spawn_main_usable", lambda: True)
    monkeypatch.setattr(disc, "_resolve_n_jobs", lambda n: 2)
    monkeypatch.setattr(
        disc,
        "_shape_group_items",
        lambda blocks: ([0], [((6, 4, 1, False), [0]), ((8, 4, 1, False), [0])]),
    )
    monkeypatch.setattr(disc, "_fit_shape_parallel", _broken_parallel)
    monkeypatch.setattr(disc, "_fit_shape_serial", _fake_serial)

    class _Model:
        name = "rlm"

    class _Block:
        n_peptides = 1

    with pytest.warns(ProteoForgeParallelFallbackWarning, match="worker pool failed"):
        out, _, meta = disc._fit_shape_streaming(_Model(), [_Block()], n_jobs=2)
    assert serial_calls == [1]
    np.testing.assert_allclose(out, [0.1])
    assert meta["parallel_fallback"] is True
    assert meta["parallel_fallback_reason"] is not None


@pytest.mark.parametrize("batching", ["shape", "protein", "scalar"])
def test_show_progress_preserves_results(batching: str) -> None:
    """Progress reporting must not change p-values or flags."""
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
        fdr=0.05,
    )
    prepared = prepare(_discordant_frame(), config)
    baseline = run_discordance(prepared, batching=batching, show_progress=False)
    with_progress = run_discordance(
        prepared,
        batching=batching,
        show_progress=True,
        n_jobs=1,
    )
    assert baseline.table.equals(with_progress.table)
    assert baseline.n_discordant == with_progress.n_discordant


def test_fit_shape_parallel_shuts_down_pool_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupt or worker errors must cancel pool workers instead of leaving them."""
    import proteoforge._discordance as disc

    shutdown_calls: list[tuple[bool, bool]] = []

    class _FakeFuture:
        def result(self) -> list[object]:
            msg = "worker boom"
            raise RuntimeError(msg)

    class _FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def submit(self, fn: object, task: object) -> _FakeFuture:
            del fn, task
            return _FakeFuture()

        def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))

    class _Block:
        n_peptides = 2

    def _fake_as_completed(futures: dict[_FakeFuture, int]) -> object:
        yield from futures

    monkeypatch.setattr(disc, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(disc, "as_completed", _fake_as_completed)

    class _Model:
        name = "rlm"

    with pytest.raises(RuntimeError, match="worker boom"):
        disc._fit_shape_parallel(
            _Model(),
            [_Block(), _Block()],
            [0, 2],
            [((6, 4, 1, False), [0, 1])],
            workers=2,
            show_progress=False,
        )
    assert shutdown_calls == [(False, True)]


def test_unknown_batching_rejected() -> None:
    config = Config(
        control_condition="control",
        conditions=CONDITIONS,
        model="rlm",
        input_is_log2=True,
    )
    prepared = prepare(_discordant_frame(), config)
    with pytest.raises(ProteoForgeValidationError, match="batching"):
        run_discordance(prepared, batching="turbo")
