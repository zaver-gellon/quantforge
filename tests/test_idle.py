"""Idle diversifier. No market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_idle_diversifier, run_trend_core
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.idle import (
    ASSETS,
    BANDS,
    OFF_NAME,
    WINDOWS,
    cells,
    idle_signal_kwargs,
    idle_state,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _flat_book(n=12):
    idx = pd.bdate_range("2020-01-02", periods=n)
    flat = pd.Series(100.0, index=idx)
    return idx, flat


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert len(found) == 12
    assert len({variant_name(*cell) for cell in found}) == 12
    assert ASSETS == ("GLD", "IEF", "TLT")
    assert WINDOWS == (100, 200)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "idle_off"
    names = {variant_name(*cell) for cell in found}
    for cell in found:
        neighbors = adjacent_cells(cell, [ASSETS, WINDOWS, BANDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found
            assert variant_name(*neighbor) in names


def test_state_does_not_read_a_future_close_and_waits_for_the_window():
    idx = pd.bdate_range("2014-01-02", periods=80)
    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)
    state, ready = idle_state(close, 20, 0.0, idx)
    assert not bool(ready.iloc[:19].any())
    assert not bool(state.iloc[:19].any())
    future = close.copy()
    future.iloc[40:] = close.iloc[40] * 0.5
    again, again_ready = idle_state(future, 20, 0.0, idx)
    pd.testing.assert_series_equal(state.iloc[:40], again.iloc[:40])
    pd.testing.assert_series_equal(ready.iloc[:40], again_ready.iloc[:40])


def test_unready_diversifier_matches_the_trend_core_including_dips():
    idx, flat = _flat_book(40)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    trend = pd.Series(True, index=idx)
    trend.iloc[15:25] = False
    dip = pd.Series(False, index=idx)
    dip.iloc[8:12] = True
    mix = pd.Series(False, index=idx)
    mix.iloc[20] = True
    rf = pd.Series(0.0001, index=idx)
    ready = pd.Series(False, index=idx)
    div_on = pd.Series(True, index=idx)
    # A rally in the diversifier must not leak in before its filter exists.
    div_close = close * 3
    kwargs = dict(spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005)
    plain = run_trend_core(
        open_, close, open_ * 1.1, close * 1.1, open_ * 1.2, close * 1.2,
        trend, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    idle = run_idle_diversifier(
        open_, close, open_ * 1.1, close * 1.1, open_ * 1.2, close * 1.2,
        div_close, div_close, trend, div_on, ready, dip, mix, 0.5, 0.5, rf,
        div_cost=0.0005, **kwargs,
    )
    pd.testing.assert_series_equal(plain.returns, idle.returns, check_names=False)


def test_equity_trend_on_ignores_the_diversifier():
    idx, _flat = _flat_book(8)
    close = pd.Series(100.0, index=idx)
    div = close.copy()
    div.iloc[4:] = 200.0
    trend = pd.Series(True, index=idx)
    off = pd.Series(False, index=idx)
    ready = pd.Series(True, index=idx)
    div_on = pd.Series(True, index=idx)
    rf = pd.Series(0.0, index=idx)
    plain = run_trend_core(
        close, close, close, close, close, close,
        trend, off, off, 0.5, 0.5, rf,
    )
    idle = run_idle_diversifier(
        close, close, close, close, close, close, div, div,
        trend, div_on, ready, off, off, 0.5, 0.5, rf,
    )
    pd.testing.assert_series_equal(plain.returns, idle.returns, check_names=False)


def test_off_trend_book_earns_the_next_open_to_close_and_not_the_gap():
    idx, flat = _flat_book(8)
    trend = pd.Series(True, index=idx)
    trend.iloc[3:] = False
    ready = pd.Series(True, index=idx)
    div_on = pd.Series(True, index=idx)
    off = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    # Gap up into the fill. The book buys the open, so the gap is not its return.
    div_open = flat.copy()
    div_close = flat.copy()
    div_open.iloc[4] = 120.0
    div_close.iloc[4] = 120.0
    gapped = run_idle_diversifier(
        flat, flat, flat, flat, flat, flat, div_open, div_close,
        trend, div_on, ready, off, off, 0.5, 0.5, rf,
    )
    assert gapped.returns.iloc[4] == pytest.approx(0.0)
    # Same decision, but the diversifier rallies 10% from that open to the close.
    div_close.iloc[4] = 110.0
    div_open.iloc[4] = 100.0
    div_open.iloc[5] = 110.0
    div_close.iloc[5] = 110.0
    earned = run_idle_diversifier(
        flat, flat, flat, flat, flat, flat, div_open, div_close,
        trend, div_on, ready, off, off, 0.5, 0.5, rf,
    )
    assert earned.returns.iloc[3] == pytest.approx(0.0)
    assert earned.returns.iloc[4] == pytest.approx(0.10)
    assert earned.returns.iloc[5] == pytest.approx(0.0)
    later = div_close.copy()
    later.iloc[5:] = 50.0
    unchanged = run_idle_diversifier(
        flat, flat, flat, flat, flat, flat, div_open, later,
        trend, div_on, ready, off, off, 0.5, 0.5, rf,
    )
    assert unchanged.returns.iloc[4] == pytest.approx(earned.returns.iloc[4])


def test_signal_leaves_equities_only_when_the_filter_is_ready_and_on():
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    ready = pd.Series(True, index=idx)
    idle = pd.Series(False, index=idx)
    idle.iloc[-1] = True
    spec = TrendCoreSpec(window=20)
    held = build_trend_signal(
        spec, close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, idle_on=idle, idle_ready=ready, idle_symbol="IEF",
        idle_info={"idle_rule": "idle_ief_w100_b0"},
    )
    # Rising SPY keeps the equity trend on, so the diversifier is not the book.
    assert held["weights"]["SPY"] == pytest.approx(0.5)
    assert held["weights"]["UPRO"] == pytest.approx(0.5)
    assert "IEF" not in held["weights"]

    trend_off = close.copy()
    trend_off.iloc[-30:] = 70.0
    parked = build_trend_signal(
        spec, trend_off, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, idle_on=idle, idle_ready=ready, idle_symbol="IEF",
    )
    assert parked["weights"] == {"IEF": 1.0}
    assert parked["trade_reason"] == "idle_diversifier"

    not_ready = ready.copy()
    not_ready.iloc[-1] = False
    unchanged = build_trend_signal(
        spec, trend_off, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, idle_on=idle, idle_ready=not_ready, idle_symbol="IEF",
    )
    assert unchanged["weights"]["SPY"] == pytest.approx(0.5)
    assert unchanged["weights"]["BIL"] == pytest.approx(0.5)
    assert "IEF" not in unchanged["weights"]

    future = idle.copy()
    future.iloc[-1] = False
    future.iloc[10] = True
    earlier = build_trend_signal(
        spec, close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, idle_on=future, idle_ready=ready, idle_symbol="GLD",
    )
    assert "GLD" not in earlier["weights"]


def test_rejected_iteration_does_not_reload(tmp_path):
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "idle_diversifier",
        "walk_forward_picks": [["2018-01-02", "idle_ief_w200_b2"]],
        "variants": [{"name": "idle_ief_w200_b2", "symbol": "IEF", "window": 200, "band": 0.02}],
    }]))
    idx = pd.bdate_range("2018-01-02", periods=30)
    close = pd.Series(np.linspace(80, 100, len(idx)), index=idx)
    assert idle_signal_kwargs(path, close, idx[-1], {"IEF": close}) == {}

    kept = json.loads(path.read_text())
    kept[0]["kept"] = True
    path.write_text(json.dumps(kept))
    kwargs = idle_signal_kwargs(path, close, idx[-1], {"IEF": close})
    assert kwargs["idle_symbol"] == "IEF"
    assert kwargs["idle_info"]["idle_window"] == 200
    with pytest.raises(Exception):
        idle_signal_kwargs(path, close, idx[-1], {})
