"""N-day confirmation of the trend band. No market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.confirm import (
    BANDS,
    CONFIRMS,
    LIVE_BAND,
    LIVE_CONFIRM,
    WINDOW,
    cells,
    confirmed_trend_state,
    is_live_cell,
    kept_confirm_trend,
    matches_live_filter,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert len(found) == 20
    assert len({variant_name(*cell) for cell in found}) == 20
    assert BANDS == (0.005, 0.01, 0.02, 0.03)
    assert CONFIRMS == (1, 2, 3, 4, 5)
    assert WINDOW == 200
    assert sum(not is_live_cell(*cell) for cell in found) == 19
    assert is_live_cell(LIVE_BAND, LIVE_CONFIRM)
    assert variant_name(0.005, 1) == "confirm_b0050_n1"
    assert variant_name(0.02, 1) == "confirm_b0200_n1"
    assert variant_name(0.03, 5) == "confirm_b0300_n5"
    names = {variant_name(*cell) for cell in found}
    for cell in found:
        neighbors = adjacent_cells(cell, [BANDS, CONFIRMS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found
            assert variant_name(*neighbor) in names


def test_one_day_confirmation_is_the_live_hysteresis_rule():
    idx = pd.bdate_range("2004-01-02", periods=500)
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)
    assert matches_live_filter(close)
    for band in (0.0, 0.005, 0.01, 0.03):
        confirmed = confirmed_trend_state(close, 200, band, 1)
        live = trend_state(close, 200, band, "daily")
        pd.testing.assert_series_equal(confirmed, live.rename("trend_on"))


def test_confirmation_ignores_a_one_day_pierce_and_waits_for_n_closes():
    idx = pd.bdate_range("2012-01-03", periods=40)
    prices = np.full(len(idx), 100.0)
    close = pd.Series(prices, index=idx)
    # SMA of 20 is 100. 103 is through a 2% band. One day is not three days.
    close.iloc[25] = 103.0
    slow = confirmed_trend_state(close, window=20, band=0.02, confirm=3)
    fast = confirmed_trend_state(close, window=20, band=0.02, confirm=1)
    assert bool(fast.iloc[25]) is True
    assert bool(slow.iloc[25]) is False
    assert bool(slow.iloc[-1]) is False

    held = close.copy()
    held.iloc[25:28] = 103.0
    waited = confirmed_trend_state(held, window=20, band=0.02, confirm=3)
    assert bool(waited.iloc[25]) is False
    assert bool(waited.iloc[26]) is False
    assert bool(waited.iloc[27]) is True


def test_a_close_back_inside_the_band_resets_the_count():
    idx = pd.bdate_range("2013-01-02", periods=40)
    prices = np.full(len(idx), 100.0)
    prices[25] = 103.0
    prices[26] = 103.0
    prices[27] = 100.5  # inside the 2% band
    prices[28] = 103.0
    prices[29] = 103.0
    close = pd.Series(prices, index=idx)
    state = confirmed_trend_state(close, window=20, band=0.02, confirm=3)
    assert not bool(state.iloc[29])
    prices[30] = 103.0
    finished = confirmed_trend_state(pd.Series(prices, index=idx), window=20, band=0.02, confirm=3)
    assert bool(finished.iloc[30]) is True


def test_state_does_not_read_a_future_close():
    idx = pd.bdate_range("2011-01-03", periods=80)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(idx)))), index=idx)
    base = confirmed_trend_state(close, 20, 0.01, 4)
    future = close.copy()
    future.iloc[50:] = close.iloc[50] * 0.5
    again = confirmed_trend_state(future, 20, 0.01, 4)
    pd.testing.assert_series_equal(base.iloc[:50], again.iloc[:50])


def test_incomplete_average_stays_off():
    idx = pd.bdate_range("2016-01-04", periods=30)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    state = confirmed_trend_state(close, window=20, band=0.0, confirm=1)
    assert not bool(state.iloc[:19].any())


def test_signal_uses_a_kept_confirmation_and_ignores_a_rejection(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=80)
    level = np.full(len(idx), 100.0)
    level[30:] = 130.0
    close = pd.Series(level, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "trend_confirm",
        "walk_forward_picks": [["2018-01-02", "confirm_b0100_n3"]],
        "variants": [{"name": "confirm_b0100_n3", "band": 0.01, "confirm": 3}],
    }]))
    assert kept_confirm_trend(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "trend_confirm",
        "walk_forward_picks": [["2018-01-02", "confirm_b0200_n1"]],
        "variants": [{"name": "confirm_b0200_n1", "band": 0.02, "confirm": 1}],
    }]))
    assert kept_confirm_trend(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "trend_confirm",
        "walk_forward_picks": [["2018-01-02", "confirm_b0100_n3"]],
        "variants": [{"name": "confirm_b0100_n3", "band": 0.01, "confirm": 3}],
    }]))
    state, info = kept_confirm_trend(path, close, as_of=idx[-1])
    assert info["confirm_rule"] == "confirm_b0100_n3"
    assert info["confirm_days"] == 3
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=state, confirm_info=info,
    )
    assert payload["diagnostics"]["trend_on"] is bool(state.iloc[-1])
    assert payload["diagnostics"]["band"] == pytest.approx(0.01)
    assert payload["diagnostics"]["confirm_days"] == 3

    future = close.copy()
    future.iloc[-5:] = 1.0
    again, _info = kept_confirm_trend(path, future, as_of=idx[-10])
    pd.testing.assert_series_equal(state.loc[: idx[-10]], again)
