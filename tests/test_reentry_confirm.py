"""Re-entry delay on the UPRO sleeve. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.confirm import confirmed_trend_state
from quantforge.strategies.reconfirm import (
    AXIS,
    BAND,
    CONFIRMS,
    LIVE_CONFIRM,
    OFF_NAME,
    WINDOW,
    cells,
    kept_reentry_confirm,
    matches_live_filter,
    reentry_trend_state,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def test_grid_is_fixed_and_the_live_book_sorts_first():
    found = cells()
    assert found == [2, 3, 5, 8, 10]
    assert CONFIRMS == (2, 3, 5, 8, 10)
    assert AXIS == (1, 2, 3, 5, 8, 10)
    assert LIVE_CONFIRM == 1
    assert WINDOW == 200
    assert BAND == 0.02
    assert OFF_NAME == "re_live"
    names = [variant_name(count) for count in found]
    assert names == ["zre_n02", "zre_n03", "zre_n05", "zre_n08", "zre_n10"]
    assert len(set(names)) == 5
    assert OFF_NAME < min(names)
    assert variant_name(1) == "zre_n01"
    with pytest.raises(ValueError):
        variant_name(4)
    for count in AXIS:
        neighbors = adjacent_cells((count,), [AXIS])
        assert neighbors
        assert (count,) not in neighbors
        for neighbor in neighbors:
            assert neighbor[0] in AXIS
    assert adjacent_cells((2,), [AXIS]) == [(1,), (3,)]
    assert adjacent_cells((10,), [AXIS]) == [(8,)]


def test_one_close_is_the_live_filter():
    idx = pd.bdate_range("2004-01-02", periods=500)
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)
    assert matches_live_filter(close)
    for band in (0.0, 0.005, 0.01, 0.03):
        delayed = reentry_trend_state(close, 200, band, 1)
        live = trend_state(close, 200, band, "daily")
        pd.testing.assert_series_equal(delayed, live.rename("trend_on"))


def test_exit_is_immediate_and_reentry_waits_for_n_closes():
    idx = pd.bdate_range("2012-01-03", periods=40)
    prices = np.full(len(idx), 100.0)
    prices[20:25] = 103.0
    prices[30] = 90.0
    prices[31:33] = 103.0
    close = pd.Series(prices, index=idx)
    slow = reentry_trend_state(close, window=20, band=0.02, confirms=5)
    fast = reentry_trend_state(close, window=20, band=0.02, confirms=1)
    both_ways = confirmed_trend_state(close, window=20, band=0.02, confirm=5)
    assert list(slow.iloc[20:25].astype(bool)) == [False, False, False, False, True]
    assert bool(fast.iloc[20]) is True
    assert bool(slow.iloc[29]) is True
    assert bool(slow.iloc[30]) is False
    assert bool(fast.iloc[30]) is False
    assert bool(both_ways.iloc[30]) is True
    assert bool(fast.iloc[31]) is True
    assert bool(slow.iloc[32]) is False


def test_a_close_back_inside_the_band_resets_the_entry_count():
    idx = pd.bdate_range("2013-01-02", periods=40)
    prices = np.full(len(idx), 100.0)
    prices[20:23] = 103.0
    prices[23] = 90.0
    prices[24] = 103.0
    prices[25] = 103.0
    prices[26] = 100.5
    prices[27] = 103.0
    prices[28] = 103.0
    close = pd.Series(prices, index=idx)
    state = reentry_trend_state(close, window=20, band=0.02, confirms=3)
    assert bool(state.iloc[22]) is True
    assert bool(state.iloc[23]) is False
    assert bool(state.iloc[28]) is False
    prices[29] = 103.0
    finished = reentry_trend_state(pd.Series(prices, index=idx), window=20, band=0.02, confirms=3)
    assert bool(finished.iloc[29]) is True


def test_an_inside_close_does_not_turn_an_open_sleeve_off():
    idx = pd.bdate_range("2014-01-02", periods=36)
    prices = np.full(len(idx), 100.0)
    prices[20:28] = 120.0
    prices[28] = 108.0
    prices[29] = 90.0
    close = pd.Series(prices, index=idx)
    state = reentry_trend_state(close, window=20, band=0.02, confirms=8)
    assert bool(state.iloc[26]) is False
    assert bool(state.iloc[27]) is True
    assert bool(state.iloc[28]) is True
    assert bool(state.iloc[29]) is False


def test_state_does_not_read_a_future_close():
    idx = pd.bdate_range("2011-01-03", periods=80)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(idx)))), index=idx)
    base = reentry_trend_state(close, 20, 0.01, 4)
    future = close.copy()
    future.iloc[50:] = close.iloc[50] * 0.5
    again = reentry_trend_state(future, 20, 0.01, 4)
    pd.testing.assert_series_equal(base.iloc[:50], again.iloc[:50])


def test_incomplete_average_stays_off():
    idx = pd.bdate_range("2016-01-04", periods=30)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    state = reentry_trend_state(close, window=20, band=0.0, confirms=3)
    assert not bool(state.iloc[:19].any())


def test_signal_uses_a_kept_delay_and_ignores_a_rejection(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=260)
    level = np.full(len(idx), 100.0)
    level[220:] = 130.0
    close = pd.Series(level, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "reentry_confirm",
        "walk_forward_picks": [["2018-01-02", "zre_n03"]],
        "variants": [{"name": "zre_n03", "confirms": 3}],
    }]))
    assert kept_reentry_confirm(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "reentry_confirm",
        "walk_forward_picks": [["2018-01-02", "re_live"]],
        "variants": [{"name": "re_live", "confirms": 1}],
    }]))
    assert kept_reentry_confirm(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "reentry_confirm",
        "walk_forward_picks": [["2018-01-02", "zre_n03"]],
        "variants": [{"name": "zre_n03", "confirms": 3}],
    }]))
    state, info = kept_reentry_confirm(path, close, as_of=idx[-1])
    assert info["reentry_rule"] == "zre_n03"
    assert info["reentry_confirms"] == 3
    assert info["reentry_on"] is True
    payload = build_trend_signal(
        TrendCoreSpec(), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=state, reentry_info=info,
    )
    assert payload["diagnostics"]["trend_on"] is True
    assert payload["diagnostics"]["reentry_confirms"] == 3
    assert payload["diagnostics"]["band"] == pytest.approx(0.02)
    live = trend_state(close, 200, 0.02, "daily")
    assert bool(live.iloc[220]) is True
    assert bool(state.iloc[220]) is False
    assert bool(state.iloc[222]) is True

    leaked = close.copy()
    leaked.iloc[-5:] = 40.0
    earlier, early_info = kept_reentry_confirm(path, leaked, as_of=idx[-10])
    pd.testing.assert_series_equal(state.loc[: idx[-10]], earlier)
    assert early_info["reentry_on"] is bool(earlier.iloc[-1])
    with pytest.raises(DataError):
        kept_reentry_confirm(path, close, as_of=pd.Timestamp("1990-01-01"))
