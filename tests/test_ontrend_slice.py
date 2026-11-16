"""Constant diversifier slice inside the on-trend UPRO sleeve. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_sleeve_slice, run_trend_core
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.onslice import (
    ASSETS,
    OFF_NAME,
    SLICES,
    cells,
    kept_ontrend_slice,
    listing_ready,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _prices(n=8, step=0.0, start="2016-01-04"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100.0 * np.cumprod(1.0 + np.full(n, step)), index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    return idx, open_, close


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == ("GLD", 0.10)
    assert found[-1] == ("TLT", 0.30)
    assert len(found) == 9
    assert ASSETS == ("GLD", "IEF", "TLT")
    assert SLICES == (0.10, 0.20, 0.30)
    assert OFF_NAME == "slice_off"
    names = [variant_name(*cell) for cell in found]
    assert len(set(names)) == 9
    assert names == sorted(names)
    assert variant_name("GLD", 0.10) == "zslice_gld_w10"
    assert variant_name("TLT", 0.30) == "zslice_tlt_w30"
    assert OFF_NAME < names[0]
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [ASSETS, SLICES])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_listing_ready_does_not_fill_backward_or_a_long_gap():
    idx = pd.bdate_range("2018-01-02", periods=12)
    close = pd.Series(np.nan, index=idx, dtype=float)
    close.iloc[2] = 100.0
    close.iloc[3] = 101.0
    close.iloc[10] = 110.0
    ready = listing_ready(close, idx)
    assert not bool(ready.iloc[:2].any())
    assert bool(ready.iloc[2])
    assert bool(ready.iloc[3])
    # Five sessions after the last print stay ready. The sixth does not.
    assert bool(ready.iloc[8])
    assert not bool(ready.iloc[9])
    assert bool(ready.iloc[10])
    future = close.copy()
    future.iloc[11] = 50.0
    again = listing_ready(future, idx)
    pd.testing.assert_series_equal(ready.iloc[:11], again.iloc[:11])


def _book(n=6):
    idx, spy_o, spy_c = _prices(n)
    sso_o = pd.Series(110.0, index=idx)
    sso_c = pd.Series(110.0, index=idx)
    upro_o = pd.Series(120.0, index=idx)
    upro_c = pd.Series(120.0, index=idx)
    div_o = pd.Series(100.0, index=idx)
    div_c = pd.Series(100.0, index=idx)
    # A 10% open-to-close rally on the third session, and again on the last.
    div_c.iloc[2] = 110.0
    div_o.iloc[3:] = 110.0
    div_c.iloc[3:] = 110.0
    div_c.iloc[-1] = 121.0
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    return idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, div_o, div_c, trend, dip, mix, rf


def _run(slice_weight, ready, trend, **book):
    idx = book["idx"]
    return run_sleeve_slice(
        book["spy_o"], book["spy_c"], book["sso_o"], book["sso_c"], book["upro_o"], book["upro_c"],
        book["div_o"], book["div_c"], trend, ready, book["dip"], book["mix"],
        0.5, 0.5, slice_weight, book["rf"],
    )


def test_zero_slice_unready_and_off_trend_match_the_trend_core():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, div_o, div_c, trend, dip, mix, rf = _book()
    common = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
    )
    base = run_trend_core(**common)
    book = dict(
        idx=idx, spy_o=spy_o, spy_c=spy_c, sso_o=sso_o, sso_c=sso_c,
        upro_o=upro_o, upro_c=upro_c, div_o=div_o, div_c=div_c,
        dip=dip, mix=mix, rf=rf,
    )
    zero = _run(0.0, pd.Series(True, index=idx), trend, **book)
    unready = _run(0.20, pd.Series(False, index=idx), trend, **book)
    off = _run(0.20, pd.Series(True, index=idx), pd.Series(False, index=idx), **book)
    off_base = run_trend_core(**{**common, "trend_on": pd.Series(False, index=idx)})
    pd.testing.assert_series_equal(zero.returns, base.returns)
    pd.testing.assert_series_equal(unready.returns, base.returns)
    pd.testing.assert_series_equal(off.returns, off_base.returns)
    # Missing diversifier prices are not read when the slice is never held.
    blank = div_c.copy() * np.nan
    blank_open = div_o.copy() * np.nan
    ignored = run_sleeve_slice(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, blank_open, blank,
        trend, pd.Series(False, index=idx), dip, mix, 0.5, 0.5, 0.20, rf,
    )
    pd.testing.assert_series_equal(ignored.returns, base.returns)


def test_on_trend_slice_earns_the_diversifier_and_ignores_a_future_print():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, div_o, div_c, trend, dip, mix, rf = _book()
    ready = pd.Series(True, index=idx)
    book = dict(
        idx=idx, spy_o=spy_o, spy_c=spy_c, sso_o=sso_o, sso_c=sso_c,
        upro_o=upro_o, upro_c=upro_c, div_o=div_o, div_c=div_c,
        dip=dip, mix=mix, rf=rf,
    )
    held = _run(0.10, ready, trend, **book)
    # Session 1 allocates a flat book. Session 2 earns 10% on a 10% slice.
    assert held.returns.iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert held.returns.iloc[2] == pytest.approx(0.01, abs=1e-12)
    later = div_c.copy()
    later.iloc[-1] = 200.0
    book["div_c"] = later
    again = _run(0.10, ready, trend, **book)
    pd.testing.assert_series_equal(held.returns.iloc[:-1], again.returns.iloc[:-1])
    assert again.returns.iloc[-1] != pytest.approx(held.returns.iloc[-1])


def test_signal_moves_only_the_on_trend_sleeve(tmp_path):
    idx = pd.bdate_range("2024-01-02", periods=30)
    close = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    trend = pd.Series(True, index=idx)
    ready = pd.Series(True, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], apply_dip=False, trend_override=trend,
        slice_symbol="GLD", slice_weight=0.10, slice_ready=ready,
        slice_info={"slice_rule": "zslice_gld_w10", "slice_live": False},
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["UPRO"] == pytest.approx(0.4)
    assert payload["weights"]["GLD"] == pytest.approx(0.1)
    assert payload["diagnostics"]["slice_applied"] is True
    off = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], apply_dip=False,
        trend_override=pd.Series(False, index=idx),
        slice_symbol="GLD", slice_weight=0.10, slice_ready=ready,
    )
    assert off["weights"] == {"SPY": 0.5, "BIL": 0.5}
    assert off["diagnostics"]["slice_applied"] is False
    # A rejected rule is not reloaded.
    assert kept_ontrend_slice(tmp_path / "missing.json") == (None, None, None, None)
    log = tmp_path / "iterations.json"
    log.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "ontrend_slice",
        "walk_forward_picks": [["2024-01-02", "zslice_gld_w10"]],
        "variants": [{"name": "zslice_gld_w10", "asset": "GLD", "slice_weight": 0.10}],
    }]))
    assert kept_ontrend_slice(log, {"GLD": close}, as_of=idx[-1]) == (None, None, None, None)
