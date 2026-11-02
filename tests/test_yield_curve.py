"""Yield-curve recession clock. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.engine.backtest import run_curve_clock, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.curve import (
    DEPTHS,
    LENGTHS,
    OFF_NAME,
    cells,
    kept_curve_clock,
    recession_clock,
    treasury_spread,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _book(n=400, seed=5):
    idx = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(seed)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    close = pd.Series(close_px, index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    sso_c = close * (1.0 + rng.normal(0.0, 0.001, n))
    upro_c = close * (1.0 + rng.normal(0.0, 0.001, n))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(n) > 0.35, index=idx)
    dip = pd.Series(rng.random(n) > 0.85, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    return idx, open_, close, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [
        (0.0, 126), (0.0, 252), (0.0, 504),
        (0.25, 126), (0.25, 252), (0.25, 504),
        (0.50, 126), (0.50, 252), (0.50, 504),
    ]
    assert DEPTHS == (0.0, 0.25, 0.50)
    assert LENGTHS == (126, 252, 504)
    assert OFF_NAME == "curve_off"
    assert len({variant_name(*cell) for cell in found}) == 9
    assert variant_name(0.0, 126) == "curve_d000_k126"
    assert variant_name(0.25, 252) == "curve_d025_k252"
    assert variant_name(0.50, 504) == "curve_d050_k504"
    for cell in found:
        neighbors = adjacent_cells(cell, [DEPTHS, LENGTHS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_clock_starts_on_a_crossing_and_does_not_extend():
    idx = pd.bdate_range("2020-01-02", periods=8)
    spread = pd.Series([0.4, 0.1, -0.1, 0.2, -0.2, -0.3, 0.5, 0.4], index=idx)
    clock = recession_clock(spread, 0.0, 3)
    assert list(clock.astype(bool)) == [False, False, True, True, True, False, False, False]
    # Already inverted at the first print is not a crossing.
    early = pd.Series([-0.2, -0.3, 0.1, -0.05], index=idx[:4])
    opened = recession_clock(early, 0.0, 2)
    assert list(opened.astype(bool)) == [False, False, False, True]
    # A crossing during the run does not add sessions.
    inside = pd.Series([0.1, -0.1, 0.2, -0.1, -0.2, 0.3], index=idx[:6])
    held = recession_clock(inside, 0.0, 3)
    assert list(held.astype(bool)) == [False, True, True, True, False, False]
    # Depth is an inversion, not a raw negative spread.
    shallow = pd.Series([0.2, -0.1, -0.3], index=idx[:3])
    assert list(recession_clock(shallow, 0.25, 2).astype(bool)) == [False, False, True]


def test_clock_ignores_a_missing_print_and_a_future_crossing():
    idx = pd.bdate_range("2020-01-02", periods=5)
    spread = pd.Series([0.2, np.nan, -0.1, 0.3, -0.2], index=idx)
    clock = recession_clock(spread, 0.0, 2)
    # The missing print does not itself cross. The following negative print
    # crosses against the last finite side, which was above, and the last
    # print is a new crossing after that run has ended.
    assert list(clock.astype(bool)) == [False, False, True, True, True]
    prefix = recession_clock(spread.iloc[:3], 0.0, 2)
    pd.testing.assert_series_equal(prefix, clock.iloc[:3])


def test_percent_yields_become_the_spread_and_a_times_ten_quote_is_refused():
    idx = pd.bdate_range("2018-01-02", periods=6)
    tnx = pd.Series([2.5, 2.6, 2.4, 2.7, 2.8, 2.3], index=idx)
    irx = pd.Series([1.0, 1.1, 1.2, 0.9, 1.0, 1.1], index=idx)
    spread = treasury_spread(tnx, irx, idx)
    assert spread.iloc[0] == pytest.approx(1.5)
    assert spread.iloc[-1] == pytest.approx(1.2)
    # A one-session hole is filled from the prior print. A later print is not.
    hole = tnx.copy()
    hole.iloc[2] = np.nan
    filled = treasury_spread(hole, irx, idx)
    assert filled.iloc[2] == pytest.approx(2.6 - 1.2)
    scaled = tnx * 10.0
    with pytest.raises(DataError):
        treasury_spread(scaled, irx, idx)


def test_cash_off_matches_the_trend_core_book():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf = _book()
    args = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    base = run_trend_core(**args)
    flat = run_curve_clock(**args, cash_on=pd.Series(False, index=idx))
    pd.testing.assert_series_equal(flat.returns, base.returns)
    pd.testing.assert_series_equal(flat.turnover, base.turnover)


def test_cash_on_earns_the_bill_and_ignores_a_future_flag():
    idx = pd.bdate_range("2020-01-02", periods=4)
    spy_c = pd.Series([100.0, 100.0, 100.0, 50.0], index=idx)
    spy_o = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    mix.iloc[0] = True
    rf = pd.Series(0.0, index=idx)
    args = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=spy_o, sso_close=spy_c,
        upro_open=spy_o, upro_close=spy_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0, sso_cost=0.0, upro_cost=0.0,
    )
    # The crash is the open-to-close move on the last day. A clock known at
    # the prior close sells at that open, so the book misses the crash.
    cash = pd.Series([False, False, True, False], index=idx)
    parked = run_curve_clock(**args, cash_on=cash)
    assert parked.returns.iloc[-1] == pytest.approx(0.0)
    assert parked.exposure.iloc[-1] == pytest.approx(0.0)
    exposed = run_curve_clock(**args, cash_on=pd.Series(False, index=idx))
    assert exposed.returns.iloc[-1] < -0.2
    # A flag after the crash does not change the crash session.
    later = cash.copy()
    later.iloc[-1] = True
    again = run_curve_clock(**args, cash_on=later)
    assert again.returns.iloc[-1] == pytest.approx(parked.returns.iloc[-1])
    # The decision at the crash close affects only the next session, which
    # does not exist. The prior session is unchanged.
    pd.testing.assert_series_equal(again.returns.iloc[:-1], parked.returns.iloc[:-1])


def test_kept_clock_uses_only_the_selected_cell(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=6)
    spread = pd.Series([0.5, 0.2, -0.05, -0.4, 0.3, 0.2], index=idx)
    close = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "yield_curve",
    }]))
    assert kept_curve_clock(path, spread, as_of=idx[-1]) == (None, None)
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "yield_curve",
        "walk_forward_picks": [
            ["2018-01-02", "curve_d000_k126"],
            ["2018-01-05", "curve_off"],
        ],
        "variants": [
            {"name": "curve_d000_k126", "depth": 0.0, "length": 126},
            {"name": "curve_d025_k126", "depth": 0.25, "length": 126},
            {"name": "curve_off", "depth": None, "length": None},
        ],
    }]))
    # 2018-01-05 is the fourth session (Tue after Mon Jan 1 holiday: Jan 2, 3, 4, 5).
    series, info = kept_curve_clock(path, spread, as_of=idx[-1])
    assert info["curve_rule"] == "curve_off"
    assert info["curve_live"] is True
    assert bool(series.iloc[-1]) is False
    earlier, early_info = kept_curve_clock(path, spread, as_of=pd.Timestamp("2018-01-04"))
    assert early_info["curve_rule"] == "curve_d000_k126"
    # Spread crosses below 0 on the third session, so that cell is in cash.
    assert bool(earlier.iloc[2]) is True
    # The deeper cell would not have crossed yet. The selected cell is the shallow one.
    deep = recession_clock(spread.loc[: pd.Timestamp("2018-01-04")], 0.25, 126)
    assert bool(deep.iloc[2]) is False
    on = pd.Series(True, index=idx)
    parked = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[2], apply_dip=False,
        curve_on=earlier, curve_info=early_info,
    )
    assert parked["weights"] == {"BIL": 1.0}
    assert parked["trade_reason"] == "curve_clock"
    assert parked["diagnostics"]["curve_on"] is True
    # A later as_of under the off cell is the plain book, and it does not
    # read the crossing that a deeper cell would see after as_of.
    plain = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], apply_dip=False,
        curve_on=series, curve_info=info,
    )
    # The off cell leaves the sleeve in BIL because the 200-day average is not
    # ready, and it leaves the core in SPY. It does not park the whole book.
    assert plain["weights"].get("SPY") == pytest.approx(0.5)
    assert plain["weights"].get("BIL") == pytest.approx(0.5)
    assert plain["diagnostics"]["curve_on"] is False
