"""Off-trend dip cut. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_core_hold, run_dip_cut, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.dipcut import (
    CUTS,
    OFF_NAME,
    SCOPES,
    cells,
    kept_dip_cut,
    offtrend_cut,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _down_book():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
    trend = pd.Series([True, True, True, True, False, True], index=idx)
    return idx, close, trend


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert SCOPES == ("born", "carry")
    assert CUTS == (0.50, 1.0)
    assert OFF_NAME == "cut_off"
    assert found == [("born", 0.50), ("born", 1.0), ("carry", 0.50), ("carry", 1.0)]
    names = [variant_name(*cell) for cell in found]
    assert names == [
        "zcut_born_c050", "zcut_born_c100",
        "zcut_carry_c050", "zcut_carry_c100",
    ]
    assert OFF_NAME < min(names)
    for cell in found:
        neighbors = adjacent_cells(cell, [SCOPES, CUTS])
        assert neighbors
        assert cell not in neighbors
    assert set(adjacent_cells(("carry", 1.0), [SCOPES, CUTS])) == {("born", 1.0), ("carry", 0.50)}
    with pytest.raises(ValueError):
        variant_name("fresh", 1.0)
    with pytest.raises(ValueError):
        variant_name("carry", 0.25)


def test_clock_matches_the_live_dip_and_does_not_read_a_future_close():
    idx = pd.bdate_range("2019-01-02", periods=40)
    rng = np.random.default_rng(4)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx)
    trend = pd.Series(rng.random(len(idx)) > 0.4, index=idx)
    live = dip_active(close)
    for scope, cut in cells():
        invested, sso, clock = offtrend_cut(close, trend, scope, cut)
        pd.testing.assert_series_equal(clock.astype(bool), live.astype(bool), check_names=False)
        assert invested.ge(-1e-12).all() and invested.le(1.0 + 1e-12).all()
        # On-trend holds stay fully invested in SSO.
        on_hold = clock & trend
        assert invested.loc[on_hold].eq(1.0).all()
        assert sso.loc[on_hold].all()
        # A quiet off-trend session stays in SPY.
        quiet = ~clock
        assert invested.loc[quiet].eq(1.0).all()
        assert not sso.loc[quiet].any()
    prefix_inv, prefix_sso, prefix_clock = offtrend_cut(close.iloc[:30], trend.iloc[:30], "carry", 1.0)
    full_inv, full_sso, full_clock = offtrend_cut(close, trend, "carry", 1.0)
    pd.testing.assert_series_equal(prefix_inv, full_inv.iloc[:30])
    pd.testing.assert_series_equal(prefix_sso, full_sso.iloc[:30])
    pd.testing.assert_series_equal(prefix_clock, full_clock.iloc[:30])


def test_carry_cashes_a_hold_that_crosses_the_flip_and_born_does_not():
    _idx, close, trend = _down_book()
    live = dip_active(close)
    assert list(live.astype(bool)) == [False, False, False, True, True, True]
    carry_inv, carry_sso, _clock = offtrend_cut(close, trend, "carry", 1.0)
    born_inv, born_sso, _clock = offtrend_cut(close, trend, "born", 1.0)
    # The hold starts while the band is on. The next close is off-trend.
    assert list(carry_inv) == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert list(carry_sso.astype(bool)) == [False, False, False, True, False, True]
    assert list(born_inv) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert list(born_sso.astype(bool)) == [False, False, False, True, True, True]
    half_inv, half_sso, _clock = offtrend_cut(close, trend, "carry", 0.5)
    assert list(half_inv) == [1.0, 1.0, 1.0, 1.0, 0.5, 1.0]
    assert list(half_sso.astype(bool)) == [False, False, False, True, False, True]


def test_born_cashes_a_hold_that_starts_off_trend():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 7.5], index=idx)
    trend = pd.Series(False, index=idx)
    born_inv, born_sso, clock = offtrend_cut(close, trend, "born", 1.0)
    carry_inv, carry_sso, _clock = offtrend_cut(close, trend, "carry", 0.5)
    # Three down closes enter at the fourth close. The up close ends the hold.
    assert list(clock.astype(bool)) == [False, False, False, True, True, False]
    assert list(born_inv) == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert not born_sso.any()
    assert list(carry_inv) == [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
    assert not carry_sso.any()


def test_identity_path_matches_the_trend_core_and_a_full_cut_matches_cash():
    idx = pd.bdate_range("2014-01-02", periods=250)
    rng = np.random.default_rng(11)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    close = pd.Series(close_px, index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    sso_c = close * (1.0 + rng.normal(0.0, 0.002, len(idx)))
    upro_c = close * (1.0 + rng.normal(0.0, 0.002, len(idx)))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(len(idx)) > 0.3, index=idx)
    dip = pd.Series(rng.random(len(idx)) > 0.8, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    args = dict(
        spy_open=open_, spy_close=close, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    base = run_trend_core(**args, dip_on=dip)
    same = run_dip_cut(**args, core_invested=pd.Series(1.0, index=idx), core_sso=dip)
    pd.testing.assert_series_equal(same.returns, base.returns)
    pd.testing.assert_series_equal(same.turnover, base.turnover)
    off = pd.Series(False, index=idx)
    cashed_args = dict(args)
    cashed_args["trend_on"] = off
    cashed = run_dip_cut(**cashed_args, core_invested=pd.Series(0.0, index=idx), core_sso=off)
    parked = run_core_hold(**cashed_args, dip_on=off, off_core_fraction=0.0)
    pd.testing.assert_series_equal(cashed.returns, parked.returns)
    later = dip.copy()
    later.iloc[-1] = not bool(later.iloc[-1])
    unchanged = run_dip_cut(**args, core_invested=pd.Series(1.0, index=idx), core_sso=later)
    pd.testing.assert_series_equal(unchanged.returns, same.returns)
    with pytest.raises(ValueError):
        run_dip_cut(**args, core_invested=pd.Series(1.5, index=idx), core_sso=dip)


def test_half_cut_earns_half_the_core_while_the_trend_is_off():
    idx = pd.bdate_range("2020-01-02", periods=3)
    flat = pd.Series(100.0, index=idx)
    spy_c = pd.Series([100.0, 100.0, 110.0], index=idx)
    trend = pd.Series(False, index=idx)
    sso_flag = pd.Series(False, index=idx)
    invested = pd.Series(0.5, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    cut = run_dip_cut(
        flat, spy_c, flat, flat, flat, flat, trend, invested, sso_flag, mix,
        0.5, 0.5, rf,
    )
    # 25% of the book is SPY at +10%. The sleeve is cash.
    assert cut.returns.iloc[-1] == pytest.approx(0.025)
    full = run_dip_cut(
        flat, spy_c, flat, flat, flat, flat, trend, pd.Series(1.0, index=idx), sso_flag, mix,
        0.5, 0.5, rf,
    )
    assert full.returns.iloc[-1] == pytest.approx(0.05)


def test_signal_cashes_the_core_and_keeps_the_dip_clock():
    idx = pd.bdate_range("2018-01-02", periods=30)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    off = pd.Series(False, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(window=10), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        trend_override=off, apply_dip=False,
        core_hold=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(0.0, index=idx),
        dip_cut_info={"dip_cut_rule": "zcut_carry_c100", "dip_clock_on": True, "dip_cut": 1.0},
    )
    assert payload["weights"] == {"BIL": 1.0}
    assert payload["diagnostics"]["trend_on"] is False
    assert payload["diagnostics"]["dip_on"] is True
    assert payload["diagnostics"]["dip_cut_rule"] == "zcut_carry_c100"
    half = build_trend_signal(
        TrendCoreSpec(window=10), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        trend_override=off, apply_dip=False,
        core_hold=pd.Series(0.5, index=idx),
        dip_boost=pd.Series(0.0, index=idx),
        dip_cut_info={"dip_cut_rule": "zcut_carry_c050", "dip_clock_on": True},
    )
    assert half["weights"]["SPY"] == pytest.approx(0.25)
    assert half["weights"]["BIL"] == pytest.approx(0.75)
    assert "SSO" not in half["weights"]


def test_rejected_rule_does_not_change_the_signal(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=30)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": False,
            "status": "tested",
            "overlay": "dip_cut",
            "walk_forward_picks": [["2018-01-02", "zcut_carry_c100"]],
            "variants": [{"name": "zcut_carry_c100", "scope": "carry", "cut": 1.0}],
        }
    ]))
    hold, boost, info = kept_dip_cut(path, close, as_of=idx[-1])
    assert hold is None and boost is None and info is None


def test_kept_rule_reloads_only_closes_through_as_of(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(80, 70, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "dip_cut",
            "walk_forward_picks": [["2018-01-02", "zcut_carry_c100"]],
            "variants": [{"name": "zcut_carry_c100", "scope": "carry", "cut": 1.0}],
        }
    ]))
    hold, boost, info = kept_dip_cut(path, close, as_of=idx[40])
    assert info["dip_cut_rule"] == "zcut_carry_c100"
    assert info["dip_cut_scope"] == "carry"
    assert info["dip_cut"] == 1.0
    assert len(hold) == 41
    assert len(boost) == 41
    # Forty sessions is before the 200-day average exists, so the band is off.
    # The three-down clock is on, so the whole core is cash.
    assert float(hold.iloc[-1]) == 0.0
    assert float(boost.iloc[-1]) == 0.0
    assert info["dip_clock_on"] is True

    off_path = tmp_path / "off.json"
    off_path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "dip_cut",
            "walk_forward_picks": [["2018-01-02", "cut_off"]],
            "variants": [{"name": "cut_off"}],
        }
    ]))
    hold, boost, info = kept_dip_cut(off_path, close, as_of=idx[-1])
    assert hold is None and boost is None and info is None
