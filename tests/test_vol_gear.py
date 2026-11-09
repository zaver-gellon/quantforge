"""Realized-vol leverage on the UPRO sleeve. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_sleeve_gear, run_trend_core, run_vol_gear
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.volgear import (
    LOOKBACKS,
    MAX_LEVERAGE,
    MIN_LEVERAGE,
    OFF_NAME,
    TARGET_VOLS,
    cells,
    kept_vol_gear,
    sleeve_leverage,
    upro_share,
    variant_name,
)


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == (21, 0.36)
    assert found[-1] == (126, 0.60)
    assert len(found) == 9
    assert LOOKBACKS == (21, 63, 126)
    assert TARGET_VOLS == (0.36, 0.48, 0.60)
    assert MIN_LEVERAGE == 1.0
    assert MAX_LEVERAGE == 3.0
    assert OFF_NAME == "vgear_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 9
    assert variant_name(21, 0.36) == "vgear_lb021_tv36"
    assert variant_name(63, 0.48) == "vgear_lb063_tv48"
    assert variant_name(126, 0.60) == "vgear_lb126_tv60"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [LOOKBACKS, TARGET_VOLS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_leverage_does_not_read_a_future_close_and_stays_inside_the_band():
    idx = pd.bdate_range("2014-01-02", periods=80)
    rng = np.random.default_rng(4)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)
    base = sleeve_leverage(close, 21, 0.48)
    future = close.copy()
    future.iloc[40:] = close.iloc[40] * 0.5
    again = sleeve_leverage(future, 21, 0.48)
    pd.testing.assert_series_equal(base.iloc[:40], again.iloc[:40])
    assert float(base.iloc[19]) == 3.0
    assert float(base.min()) >= 1.0 - 1e-12
    assert float(base.max()) <= 3.0 + 1e-12
    quiet = pd.Series(100.0 * (1.001 ** np.arange(len(idx))), index=idx)
    assert float(sleeve_leverage(quiet, 21, 0.48).iloc[-1]) == 3.0
    low = sleeve_leverage(close, 21, 0.36)
    high = sleeve_leverage(close, 21, 0.60)
    assert bool((high.iloc[21:] + 1e-12 >= low.iloc[21:]).all())
    share = upro_share(base)
    assert share.iloc[-1] == pytest.approx((float(base.iloc[-1]) - 1.0) / 2.0)
    assert float(share.min()) >= 0.0
    assert float(share.max()) <= 1.0


def _book(n=80, seed=5):
    idx = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(seed)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    spy_c = pd.Series(close_px, index=idx)
    spy_o = spy_c.shift(1)
    spy_o.iloc[0] = spy_c.iloc[0]
    sso_c = spy_c * (1.0 + rng.normal(0.0, 0.002, n))
    upro_c = spy_c * (1.0 + rng.normal(0.0, 0.004, n))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(n) > 0.3, index=idx)
    dip = pd.Series(rng.random(n) > 0.85, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    return idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf


def test_full_upro_share_matches_the_trend_core_and_zero_matches_the_spy_sleeve():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf = _book()
    common = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    plain = run_trend_core(**common)
    full = run_vol_gear(**common, upro_share=pd.Series(1.0, index=idx))
    pd.testing.assert_series_equal(full.returns, plain.returns, check_names=False)
    closed = run_vol_gear(**common, upro_share=pd.Series(1.0, index=idx), execution="next_close")
    plain_close = run_trend_core(**common, execution="next_close")
    pd.testing.assert_series_equal(closed.returns, plain_close.returns, check_names=False)
    spy_sleeve = run_vol_gear(**common, upro_share=pd.Series(0.0, index=idx))
    geared = run_sleeve_gear(**common, fast_on=pd.Series(False, index=idx))
    pd.testing.assert_series_equal(spy_sleeve.returns, geared.returns, check_names=False)
    off = trend.copy()
    off.iloc[:] = False
    ignored = run_vol_gear(**{**common, "trend_on": off}, upro_share=pd.Series(0.0, index=idx))
    parked = run_trend_core(**{**common, "trend_on": off})
    pd.testing.assert_series_equal(ignored.returns, parked.returns, check_names=False)


def test_share_known_at_the_close_applies_on_the_next_session():
    idx = pd.bdate_range("2020-01-02", periods=6)
    spy_c = pd.Series(100.0 * (1.01 ** np.arange(len(idx))), index=idx)
    spy_o = spy_c.shift(1)
    spy_o.iloc[0] = spy_c.iloc[0]
    upro_c = pd.Series(100.0 * (1.03 ** np.arange(len(idx))), index=idx)
    upro_o = upro_c.shift(1)
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    share = pd.Series(0.5, index=idx)
    share.iloc[2] = 0.0
    result = run_vol_gear(
        spy_o, spy_c, spy_o, spy_c, upro_o, upro_c,
        trend, share, dip, mix, 0.5, 0.5, rf,
    )
    # share[0] is filled at session 1: half the book in SPY (core plus half the sleeve) and a quarter in UPRO.
    assert result.returns.iloc[1] == pytest.approx(0.5 * 0.01 + 0.25 * 0.01 + 0.25 * 0.03)
    # The blend drifts, so the next session is not a fresh 50/25/25, but UPRO is still held.
    assert result.returns.iloc[2] > 0.01
    assert result.turnover.iloc[2] == 0.0
    # share[2] is the decision at that close. Session 3 holds the sleeve in SPY, so the book is all SPY.
    assert result.returns.iloc[3] == pytest.approx(0.01)
    assert result.turnover.iloc[3] > 0.0


def test_signal_blend_stays_in_equities_and_a_rejected_rule_does_not_load(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    on = pd.Series(True, index=idx)
    share = pd.Series(1.0, index=idx)
    share.iloc[-1] = 0.5
    spec = TrendCoreSpec(window=20)
    blended = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, sleeve_upro_share=share,
        vol_gear_info={"vol_gear_rule": "vgear_lb021_tv48", "vol_gear_live": False},
    )
    assert blended["weights"]["SPY"] == pytest.approx(0.75)
    assert blended["weights"]["UPRO"] == pytest.approx(0.25)
    assert "BIL" not in blended["weights"]
    assert blended["trade_reason"] == "vol_gear"
    assert blended["diagnostics"]["sleeve_leverage_applied"] == pytest.approx(2.0)
    plain = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on,
    )
    full = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, sleeve_upro_share=pd.Series(1.0, index=idx),
    )
    assert full["weights"] == plain["weights"]
    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, sleeve_upro_share=pd.Series(0.0, index=idx),
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]
    dipped = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, sleeve_upro_share=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]

    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "id": "2026-11-09-vol-gear",
        "kept": False,
        "status": "tested",
        "overlay": "vol_gear",
    }]))
    assert kept_vol_gear(path, close, as_of=idx[-1]) == (None, None)
    path.write_text(json.dumps([{
        "id": "2026-11-09-vol-gear",
        "kept": True,
        "status": "tested",
        "overlay": "vol_gear",
        "variants": [{
            "name": "vgear_lb021_tv48",
            "lookback": 21,
            "target_vol": 0.48,
        }],
        "walk_forward_picks": [["2018-01-02", "vgear_off"], ["2018-02-01", "vgear_lb021_tv48"]],
    }]))
    series, info = kept_vol_gear(path, close, as_of=idx[-1])
    assert info["vol_gear_rule"] == "vgear_lb021_tv48"
    assert info["vol_gear_lookback"] == 21
    assert info["vol_gear_target"] == pytest.approx(0.48)
    assert info["vol_gear_live"] is False
    assert float(series.loc[:pd.Timestamp("2018-01-31")].eq(1.0).all())
    later = series.loc[pd.Timestamp("2018-02-01"):]
    expected = upro_share(sleeve_leverage(close, 21, 0.48)).reindex(later.index)
    pd.testing.assert_series_equal(later, expected, check_names=False)
    earlier, early_info = kept_vol_gear(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["vol_gear_live"] is True
    assert float(earlier.iloc[-1]) == 1.0
