"""Sleeve gear: UPRO becomes SPY while a shorter trend is off. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_sleeve_gear, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.sleevegear import (
    BANDS,
    OFF_NAME,
    WINDOWS,
    cells,
    kept_sleeve_gear,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _args(spy_o, spy_c, trend, fast, dip, mix, rf, sso_o=None, sso_c=None, upro_o=None, upro_c=None, cost=0.0005):
    sso_o = spy_o if sso_o is None else sso_o
    sso_c = spy_c if sso_c is None else sso_c
    upro_o = spy_o if upro_o is None else upro_o
    upro_c = spy_c if upro_c is None else upro_c
    return dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, fast_on=fast, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=cost, sso_cost=cost, upro_cost=cost,
    )


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [
        (20, 0.0), (20, 0.02), (50, 0.0), (50, 0.02), (100, 0.0), (100, 0.02),
    ]
    assert WINDOWS == (20, 50, 100)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "gear_off"
    assert len({variant_name(*cell) for cell in found}) == 6
    assert variant_name(20, 0.0) == "gear_w020_b0"
    assert variant_name(20, 0.02) == "gear_w020_b2"
    assert variant_name(100, 0.02) == "gear_w100_b2"
    assert OFF_NAME not in {variant_name(*cell) for cell in found}
    for cell in found:
        neighbors = adjacent_cells(cell, [WINDOWS, BANDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_fast_on_matches_the_trend_core_and_an_off_trend_ignores_the_fast_flag():
    idx = pd.bdate_range("2014-01-02", periods=80)
    rng = np.random.default_rng(11)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    spy_c = pd.Series(close_px, index=idx)
    spy_o = spy_c.shift(1)
    spy_o.iloc[0] = spy_c.iloc[0]
    sso_c = spy_c * (1.0 + rng.normal(0.0, 0.002, len(idx)))
    upro_c = spy_c * (1.0 + rng.normal(0.0, 0.004, len(idx)))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(len(idx)) > 0.35, index=idx)
    dip = pd.Series(rng.random(len(idx)) > 0.85, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    fast_on = pd.Series(True, index=idx)
    common = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, dip_on=dip, mix_rebalance=mix,
        core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    plain = run_trend_core(**common, trend_on=trend)
    geared = run_sleeve_gear(**common, trend_on=trend, fast_on=fast_on)
    pd.testing.assert_series_equal(geared.returns, plain.returns, check_names=False)
    pd.testing.assert_series_equal(geared.turnover, plain.turnover, check_names=False)

    slow_off = pd.Series(False, index=idx)
    cash = run_trend_core(**common, trend_on=slow_off)
    ignored = run_sleeve_gear(**common, trend_on=slow_off, fast_on=pd.Series(False, index=idx))
    still = run_sleeve_gear(**common, trend_on=slow_off, fast_on=fast_on)
    pd.testing.assert_series_equal(ignored.returns, cash.returns, check_names=False)
    pd.testing.assert_series_equal(still.returns, cash.returns, check_names=False)

    slow_on = pd.Series(True, index=idx)
    levered = run_sleeve_gear(**common, trend_on=slow_on, fast_on=fast_on)
    plain_on = run_trend_core(**common, trend_on=slow_on)
    delevered = run_sleeve_gear(**common, trend_on=slow_on, fast_on=pd.Series(False, index=idx))
    pd.testing.assert_series_equal(levered.returns, plain_on.returns, check_names=False)
    assert float((delevered.returns - plain_on.returns).abs().max()) > 1e-6
    missing = fast_on.astype(object)
    missing.iloc[4] = np.nan
    with pytest.raises(ValueError, match="fast_on"):
        run_sleeve_gear(**common, trend_on=slow_on, fast_on=missing)


def test_the_upro_crash_is_not_earned_when_the_fast_flag_is_already_off():
    idx = pd.bdate_range("2020-01-02", periods=5)
    spy_o = pd.Series(100.0, index=idx)
    spy_c = pd.Series(100.0, index=idx)
    upro_o = pd.Series([100.0, 100.0, 100.0, 100.0, 80.0], index=idx)
    upro_c = pd.Series([100.0, 100.0, 100.0, 80.0, 80.0], index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    # The flag turns off at the third close, so the fourth session switches at the open.
    fast = pd.Series([True, True, False, False, False], index=idx)
    switched = run_sleeve_gear(**_args(spy_o, spy_c, trend, fast, dip, mix, rf, upro_o=upro_o, upro_c=upro_c))
    stayed = run_sleeve_gear(**_args(
        spy_o, spy_c, trend, pd.Series(True, index=idx), dip, mix, rf, upro_o=upro_o, upro_c=upro_c, cost=0.0,
    ))
    # Sell the UPRO half and buy the SPY half, 5 bp each. The open-to-close crash belongs to SPY, which is flat.
    assert switched.returns.iloc[3] == pytest.approx(-0.0005)
    assert stayed.returns.iloc[3] == pytest.approx(-0.10)
    later = fast.copy()
    later.iloc[-1] = True
    again = run_sleeve_gear(**_args(spy_o, spy_c, trend, later, dip, mix, rf, upro_o=upro_o, upro_c=upro_c))
    pd.testing.assert_series_equal(again.returns, switched.returns)


def test_the_dip_does_not_cover_the_spy_sleeve():
    idx = pd.bdate_range("2020-01-02", periods=3)
    spy_o = pd.Series(100.0, index=idx)
    spy_c = pd.Series(100.0, index=idx)
    sso_o = pd.Series([100.0, 100.0, 100.0], index=idx)
    sso_c = pd.Series([100.0, 100.0, 200.0], index=idx)
    upro_o = pd.Series(100.0, index=idx)
    upro_c = pd.Series(100.0, index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(True, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    fast = pd.Series(False, index=idx)
    result = run_sleeve_gear(**_args(
        spy_o, spy_c, trend, fast, dip, mix, rf,
        sso_o=sso_o, sso_c=sso_c, upro_o=upro_o, upro_c=upro_c, cost=0.0,
    ))
    # Half the book is SSO and doubles. The sleeve is SPY, not SSO and not UPRO.
    assert result.returns.iloc[2] == pytest.approx(0.5)


def test_signal_uses_a_kept_gear_only_while_the_slow_trend_is_on(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "sleeve_gear",
        "walk_forward_picks": [["2018-01-02", "gear_w020_b0"]],
        "variants": [{"name": "gear_w020_b0", "window": 20, "band": 0.0}],
    }]))
    assert kept_sleeve_gear(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "sleeve_gear",
        "walk_forward_picks": [
            ["2018-01-03", "gear_w020_b0"],
            ["2018-02-01", "gear_off"],
        ],
        "variants": [
            {"name": "gear_w020_b0", "window": 20, "band": 0.0},
            {"name": "gear_off", "window": None, "band": None},
        ],
    }]))
    series, info = kept_sleeve_gear(path, close, as_of=idx[-1])
    assert info["sleeve_gear_rule"] == "gear_off"
    assert info["sleeve_gear_live"] is True
    assert bool(series.iloc[-1]) is True

    earlier, early_info = kept_sleeve_gear(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["sleeve_gear_rule"] == "gear_w020_b0"
    assert early_info["sleeve_gear_window"] == 20
    assert early_info["sleeve_gear_live"] is False
    assert earlier.index[-1] < pd.Timestamp("2018-02-01")
    bumped = close.copy()
    bumped.iloc[-1] = 50.0
    unchanged, unchanged_info = kept_sleeve_gear(path, bumped, as_of=pd.Timestamp("2018-01-31"))
    pd.testing.assert_series_equal(unchanged, earlier)
    assert unchanged_info["sleeve_gear_fast_on"] == early_info["sleeve_gear_fast_on"]

    on = pd.Series(True, index=idx)
    fast_off = pd.Series(False, index=idx)
    fast_off.iloc[-2] = True
    delevered = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, sleeve_fast=fast_off, sleeve_gear_info=info,
    )
    assert delevered["weights"]["SPY"] == pytest.approx(1.0)
    assert "UPRO" not in delevered["weights"]
    assert delevered["trade_reason"] == "sleeve_gear"
    assert delevered["diagnostics"]["sleeve_unlevered"] is True

    fast_on = pd.Series(True, index=idx)
    levered = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, sleeve_fast=fast_on, sleeve_gear_info=info,
    )
    assert levered["weights"]["SPY"] == pytest.approx(0.5)
    assert levered["weights"]["UPRO"] == pytest.approx(0.5)

    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, sleeve_fast=fast_off,
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]

    dipped = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, sleeve_fast=pd.Series(False, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]
