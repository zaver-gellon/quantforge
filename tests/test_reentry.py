"""Fast re-entry of the SPY core. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_core_hold, run_fast_reentry, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.reentry import (
    BANDS,
    OFF_NAME,
    WINDOWS,
    cells,
    kept_fast_reentry,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _args(spy_o, spy_c, trend, fast, dip, mix, rf, sso_o=None, sso_c=None, upro_o=None, upro_c=None):
    sso_o = spy_o if sso_o is None else sso_o
    sso_c = spy_c if sso_c is None else sso_c
    upro_o = spy_o if upro_o is None else upro_o
    upro_c = spy_c if upro_c is None else upro_c
    return dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, fast_on=fast, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [
        (20, 0.0), (20, 0.02), (50, 0.0), (50, 0.02), (100, 0.0), (100, 0.02),
    ]
    assert WINDOWS == (20, 50, 100)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "reentry_off"
    assert len({variant_name(*cell) for cell in found}) == 6
    assert variant_name(20, 0.0) == "reentry_w020_b0"
    assert variant_name(20, 0.02) == "reentry_w020_b2"
    assert variant_name(100, 0.02) == "reentry_w100_b2"
    assert OFF_NAME not in {variant_name(*cell) for cell in found}
    for cell in found:
        neighbors = adjacent_cells(cell, [WINDOWS, BANDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_fast_flag_is_ignored_while_the_slow_trend_is_on_and_matches_the_boundaries():
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
    trend = pd.Series(True, index=idx)
    dip = pd.Series(rng.random(len(idx)) > 0.85, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    fast_off = pd.Series(False, index=idx)
    fast_on = pd.Series(True, index=idx)
    common = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, dip_on=dip, mix_rebalance=mix,
        core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    base = run_trend_core(**common, trend_on=trend)
    ignored = run_fast_reentry(**common, trend_on=trend, fast_on=fast_off)
    pd.testing.assert_series_equal(ignored.returns, base.returns)
    pd.testing.assert_series_equal(ignored.turnover, base.turnover)

    slow_off = pd.Series(False, index=idx)
    reentered = run_fast_reentry(**common, trend_on=slow_off, fast_on=fast_on)
    core_on = run_trend_core(**common, trend_on=slow_off)
    pd.testing.assert_series_equal(reentered.returns, core_on.returns)
    pd.testing.assert_series_equal(reentered.turnover, core_on.turnover)

    parked = run_fast_reentry(**common, trend_on=slow_off, fast_on=fast_off)
    cash_core = run_core_hold(**common, trend_on=slow_off, off_core_fraction=0.0)
    pd.testing.assert_series_equal(parked.returns, cash_core.returns)
    pd.testing.assert_series_equal(parked.turnover, cash_core.turnover)
    with pytest.raises(ValueError):
        missing = fast_on.iloc[:-1]
        run_fast_reentry(**common, trend_on=slow_off, fast_on=missing)


def test_the_reentry_fills_at_the_next_open_and_ignores_a_later_flag():
    idx = pd.bdate_range("2020-01-02", periods=5)
    spy_o = pd.Series([100.0, 100.0, 100.0, 120.0, 120.0], index=idx)
    spy_c = pd.Series([100.0, 100.0, 100.0, 120.0, 120.0], index=idx)
    trend = pd.Series(False, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    fast = pd.Series([False, False, True, True, True], index=idx)
    result = run_fast_reentry(**_args(spy_o, spy_c, trend, fast, dip, mix, rf))
    # The gap into the fourth session is earned by cash. The buy is that open.
    assert result.returns.iloc[3] == pytest.approx(-0.00025)
    assert result.exposure.iloc[3] == pytest.approx(0.5)
    assert result.exposure.iloc[2] == pytest.approx(0.0)

    later = fast.copy()
    later.iloc[-1] = False
    again = run_fast_reentry(**_args(spy_o, spy_c, trend, later, dip, mix, rf))
    pd.testing.assert_series_equal(again.returns, result.returns)
    pd.testing.assert_series_equal(again.exposure, result.exposure)
    earlier_flag = fast.copy()
    earlier_flag.iloc[-2] = False
    shifted = run_fast_reentry(**_args(spy_o, spy_c, trend, earlier_flag, dip, mix, rf))
    pd.testing.assert_series_equal(shifted.returns.iloc[:4], result.returns.iloc[:4])
    assert shifted.exposure.iloc[-1] == pytest.approx(0.0)
    assert result.exposure.iloc[-1] == pytest.approx(0.5)


def test_a_same_day_crash_is_not_earned_when_the_flag_turns_on_at_that_close():
    idx = pd.bdate_range("2020-01-02", periods=5)
    spy_o = pd.Series([100.0, 100.0, 100.0, 100.0, 80.0], index=idx)
    spy_c = pd.Series([100.0, 100.0, 100.0, 80.0, 80.0], index=idx)
    trend = pd.Series(False, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    fast = pd.Series([False, False, False, True, True], index=idx)
    result = run_fast_reentry(**_args(spy_o, spy_c, trend, fast, dip, mix, rf))
    assert result.returns.iloc[3] == pytest.approx(0.0)
    assert result.exposure.iloc[3] == pytest.approx(0.0)
    assert result.returns.iloc[4] == pytest.approx(-0.00025)
    assert result.exposure.iloc[4] == pytest.approx(0.5)


def test_signal_uses_a_kept_reentry_only_while_the_slow_trend_is_off(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "fast_reentry",
        "walk_forward_picks": [["2018-01-02", "reentry_w020_b0"]],
        "variants": [{"name": "reentry_w020_b0", "window": 20, "band": 0.0}],
    }]))
    assert kept_fast_reentry(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "fast_reentry",
        "walk_forward_picks": [
            ["2018-01-03", "reentry_w020_b0"],
            ["2018-02-01", "reentry_off"],
        ],
        "variants": [
            {"name": "reentry_w020_b0", "window": 20, "band": 0.0},
            {"name": "reentry_off", "window": None, "band": None},
        ],
    }]))
    series, info = kept_fast_reentry(path, close, as_of=idx[-1])
    assert info["reentry_rule"] == "reentry_off"
    assert info["reentry_live"] is True
    assert series.iloc[-1] == pytest.approx(1.0)
    assert (series.loc[: pd.Timestamp("2018-01-31")] <= 1.0).all()

    earlier, early_info = kept_fast_reentry(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["reentry_rule"] == "reentry_w020_b0"
    assert early_info["reentry_window"] == 20
    assert early_info["reentry_live"] is False
    assert earlier.index[-1] < pd.Timestamp("2018-02-01")
    bumped = close.copy()
    bumped.iloc[-1] = 50.0
    unchanged, unchanged_info = kept_fast_reentry(path, bumped, as_of=pd.Timestamp("2018-01-31"))
    pd.testing.assert_series_equal(unchanged, earlier)
    assert unchanged_info["reentry_fast_on"] == early_info["reentry_fast_on"]

    off = pd.Series(False, index=idx)
    parked = series.copy()
    parked.iloc[-1] = 0.0
    signal = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, core_hold=parked, core_hold_info=info,
    )
    assert signal["weights"]["BIL"] == pytest.approx(1.0)
    assert "SPY" not in signal["weights"]
    assert "UPRO" not in signal["weights"]
    assert signal["trade_reason"] == "fast_reentry"
    assert signal["diagnostics"]["reentry_rule"] == "reentry_off"

    on = pd.Series(True, index=idx)
    invested = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, core_hold=parked, core_hold_info=info,
    )
    assert invested["weights"]["SPY"] == pytest.approx(0.5)
    assert invested["weights"]["UPRO"] == pytest.approx(0.5)
