"""Two-sided UPRO drift band. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.engine.backtest import run_drift_band, run_symmetric_drift, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.symdrift import (
    BANDS,
    OFF_NAME,
    cells,
    kept_sym_drift,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def _args(spy, upro_o, upro_c, trend, dip, rf, sso_o=None, sso_c=None):
    sso_o = spy if sso_o is None else sso_o
    sso_c = spy if sso_c is None else sso_c
    return dict(
        spy_open=spy, spy_close=spy, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0, sso_cost=0.0, upro_cost=0.0,
    )


def test_grid_is_fixed_and_the_live_book_sorts_first():
    found = cells()
    assert found == [0.02, 0.04, 0.06, 0.08, 0.10]
    assert BANDS == (0.02, 0.04, 0.06, 0.08, 0.10)
    assert OFF_NAME == "sym_live"
    names = [variant_name(band) for band in found]
    assert len(set(names)) == 5
    assert variant_name(0.02) == "zsym_b02"
    assert variant_name(0.10) == "zsym_b10"
    assert OFF_NAME < min(names)
    for band in found:
        neighbors = adjacent_cells((band,), [BANDS])
        assert neighbors
        assert (band,) not in neighbors
        for neighbor in neighbors:
            assert neighbor[0] in found


def test_both_sides_reset_next_open_and_a_wide_band_does_not():
    idx = pd.bdate_range("2020-01-02", periods=4)
    spy = pd.Series(100.0, index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    up_o = pd.Series([100.0, 100.0, 140.0, 140.0], index=idx)
    up_c = pd.Series([100.0, 140.0, 140.0, 140.0], index=idx)
    down_o = pd.Series([100.0, 100.0, 60.0, 60.0], index=idx)
    down_c = pd.Series([100.0, 60.0, 60.0, 60.0], index=idx)
    up_args = _args(spy, up_o, up_c, trend, dip, rf)
    down_args = _args(spy, down_o, down_c, trend, dip, rf)
    tight_up, mix_up = run_symmetric_drift(**up_args, band=0.05)
    one_sided, one_mix = run_drift_band(**up_args, band=0.05)
    assert list(mix_up) == [False, True, False, False]
    assert list(one_mix) == list(mix_up)
    pd.testing.assert_series_equal(tight_up.returns, one_sided.returns)
    tight_down, mix_down = run_symmetric_drift(**down_args, band=0.05)
    ignored, ignored_mix = run_drift_band(**down_args, band=0.05)
    assert list(mix_down) == [False, True, False, False]
    assert list(ignored_mix) == [False, False, False, False]
    assert tight_down.turnover.iloc[2] == pytest.approx(0.125)
    assert ignored.turnover.iloc[2] == 0.0
    wide, wide_mix = run_symmetric_drift(**down_args, band=0.20)
    assert list(wide_mix) == [False, False, False, False]
    assert wide.turnover.iloc[2] == 0.0


def test_a_cash_sleeve_is_not_rebalanced():
    idx = pd.bdate_range("2020-01-02", periods=4)
    falling = pd.Series([100.0, 100.0, 50.0, 50.0], index=idx)
    flat = pd.Series(100.0, index=idx)
    cash, cash_mix = run_symmetric_drift(
        **_args(falling, flat, flat, pd.Series(False, index=idx), pd.Series(False, index=idx), pd.Series(0.0, index=idx)),
        band=0.02,
    )
    assert list(cash_mix) == [False, False, False, False]
    assert cash.turnover.iloc[2] == 0.0
    assert cash.turnover.iloc[3] == 0.0


def test_a_band_that_cannot_bind_matches_a_book_with_no_calendar():
    idx = pd.bdate_range("2014-01-02", periods=80)
    rng = np.random.default_rng(3)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx))))
    spy_c = pd.Series(close_px, index=idx)
    spy_o = spy_c.shift(1)
    spy_o.iloc[0] = spy_c.iloc[0]
    upro_c = spy_c * (1.0 + rng.normal(0.0, 0.01, len(idx)))
    upro_o = upro_c.shift(1)
    upro_o.iloc[0] = upro_c.iloc[0]
    sso_c = spy_c * (1.0 + rng.normal(0.0, 0.004, len(idx)))
    sso_o = sso_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    trend = pd.Series(rng.random(len(idx)) > 0.3, index=idx)
    dip = pd.Series(rng.random(len(idx)) > 0.85, index=idx)
    rf = pd.Series(0.02 / 252.0, index=idx)
    common = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    wide, _mix = run_symmetric_drift(**common, band=1.0)
    quiet = run_trend_core(**common, mix_rebalance=pd.Series(False, index=idx))
    recorded = run_trend_core(**common, mix_rebalance=pd.Series(False, index=idx), record_sleeve=True)
    pd.testing.assert_series_equal(wide.returns, quiet.returns)
    pd.testing.assert_series_equal(wide.turnover, quiet.turnover)
    assert quiet.sleeve_weight is None
    assert recorded.sleeve_weight is not None
    pd.testing.assert_series_equal(recorded.returns, quiet.returns)
    with pytest.raises(ValueError):
        run_symmetric_drift(**common, band=-0.01)
    with pytest.raises(ValueError):
        run_drift_band(**common, band=0.02, sides="both")


def test_mix_flag_does_not_use_a_later_close():
    idx = pd.bdate_range("2020-01-02", periods=5)
    spy = pd.Series(100.0, index=idx)
    upro_o = pd.Series([100.0, 100.0, 60.0, 60.0, 60.0], index=idx)
    upro_c = pd.Series([100.0, 60.0, 60.0, 60.0, 20.0], index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    full, full_mix = run_symmetric_drift(**_args(spy, upro_o, upro_c, trend, dip, rf), band=0.05)
    prefix_args = _args(
        spy.iloc[:4], upro_o.iloc[:4], upro_c.iloc[:4], trend.iloc[:4], dip.iloc[:4], rf.iloc[:4],
    )
    prefix, prefix_mix = run_symmetric_drift(**prefix_args, band=0.05)
    pd.testing.assert_series_equal(prefix.returns, full.returns.iloc[:4])
    pd.testing.assert_series_equal(prefix_mix, full_mix.iloc[:4])
    later = upro_c.copy()
    later.iloc[-1] = 60.0
    again, again_mix = run_symmetric_drift(**_args(spy, upro_o, later, trend, dip, rf), band=0.05)
    pd.testing.assert_series_equal(again_mix.iloc[:4], full_mix.iloc[:4])
    pd.testing.assert_series_equal(again.returns.iloc[:4], full.returns.iloc[:4])
    assert again_mix.iloc[-1] != full_mix.iloc[-1]


def test_signal_uses_the_band_instead_of_the_quarter_date():
    idx = pd.bdate_range("2020-03-20", "2020-04-03")
    close = pd.Series(100.0, index=idx)
    trend = pd.Series(True, index=idx)
    quarter = pd.Timestamp("2020-03-31")
    assert bool(rebalance_mask(idx, "quarterly").loc[quarter])
    spec = TrendCoreSpec(window=20)
    plain = build_trend_signal(spec, close, as_of=quarter, apply_dip=False, trend_override=trend)
    assert plain["rebalance_mix"] is True
    assert plain["trade_reason"] == "scheduled_rebalance"
    info = {"sym_rule": "zsym_b02", "sym_calendar": False, "sym_band": 0.02}
    blocked = pd.Series(False, index=idx)
    held = build_trend_signal(
        spec, close, as_of=quarter, apply_dip=False, trend_override=trend,
        sym_mix=blocked, sym_info=info,
    )
    assert held["rebalance_mix"] is False
    assert held["trade"] is False
    assert held["diagnostics"]["sym_mix"] is False
    fired = blocked.copy()
    fired.loc[quarter] = True
    trimmed = build_trend_signal(
        spec, close, as_of=quarter, apply_dip=False, trend_override=trend,
        sym_mix=fired, sym_info=info,
    )
    assert trimmed["rebalance_mix"] is True
    assert trimmed["trade_reason"] == "sym_drift"
    calendar = build_trend_signal(
        spec, close, as_of=quarter, apply_dip=False, trend_override=trend,
        sym_mix=blocked,
        sym_info={"sym_rule": "sym_live", "sym_calendar": True, "sym_band": None},
    )
    assert calendar["rebalance_mix"] is True
    assert calendar["trade_reason"] == "scheduled_rebalance"
    with pytest.raises(ValueError):
        build_trend_signal(
            spec, close, as_of=quarter, apply_dip=False, trend_override=trend,
            drift_mix=fired, drift_info={"drift_rule": "drift_b02", "drift_calendar": False, "drift_band": 0.02},
            sym_mix=fired, sym_info=info,
        )


def _book(last_upro: float):
    idx = pd.bdate_range("2016-01-04", periods=250)
    spy_c = pd.Series(100.0, index=idx)
    spy_c.iloc[200:] = 120.0
    spy_o = spy_c.shift(1)
    spy_o.iloc[0] = spy_c.iloc[0]
    upro_c = spy_c.copy()
    upro_o = spy_o.copy()
    upro_c.iloc[-2] = last_upro
    upro_o.iloc[-1] = last_upro
    upro_c.iloc[-1] = last_upro
    rf = pd.Series(0.0, index=idx)
    return idx, spy_o, spy_c, upro_o, upro_c, rf


def test_kept_band_replays_only_the_selected_cell(tmp_path):
    idx, spy_o, spy_c, upro_o, upro_c, rf = _book(40.0)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False, "status": "tested", "overlay": "sym_drift",
    }]))
    assert kept_sym_drift(path, spy_o, spy_c, spy_o, spy_c, upro_o, upro_c, rf) == (None, None)
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "sym_drift",
        "walk_forward_picks": [
            [str(idx[0].date()), "zsym_b02"],
            [str(idx[-1].date()), "sym_live"],
        ],
        "variants": [
            {"name": "zsym_b02", "band": 0.02},
            {"name": "sym_live", "band": None},
        ],
    }]))
    trend = trend_state(spy_c, 200, 0.02, "daily")
    dip = pd.Series(False, index=idx)
    _direct, direct_mix = run_symmetric_drift(
        spy_o, spy_c, spy_o, spy_c, upro_o, upro_c,
        trend, dip, 0.02, 0.5, 0.5, rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    _one, one_mix = run_drift_band(
        spy_o, spy_c, spy_o, spy_c, upro_o, upro_c,
        trend, dip, 0.02, 0.5, 0.5, rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    assert bool(direct_mix.iloc[-2]) is True
    assert bool(one_mix.iloc[-2]) is False
    series, info = kept_sym_drift(
        path, spy_o, spy_c, spy_o, spy_c, upro_o, upro_c, rf, as_of=idx[-2],
    )
    assert info["sym_rule"] == "zsym_b02"
    assert info["sym_calendar"] is False
    pd.testing.assert_series_equal(series, direct_mix.iloc[:-1])
    calendar, calendar_info = kept_sym_drift(
        path, spy_o, spy_c, spy_o, spy_c, upro_o, upro_c, rf, as_of=idx[-1],
    )
    assert calendar is None
    assert calendar_info["sym_rule"] == "sym_live"
    assert calendar_info["sym_calendar"] is True
    bumped = upro_c.copy()
    bumped.iloc[-1] = float(bumped.iloc[-1]) * 3.0
    again, _info = kept_sym_drift(
        path, spy_o, spy_c, spy_o, spy_c, upro_o, bumped, rf, as_of=idx[-2],
    )
    pd.testing.assert_series_equal(again, series)
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "sym_drift",
        "walk_forward_picks": [["2016-01-04", "zsym_b50"]],
        "variants": [{"name": "zsym_b50", "band": 0.50}],
    }]))
    with pytest.raises(DataError):
        kept_sym_drift(path, spy_o, spy_c, spy_o, spy_c, upro_o, upro_c, rf)
