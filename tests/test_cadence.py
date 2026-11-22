"""Evaluation cadence of the 200-day band. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.schedule import week_ends
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.cadence import (
    BAND,
    FREQS,
    OFF_NAME,
    WINDOW,
    cadence_state,
    cells,
    evaluation_mask,
    kept_eval_cadence,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def test_grid_is_fixed_and_each_cadence_has_the_other_as_a_neighbor():
    assert FREQS == ("weekly", "monthly")
    assert WINDOW == 200
    assert BAND == 0.02
    assert OFF_NAME == "cad_off"
    assert cells() == ["weekly", "monthly"]
    names = [variant_name(freq) for freq in cells()]
    assert names == ["zcad_weekly", "zcad_monthly"]
    assert OFF_NAME < min(names)
    assert set(adjacent_cells(("weekly",), [list(FREQS)])) == {("monthly",)}
    assert set(adjacent_cells(("monthly",), [list(FREQS)])) == {("weekly",)}
    with pytest.raises(ValueError):
        variant_name("daily")
    with pytest.raises(ValueError):
        cadence_state(pd.Series([1.0], index=pd.bdate_range("2020-01-02", periods=1)), "daily")
    spec = TrendCoreSpec(freq="weekly")
    assert spec.freq == "weekly"
    assert "weekly" in spec.name()
    with pytest.raises(ValueError):
        TrendCoreSpec(freq="yearly")


def test_weekly_ignores_a_midweek_break_and_does_not_read_a_later_close():
    idx = pd.bdate_range("2017-06-01", "2018-03-30")
    close = pd.Series(np.linspace(80.0, 140.0, len(idx)), index=idx)
    crash_day = pd.Timestamp("2018-03-14")  # Wednesday
    friday = pd.Timestamp("2018-03-16")
    assert crash_day in close.index and friday in close.index
    assert not bool(week_ends(idx).loc[crash_day])
    assert bool(week_ends(idx).loc[friday])
    crashed = close.copy()
    crashed.loc[crash_day:] = 10.0
    daily = trend_state(crashed, 20, 0.0, "daily")
    weekly = trend_state(crashed, 20, 0.0, "weekly")
    assert bool(daily.loc[crash_day]) is False
    assert bool(weekly.loc[crash_day]) is True
    assert bool(weekly.loc[friday]) is False
    # A later close does not change the state through that Friday.
    prefix = trend_state(crashed.loc[:friday], 20, 0.0, "weekly")
    pd.testing.assert_series_equal(prefix, weekly.loc[:friday])
    mask = evaluation_mask(idx, "weekly")
    changed = weekly.astype(bool).ne(weekly.astype(bool).shift(1, fill_value=False))
    assert not bool((changed & ~mask).any())


def test_cadence_state_is_the_live_band_on_that_calendar():
    idx = pd.bdate_range("2016-01-04", periods=400)
    close = pd.Series(np.linspace(80.0, 160.0, len(idx)), index=idx)
    close.iloc[350:] = float(close.iloc[349]) * 0.5
    for freq in cells():
        got = cadence_state(close, freq)
        expected = trend_state(close, WINDOW, BAND, freq)
        pd.testing.assert_series_equal(got, expected, check_names=False)
        changed = got.astype(bool).ne(got.astype(bool).shift(1, fill_value=False))
        mask = evaluation_mask(idx, freq)
        assert not bool((changed & ~mask).any())


def test_monthly_still_waits_for_month_end_when_the_week_has_ended():
    idx = pd.bdate_range("2019-04-01", "2019-06-28")
    level = pd.Series(np.linspace(80.0, 120.0, len(idx)), index=idx)
    friday = pd.Timestamp("2019-06-14")
    month_end = pd.Timestamp("2019-06-28")
    level.loc[friday:] = 10.0
    week = trend_state(level, 20, 0.0, "weekly")
    month = trend_state(level, 20, 0.0, "monthly")
    assert bool(week.loc[friday]) is False
    assert bool(month.loc[friday]) is True
    assert bool(month.loc[month_end]) is False


def test_rejected_cadence_does_not_change_the_signal_and_a_kept_one_does(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=180)
    close = pd.Series(np.linspace(100.0, 160.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "eval_cadence",
        "walk_forward_picks": [["2018-01-02", "zcad_weekly"]],
        "variants": [{"name": "zcad_weekly", "freq": "weekly"}],
    }]))
    assert kept_eval_cadence(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "eval_cadence",
        "walk_forward_picks": [
            ["2018-01-02", "zcad_monthly"],
            ["2018-06-01", "cad_off"],
        ],
        "variants": [
            {"name": "zcad_monthly", "freq": "monthly"},
            {"name": "cad_off", "freq": None},
        ],
    }]))
    assert kept_eval_cadence(path, close, as_of=idx[-1]) == (None, None)
    state, info = kept_eval_cadence(path, close, as_of=pd.Timestamp("2018-04-02"))
    assert info["cadence_rule"] == "zcad_monthly"
    assert info["cadence"] == "monthly"
    assert info["cadence_window"] == 200
    assert info["cadence_band"] == 0.02
    assert state.index[-1] <= pd.Timestamp("2018-04-02")
    bumped = close.copy()
    bumped.iloc[-1] = 20.0
    unchanged, unchanged_info = kept_eval_cadence(path, bumped, as_of=pd.Timestamp("2018-04-02"))
    pd.testing.assert_series_equal(unchanged, state)
    assert unchanged_info["cadence_on"] == info["cadence_on"]

    # The history through April is shorter than 200 sessions, so the kept state stays off.
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=state.index[-1],
        generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=state, cadence_info=info,
    )
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    assert payload["diagnostics"]["freq"] == "monthly"
    assert payload["diagnostics"]["cadence_rule"] == "zcad_monthly"
    assert payload["diagnostics"]["trend_on"] is False

    rising = pd.Series(np.linspace(80.0, 180.0, len(idx)), index=idx)
    on_state = trend_state(rising, 20, 0.0, "weekly")
    assert bool(on_state.iloc[-1])
    held = build_trend_signal(
        TrendCoreSpec(window=20), rising, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on_state,
        cadence_info={"cadence_rule": "zcad_weekly", "cadence": "weekly", "cadence_on": True},
    )
    assert held["weights"]["UPRO"] == pytest.approx(0.5)
    assert held["diagnostics"]["freq"] == "weekly"
