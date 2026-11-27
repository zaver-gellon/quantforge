"""Turn-of-month tilt on the UPRO sleeve. Synthetic prices and the NYSE calendar."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import CustomBusinessDay

from quantforge.calendar import is_nyse_session, next_nyse_session, nyse_holidays
from quantforge.data.base import DataError
from quantforge.engine.backtest import run_trend_core
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.turnofmonth import (
    BASE_CORE,
    BASE_SLEEVE,
    FIRSTS,
    LASTS,
    OFF_NAME,
    RAISED_CORE,
    RAISED_SLEEVE,
    cells,
    kept_turn_of_month,
    turn_of_month_flag,
    variant_name,
)


def test_grid_is_fixed_and_neighbors_are_one_step():
    assert LASTS == (1, 2, 3)
    assert FIRSTS == (1, 2, 3)
    assert BASE_CORE == 0.50 and BASE_SLEEVE == 0.50
    assert RAISED_CORE == 0.30 and RAISED_SLEEVE == 0.70
    assert abs(RAISED_CORE + RAISED_SLEEVE - 1.0) < 1e-12
    assert OFF_NAME == "tom_off"
    assert cells() == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    names = [variant_name(n_last, m_first) for n_last, m_first in cells()]
    assert names[0] == "ztom_n1_m1"
    assert OFF_NAME < min(names)
    assert set(adjacent_cells((1, 1), [list(LASTS), list(FIRSTS)])) == {(2, 1), (1, 2)}
    assert set(adjacent_cells((2, 2), [list(LASTS), list(FIRSTS)])) == {(1, 2), (3, 2), (2, 1), (2, 3)}
    assert set(adjacent_cells((3, 3), [list(LASTS), list(FIRSTS)])) == {(2, 3), (3, 2)}
    with pytest.raises(ValueError):
        variant_name(4, 1)
    with pytest.raises(ValueError):
        turn_of_month_flag(pd.DatetimeIndex([pd.Timestamp("2020-01-02")]), 0, 1)


def test_november_2026_half_day_is_a_session_and_the_window_uses_the_next_one():
    assert not is_nyse_session("2026-11-26")
    assert is_nyse_session("2026-11-27")
    assert next_nyse_session("2026-11-25") == pd.Timestamp("2026-11-27")
    assert next_nyse_session("2026-11-27") == pd.Timestamp("2026-11-30")
    assert next_nyse_session("2026-11-30") == pd.Timestamp("2026-12-01")
    idx = pd.bdate_range(
        "2026-11-02", "2026-11-30", freq=CustomBusinessDay(holidays=nyse_holidays()),
    )
    assert pd.Timestamp("2026-11-26") not in idx
    assert pd.Timestamp("2026-11-27") in idx
    assert idx[0] == pd.Timestamp("2026-11-02")
    assert idx[-1] == pd.Timestamp("2026-11-30")
    short = turn_of_month_flag(idx, 1, 1)
    wide = turn_of_month_flag(idx, 3, 1)
    # The decision at Nov 25 is for Nov 27. That session is inside the last three
    # and outside the last one. The decision at Nov 27 is for Nov 30, the last session.
    assert bool(short.loc[pd.Timestamp("2026-11-25")]) is False
    assert bool(wide.loc[pd.Timestamp("2026-11-25")]) is True
    assert bool(short.loc[pd.Timestamp("2026-11-27")]) is True
    assert bool(short.loc[pd.Timestamp("2026-11-30")]) is True
    # A price index that skips Nov 27 decides the Nov 25 close for Nov 30, not for Nov 27.
    skipped = turn_of_month_flag(pd.to_datetime(["2026-11-25", "2026-11-30"]), 1, 1)
    only = turn_of_month_flag(pd.to_datetime(["2026-11-25"]), 1, 1)
    assert bool(skipped.iloc[0]) is True
    assert bool(only.iloc[0]) is False


def test_flag_ignores_prices_and_does_not_use_a_later_bar_for_an_earlier_decision():
    idx = pd.bdate_range("2020-01-02", "2020-04-30")
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    bumped = close.copy()
    bumped.iloc[-1] = 50.0
    pd.testing.assert_series_equal(turn_of_month_flag(close.index, 2, 2), turn_of_month_flag(bumped.index, 2, 2))
    full = turn_of_month_flag(idx, 1, 3)
    for end in range(1, len(idx) - 1):
        prefix = turn_of_month_flag(idx[: end + 1], 1, 3)
        pd.testing.assert_series_equal(prefix.iloc[:-1], full.iloc[:end], check_names=False)


def _book(trend, tilt):
    idx = trend.index
    spy_c = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    spy_o = spy_c.shift(1).fillna(float(spy_c.iloc[0]))
    upro_c = pd.Series(np.linspace(40.0, 80.0, len(idx)), index=idx)
    upro_o = upro_c.shift(1).fillna(float(upro_c.iloc[0]))
    sso_c = spy_c * 1.2
    sso_o = spy_o * 1.2
    rf = pd.Series(0.0, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    return run_trend_core(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        trend, dip, mix, 0.5, 0.5, rf,
        tilt_on=tilt, tilt_core=RAISED_CORE, tilt_sleeve=RAISED_SLEEVE,
    )


def test_a_dark_window_matches_the_live_book_and_an_open_one_does_not():
    idx = pd.bdate_range("2019-01-02", periods=40)
    trend = pd.Series(True, index=idx)
    plain = run_trend_core(
        *_prices(idx), trend, pd.Series(False, index=idx), pd.Series(False, index=idx),
        0.5, 0.5, pd.Series(0.0, index=idx),
    )
    dark = _book(trend, pd.Series(False, index=idx))
    pd.testing.assert_series_equal(dark.returns, plain.returns)
    window = pd.Series(False, index=idx)
    window.iloc[10:16] = True
    lit = _book(trend, window)
    assert float((lit.returns - plain.returns).abs().max()) > 1e-6
    assert float(lit.turnover.sum()) > float(plain.turnover.sum())
    off = pd.Series(False, index=idx)
    held_off = _book(off, pd.Series(True, index=idx))
    base_off = _book(off, pd.Series(False, index=idx))
    pd.testing.assert_series_equal(held_off.returns, base_off.returns)


def _prices(idx):
    spy_c = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    spy_o = spy_c.shift(1).fillna(float(spy_c.iloc[0]))
    upro_c = pd.Series(np.linspace(40.0, 80.0, len(idx)), index=idx)
    upro_o = upro_c.shift(1).fillna(float(upro_c.iloc[0]))
    sso_c = spy_c * 1.2
    sso_o = spy_o * 1.2
    return spy_o, spy_c, sso_o, sso_c, upro_o, upro_c


def test_rejected_tilt_does_not_change_the_signal_and_a_kept_one_raises_upro(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.Series(np.linspace(100.0, 160.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "turn_of_month",
        "walk_forward_picks": [["2020-01-02", "ztom_n1_m3"]],
        "variants": [{"name": "ztom_n1_m3", "n_last": 1, "m_first": 3}],
    }]))
    assert kept_turn_of_month(path, close) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "turn_of_month",
        "walk_forward_picks": [
            ["2020-01-02", "ztom_n1_m1"],
            ["2020-02-03", "tom_off"],
        ],
        "variants": [
            {"name": "ztom_n1_m1", "n_last": 1, "m_first": 1},
            {"name": "tom_off", "n_last": None, "m_first": None},
        ],
    }]))
    assert kept_turn_of_month(path, close, as_of=idx[-1]) == (None, None)
    flag, info = kept_turn_of_month(path, close, as_of=pd.Timestamp("2020-01-31"))
    assert info["turn_rule"] == "ztom_n1_m1"
    assert info["turn_n_last"] == 1
    assert info["turn_sleeve_weight"] == 0.70
    assert info["turn_live"] is False
    expected = turn_of_month_flag(close.loc[: pd.Timestamp("2020-01-31")].index, 1, 1)
    pd.testing.assert_series_equal(flag, expected, check_names=False)

    window = pd.Series(False, index=idx)
    window.iloc[-1] = True
    on = pd.Series(True, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(window=5), close, trend_override=on, turn_on=window, turn_info=info,
    )
    assert payload["weights"]["UPRO"] == pytest.approx(0.70)
    assert payload["weights"]["SPY"] == pytest.approx(0.30)
    assert payload["diagnostics"]["turn_on"] is True
    assert payload["diagnostics"]["turn_window"] is True
    assert payload["diagnostics"]["core_weight"] == pytest.approx(0.30)
    assert payload["trade_reason"] == "turn_of_month"
    off = build_trend_signal(
        TrendCoreSpec(window=5), close, trend_override=pd.Series(False, index=idx),
        turn_on=window, turn_info=info,
    )
    assert off["weights"].get("SPY") == pytest.approx(0.5)
    assert off["weights"].get("BIL") == pytest.approx(0.5)
    assert "UPRO" not in off["weights"]
    assert off["diagnostics"]["turn_on"] is False
    assert off["diagnostics"]["turn_window"] is True
    assert off["diagnostics"]["trend_on"] is False
    plain = build_trend_signal(TrendCoreSpec(window=5), close, trend_override=on)
    assert "turn_on" not in plain["diagnostics"]
    assert plain["weights"].get("UPRO") == pytest.approx(0.5)

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "turn_of_month",
        "variants": [{"name": "ztom_n1_m1", "n_last": 4, "m_first": 1}],
        "walk_forward_picks": [["2020-01-02", "ztom_n1_m1"]],
    }]))
    with pytest.raises(DataError):
        kept_turn_of_month(broken, close)
