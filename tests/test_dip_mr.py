"""RSI(2) trigger for the SPY-half dip. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.dipmr import (
    EXIT_RSI,
    EXIT_SMA,
    OFF_NAME,
    THRESHOLDS,
    cells,
    dip_mr_flag,
    kept_dip_mr,
    live_dip_flag,
    variant_name,
)
from quantforge.strategies.meanrev import wilder_rsi
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def test_grid_is_fixed_and_neighbors_are_one_threshold_step():
    assert THRESHOLDS == (5, 10, 15)
    assert EXIT_RSI == 70.0
    assert EXIT_SMA == 5
    assert OFF_NAME == "dmr_off"
    assert cells() == [5, 10, 15]
    names = [variant_name(level) for level in cells()]
    assert names == ["zdmr_rsi05", "zdmr_rsi10", "zdmr_rsi15"]
    assert OFF_NAME < min(names)
    assert set(adjacent_cells((5,), [list(THRESHOLDS)])) == {(10,)}
    assert set(adjacent_cells((10,), [list(THRESHOLDS)])) == {(5,), (15,)}
    assert set(adjacent_cells((15,), [list(THRESHOLDS)])) == {(10,)}
    with pytest.raises(ValueError):
        variant_name(20)
    with pytest.raises(ValueError):
        dip_mr_flag(pd.Series([1.0], index=pd.bdate_range("2020-01-02", periods=1)), 20)


def test_flag_does_not_read_a_later_close():
    idx = pd.bdate_range("2019-01-02", periods=40)
    close = pd.Series(100.0, index=idx)
    close.iloc[20:] = [90, 80, 70, 71, 72, 60, 110] + [100.0] * 13
    full = dip_mr_flag(close, 10)
    for end in range(5, len(close)):
        prefix = dip_mr_flag(close.iloc[: end + 1], 10)
        assert bool(prefix.iloc[-1]) == bool(full.iloc[end])
    bumped = close.copy()
    bumped.iloc[-1] = 40.0
    unchanged = dip_mr_flag(bumped.iloc[:-1], 10)
    pd.testing.assert_series_equal(unchanged, full.iloc[:-1], check_names=False)


def test_one_down_close_enters_without_three_down_closes():
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.Series(100.0, index=idx)
    close.iloc[-1] = 80.0
    flag = dip_mr_flag(close, 5)
    live = live_dip_flag(close)
    assert float(wilder_rsi(close, 2).iloc[-1]) < 5
    assert bool(flag.iloc[-1]) is True
    assert bool(live.iloc[-1]) is False


def test_small_up_close_stays_on_while_the_live_dip_exits():
    idx = pd.bdate_range("2020-01-02", periods=12)
    close = pd.Series([100.0] * 8 + [90.0, 80.0, 70.0, 71.0], index=idx)
    flag = dip_mr_flag(close, 15)
    live = live_dip_flag(close)
    rsi = wilder_rsi(close, 2)
    average = close.rolling(5).mean()
    assert bool(flag.iloc[-2]) is True
    assert bool(live.iloc[-2]) is True
    assert float(rsi.iloc[-1]) <= 70.0
    assert float(close.iloc[-1]) <= float(average.iloc[-1])
    assert bool(flag.iloc[-1]) is True
    assert bool(live.iloc[-1]) is False


def test_close_above_the_five_day_average_exits_and_does_not_reenter():
    idx = pd.bdate_range("2020-01-02", periods=10)
    # The last close is above the 5-day average. RSI(2) is also hot, so both
    # exit clauses are true, and the entry clause is false. The flag is off.
    close = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0, 61.0, 62.0, 80.0, 90.0, 100.0], index=idx)
    flag = dip_mr_flag(close, 15)
    average = close.rolling(5).mean()
    rsi = wilder_rsi(close, 2)
    assert float(close.iloc[-1]) > float(average.iloc[-1])
    assert float(rsi.iloc[-1]) > 70.0
    assert bool(flag.iloc[-1]) is False
    # The crash close, before the rebound, is an entry.
    assert bool(flag.iloc[4]) is True


def test_entry_thresholds_are_nested_and_the_holds_differ():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-04", periods=400)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(idx))), index=idx)
    rsi = wilder_rsi(close, 2)
    below5 = (rsi < 5).fillna(False)
    below10 = (rsi < 10).fillna(False)
    below15 = (rsi < 15).fillna(False)
    assert bool((below5 & ~below10).any()) is False
    assert bool((below10 & ~below15).any()) is False
    assert int((below15 & ~below5).sum()) > 0
    assert int((dip_mr_flag(close, 5) != dip_mr_flag(close, 15)).sum()) > 0


def test_rejected_trigger_does_not_change_the_signal_and_a_kept_one_does(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=40)
    close = pd.Series(100.0, index=idx)
    close.iloc[-1] = 80.0
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "dip_mr",
        "walk_forward_picks": [["2020-01-02", "zdmr_rsi05"]],
        "variants": [{"name": "zdmr_rsi05", "threshold": 5}],
    }]))
    assert kept_dip_mr(path, close) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_mr",
        "walk_forward_picks": [
            ["2020-01-02", "zdmr_rsi10"],
            ["2020-02-03", "dmr_off"],
        ],
        "variants": [
            {"name": "zdmr_rsi10", "threshold": 10},
            {"name": "dmr_off", "threshold": None},
        ],
    }]))
    assert kept_dip_mr(path, close, as_of=idx[-1]) == (None, None)
    flag, info = kept_dip_mr(path, close, as_of=pd.Timestamp("2020-01-31"))
    assert info["dip_mr_rule"] == "zdmr_rsi10"
    assert info["dip_mr_threshold"] == 10
    assert info["dip_mr_live"] is False
    expected = dip_mr_flag(close.loc[: pd.Timestamp("2020-01-31")], 10)
    pd.testing.assert_series_equal(flag, expected, check_names=False)
    bumped = close.copy()
    bumped.iloc[-1] = 50.0
    again, again_info = kept_dip_mr(path, bumped, as_of=pd.Timestamp("2020-01-31"))
    pd.testing.assert_series_equal(again, flag)
    assert again_info["dip_mr_rule"] == info["dip_mr_rule"]

    payload = build_trend_signal(
        TrendCoreSpec(window=20), close.loc[: pd.Timestamp("2020-01-31")],
        generated_at="2026-01-01T00:00:00Z",
        dip_mr_flag=flag, dip_mr_info=info,
    )
    assert payload["diagnostics"]["dip_rule"] == "zdmr_rsi10"
    assert payload["diagnostics"]["dip_mr_threshold"] == 10
    plain = build_trend_signal(
        TrendCoreSpec(window=20), close.loc[: pd.Timestamp("2020-01-31")],
        generated_at="2026-01-01T00:00:00Z",
    )
    assert plain["diagnostics"]["dip_rule"] == "3_down_closes"
    assert "dip_mr_rule" not in plain["diagnostics"]

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_mr",
        "variants": [{"name": "zdmr_rsi05", "threshold": 7}],
        "walk_forward_picks": [["2020-01-02", "zdmr_rsi05"]],
    }]))
    with pytest.raises(DataError):
        kept_dip_mr(broken, close)


def test_signal_uses_the_flag_and_refuses_it_beside_another_dip_rule():
    idx = pd.bdate_range("2021-01-04", periods=6)
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
    flag = pd.Series([False, False, False, True, False, True], index=idx)
    info = {"dip_mr_rule": "zdmr_rsi05", "dip_mr_threshold": 5, "dip_mr_live": False}
    payload = build_trend_signal(TrendCoreSpec(), close, dip_mr_flag=flag, dip_mr_info=info)
    assert payload["weights"].get("SSO") == pytest.approx(0.5)
    assert "SPY" not in payload["weights"]
    assert payload["diagnostics"]["dip_on"] is True
    assert payload["diagnostics"]["dip_rule"] == "zdmr_rsi05"
    with pytest.raises(ValueError, match="sized dip"):
        build_trend_signal(
            TrendCoreSpec(), close, dip_boost=flag.astype(float), dip_mr_flag=flag, dip_mr_info=info,
        )
    with pytest.raises(ValueError, match="dip-hold"):
        build_trend_signal(TrendCoreSpec(), close, dip_hold_flag=flag, dip_mr_flag=flag)
    with pytest.raises(ValueError, match="dip cut"):
        build_trend_signal(
            TrendCoreSpec(), close, dip_mr_flag=flag, dip_cut_info={"dip_cut_rule": "x"},
        )
