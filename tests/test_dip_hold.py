"""Dip-buy exit. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.diphold import (
    EXITS,
    HOLDS,
    LIVE_EXIT,
    LIVE_HOLD,
    OFF_NAME,
    cells,
    dip_hold_flag,
    episode_lengths,
    is_live_cell,
    kept_dip_hold,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _decline():
    idx = pd.bdate_range("2020-01-02", periods=8)
    close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0], index=idx)
    return idx, close


def test_grid_is_fixed_and_the_live_exit_is_not_a_new_trial():
    found = cells()
    assert found == [
        ("down", 3), ("down", 5), ("down", 10),
        ("fixed", 3), ("fixed", 5), ("fixed", 10),
        ("up", 3), ("up", 10),
    ]
    assert EXITS == ("down", "fixed", "up")
    assert HOLDS == (3, 5, 10)
    assert OFF_NAME == "hold_live"
    assert is_live_cell(LIVE_EXIT, LIVE_HOLD)
    assert not any(is_live_cell(*cell) for cell in found)
    names = [variant_name(*cell) for cell in found]
    assert names == sorted(names)
    assert OFF_NAME < names[0]
    assert variant_name("down", 3) == "zhold_down_n03"
    assert variant_name("up", 10) == "zhold_up_n10"
    assert len(set(names)) == 8
    for cell in found:
        neighbors = adjacent_cells(cell, [EXITS, HOLDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found or is_live_cell(*neighbor)


def test_live_exit_matches_the_dip_and_uses_no_future_close():
    idx = pd.bdate_range("2018-01-02", periods=40)
    rng = np.random.default_rng(11)
    steps = rng.normal(0.0, 0.01, len(idx))
    close = pd.Series(100.0 * np.cumprod(1.0 + steps), index=idx)
    live = dip_active(close, "3_down_closes", 5)
    flag = dip_hold_flag(close, "up", 5)
    pd.testing.assert_series_equal(flag.astype(bool).rename("dip_on"), live.astype(bool), check_names=False)
    for end in range(5, len(close)):
        cut = dip_hold_flag(close.iloc[: end + 1], "down", 5)
        full = dip_hold_flag(close, "down", 5)
        pd.testing.assert_series_equal(cut, full.iloc[: end + 1])


def test_up_exit_holds_a_decline_and_down_exit_leaves_without_reentering():
    _idx, close = _decline()
    # Third down close is index 3. The up exit stays through the rest of the fall.
    up = dip_hold_flag(close, "up", 5)
    assert list(up.astype(bool)) == [False, False, False, True, True, True, True, True]
    # The down exit leaves on the next down close and does not re-enter that close,
    # even though three down closes are still true. The following close is a new entry.
    down = dip_hold_flag(close, "down", 5)
    assert list(down.astype(bool)) == [False, False, False, True, False, True, False, True]
    assert episode_lengths(down) == [1, 1, 1]
    assert episode_lengths(up) == [5]


def test_fixed_hold_ignores_an_up_close_and_stops_at_the_cap():
    idx = pd.bdate_range("2020-01-02", periods=10)
    # Three down closes, then a bounce, then a flat stretch.
    close = pd.Series([100, 99, 98, 97, 99, 100, 100, 100, 100, 100], index=idx, dtype=float)
    up = dip_hold_flag(close, "up", 5)
    fixed = dip_hold_flag(close, "fixed", 3)
    # Up exit sells the first green close. Fixed holds three sessions anyway.
    assert bool(up.iloc[3]) and not bool(up.iloc[4])
    assert list(fixed.astype(bool))[:7] == [False, False, False, True, True, True, False]
    assert episode_lengths(fixed) == [3]


def test_flat_close_does_not_end_either_signed_exit():
    idx = pd.bdate_range("2020-01-02", periods=7)
    close = pd.Series([100, 99, 98, 97, 97, 97, 96], index=idx, dtype=float)
    up = dip_hold_flag(close, "up", 5)
    down = dip_hold_flag(close, "down", 5)
    # The flat sessions are not an up close and not a down close, so both stay.
    assert bool(up.iloc[3]) and bool(up.iloc[4]) and bool(up.iloc[5])
    assert bool(down.iloc[3]) and bool(down.iloc[4]) and bool(down.iloc[5])
    # The next down close ends only the down exit.
    assert bool(up.iloc[6])
    assert not bool(down.iloc[6])


def test_unknown_exit_and_a_non_positive_cap_are_rejected():
    close = pd.Series([10.0, 9.0, 8.0], index=pd.bdate_range("2020-01-02", periods=3))
    with pytest.raises(ValueError, match="unknown dip exit"):
        dip_hold_flag(close, "sideways", 5)
    with pytest.raises(ValueError, match="max_hold"):
        dip_hold_flag(close, "fixed", 0)


def test_rejected_rule_does_not_load_and_a_kept_rule_ignores_later_picks(tmp_path):
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "dip_hold",
        "variants": [{"name": "zhold_down_n03", "exit": "down", "max_hold": 3}],
        "walk_forward_picks": [["2020-01-02", "zhold_down_n03"]],
    }]))
    idx, close = _decline()
    assert kept_dip_hold(path, close) == (None, None)
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_hold",
        "variants": [
            {"name": "hold_live", "exit": "up", "max_hold": 5},
            {"name": "zhold_down_n03", "exit": "down", "max_hold": 3},
        ],
        "walk_forward_picks": [
            ["2020-01-02", "hold_live"],
            ["2020-01-09", "zhold_down_n03"],
        ],
    }]))
    early, early_info = kept_dip_hold(path, close, as_of=idx[4])
    assert early_info["dip_hold_live"] is True
    assert early_info["dip_hold_rule"] == "hold_live"
    pd.testing.assert_series_equal(
        early, dip_hold_flag(close.loc[: idx[4]], "up", 5), check_names=False,
    )
    full, info = kept_dip_hold(path, close)
    assert info["dip_hold_live"] is False
    assert info["dip_hold_exit"] == "down"
    assert info["dip_hold_max"] == 3
    assert info["dip_hold_rule"] == "zhold_down_n03"
    # The switch date is 2020-01-09, which is index 5 of this calendar.
    assert list(full.astype(bool).iloc[:5]) == list(dip_hold_flag(close, "up", 5).astype(bool).iloc[:5])
    assert bool(full.iloc[-1]) == bool(dip_hold_flag(close, "down", 3).iloc[-1])
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_hold",
        "variants": [{"name": "zhold_down_n03", "exit": "sideways", "max_hold": 3}],
        "walk_forward_picks": [["2020-01-02", "zhold_down_n03"]],
    }]))
    with pytest.raises(DataError):
        kept_dip_hold(broken, close)


def test_signal_uses_the_hold_flag_and_refuses_it_beside_a_sized_dip():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
    flag = pd.Series([False, False, False, True, False, True], index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(), close, dip_hold_flag=flag,
        dip_hold_info={"dip_hold_rule": "zhold_down_n03", "dip_hold_live": False},
    )
    assert payload["weights"].get("SSO") == pytest.approx(0.5)
    assert "SPY" not in payload["weights"]
    assert payload["diagnostics"]["dip_on"] is True
    assert payload["diagnostics"]["dip_hold_rule"] == "zhold_down_n03"
    plain = build_trend_signal(TrendCoreSpec(), close)
    assert plain["diagnostics"]["dip_on"] is True
    assert "dip_hold_rule" not in plain["diagnostics"]
    with pytest.raises(ValueError, match="sized dip"):
        build_trend_signal(
            TrendCoreSpec(), close, dip_boost=flag.astype(float), dip_hold_flag=flag,
        )
