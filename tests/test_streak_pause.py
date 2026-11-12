"""Down-streak pause on the UPRO sleeve. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.pause import (
    BAND,
    FLOORS,
    MAX_HOLD,
    OFF_NAME,
    STREAKS,
    WINDOW,
    cells,
    kept_streak_pause,
    pause_share,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def _prices(values) -> pd.Series:
    idx = pd.bdate_range("2018-01-02", periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == (2, 0.0)
    assert found[-1] == (4, 0.5)
    assert len(found) == 6
    assert STREAKS == (2, 3, 4)
    assert FLOORS == (0.0, 0.5)
    assert MAX_HOLD == 5
    assert WINDOW == 200
    assert BAND == 0.02
    assert OFF_NAME == "pause_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 6
    assert variant_name(2, 0.0) == "pause_k02_f00"
    assert variant_name(3, 0.5) == "pause_k03_f50"
    assert variant_name(4, 0.0) == "pause_k04_f00"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [STREAKS, FLOORS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_pause_uses_only_that_close_and_exits_on_the_first_up_close():
    # Two flat sessions, then three down closes, then an up close.
    close = _prices([100, 100, 99, 98, 97, 98, 98])
    trend = pd.Series(True, index=close.index)
    share = pause_share(close, trend, 3, 0.0)
    assert list(share.round(6)) == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    half = pause_share(close, trend, 3, 0.5)
    assert list(half.round(6)) == [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0]
    # A two-day streak arms one close earlier and is a different clock.
    short = pause_share(close, trend, 2, 0.0)
    assert bool(short.iloc[3] < 1.0)
    assert bool(share.iloc[3] == 1.0)
    future = close.copy()
    future.iloc[5] = 90.0
    again = pause_share(future, trend, 3, 0.0)
    pd.testing.assert_series_equal(share.iloc[:5], again.iloc[:5])
    # That close is now a down close, so the pause does not exit there.
    assert float(again.iloc[5]) == 0.0
    with pytest.raises(ValueError):
        pause_share(close, trend, 1, 0.0)
    with pytest.raises(ValueError):
        pause_share(close, trend, 3, 1.0)
    with pytest.raises(ValueError):
        pause_share(close, trend, 3, 0.0, max_hold=0)


def test_trend_off_does_not_arm_and_flattens_a_pause():
    close = _prices([100, 99, 98, 97, 96, 95])
    on = pd.Series(True, index=close.index)
    armed = pause_share(close, on, 3, 0.0)
    assert float(armed.iloc[3]) == 0.0
    off = pd.Series(False, index=close.index)
    assert bool(pause_share(close, off, 2, 0.0).eq(1.0).all())
    mixed = on.copy()
    mixed.iloc[4:] = False
    flattened = pause_share(close, mixed, 3, 0.0)
    assert float(flattened.iloc[3]) == 0.0
    assert float(flattened.iloc[4]) == 1.0
    assert float(flattened.iloc[5]) == 1.0
    # A later down close while the band is off does not re-arm.
    assert float(flattened.iloc[-1]) == 1.0


def test_five_session_cap_exits_while_the_market_is_still_falling():
    close = _prices([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    trend = pd.Series(True, index=close.index)
    share = pause_share(close, trend, 3, 0.0)
    # Armed at the third down close (index 3). Five decided sessions, then out.
    # The next close is a fresh three-down signal, so the clock arms again.
    assert list(share.iloc[3:8].round(6)) == [0, 0, 0, 0, 0]
    assert float(share.iloc[8]) == 1.0
    assert float(share.iloc[9]) == 0.0


def test_three_day_pause_matches_the_dip_clock_when_the_trend_stays_on():
    rng = np.random.default_rng(7)
    steps = rng.normal(0.0, 1.0, 80)
    close = _prices(100 + np.cumsum(steps))
    trend = pd.Series(True, index=close.index)
    share = pause_share(close, trend, 3, 0.0)
    dip = dip_active(close, "3_down_closes", MAX_HOLD)
    pd.testing.assert_series_equal(share.lt(1.0), dip, check_names=False)
    half = pause_share(close, trend, 3, 0.5)
    assert bool(half.lt(1.0).equals(share.lt(1.0)))


def test_signal_blend_stays_in_equities_and_a_rejected_rule_does_not_load(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    on = pd.Series(True, index=idx)
    share = pd.Series(1.0, index=idx)
    share.iloc[-1] = 0.5
    spec = TrendCoreSpec(window=20)
    blended = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, pause_share=share,
        pause_info={"pause_rule": "pause_k03_f50", "pause_live": False},
    )
    assert blended["weights"]["SPY"] == pytest.approx(0.75)
    assert blended["weights"]["UPRO"] == pytest.approx(0.25)
    assert "BIL" not in blended["weights"]
    assert blended["trade_reason"] == "streak_pause"
    assert blended["diagnostics"]["sleeve_leverage_applied"] == pytest.approx(2.0)
    assert blended["diagnostics"]["pause_share"] == pytest.approx(0.5)
    plain = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on,
    )
    full = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, pause_share=pd.Series(1.0, index=idx),
    )
    assert full["weights"] == plain["weights"]
    assert "pause_share" not in plain["diagnostics"]
    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, pause_share=pd.Series(0.0, index=idx),
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]
    dipped = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, pause_share=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]
    with pytest.raises(ValueError, match="cannot both be kept"):
        build_trend_signal(
            spec, close, generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, trend_override=on,
            bill_share=share, pause_share=share,
        )

    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "id": "2026-11-12-streak-pause",
        "kept": False,
        "status": "tested",
        "overlay": "streak_pause",
    }]))
    assert kept_streak_pause(path, close, as_of=idx[-1]) == (None, None)
    # A rising path keeps the live 200-day state on once the average exists.
    rising = pd.Series(np.linspace(100.0, 180.0, 260), index=pd.bdate_range("2014-01-02", periods=260))
    path.write_text(json.dumps([{
        "id": "2026-11-12-streak-pause",
        "kept": True,
        "status": "tested",
        "overlay": "streak_pause",
        "variants": [{
            "name": "pause_k02_f00",
            "streak": 2,
            "floor": 0.0,
        }],
        "walk_forward_picks": [["2014-01-02", "pause_off"], ["2014-06-02", "pause_k02_f00"]],
    }]))
    series, info = kept_streak_pause(path, rising, as_of=rising.index[-1])
    assert info["pause_rule"] == "pause_k02_f00"
    assert info["pause_streak"] == 2
    assert info["pause_floor"] == pytest.approx(0.0)
    assert info["pause_live"] is False
    trend = trend_state(rising, WINDOW, BAND, "daily")
    expected = pause_share(rising, trend, 2, 0.0)
    later = series.loc[pd.Timestamp("2014-06-02"):]
    pd.testing.assert_series_equal(later, expected.reindex(later.index), check_names=False)
    assert bool(series.loc[:pd.Timestamp("2014-05-30")].eq(1.0).all())
    earlier, early_info = kept_streak_pause(path, rising, as_of=pd.Timestamp("2014-05-30"))
    assert early_info["pause_live"] is True
    assert float(earlier.iloc[-1]) == 1.0
    leaked = rising.copy()
    leaked.iloc[-1] = leaked.iloc[-2] - 5.0
    truncated, _info = kept_streak_pause(path, leaked, as_of=rising.index[-5])
    assert truncated.index[-1] == rising.index[-5]
    untouched = kept_streak_pause(path, rising, as_of=rising.index[-5])[0]
    pd.testing.assert_series_equal(truncated, untouched)
