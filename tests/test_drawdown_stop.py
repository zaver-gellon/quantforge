"""Trailing-peak sleeve stop. No market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.drawdown import (
    CUTS,
    OFF_NAME,
    REENTRY_FRACTION,
    WINDOWS,
    cells,
    kept_drawdown_scale,
    sleeve_allowed,
    sleeve_scale,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def _close(n=120, seed=5):
    idx = pd.bdate_range("2018-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)
    return idx, close


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert len(found) == 9
    assert len({variant_name(*cell) for cell in found}) == 9
    assert WINDOWS == (63, 126, 252)
    assert CUTS == (0.08, 0.12, 0.16)
    assert REENTRY_FRACTION == 0.5
    assert OFF_NAME == "dd_off"
    for cell in found:
        neighbors = adjacent_cells(cell, [WINDOWS, CUTS])
        assert neighbors
        assert cell not in neighbors


def test_stop_does_not_read_a_future_close():
    idx, close = _close(90)
    base = sleeve_allowed(close, 20, 0.08)
    future = close.copy()
    future.iloc[50:] = close.iloc[50] * 0.4
    again = sleeve_allowed(future, 20, 0.08)
    pd.testing.assert_series_equal(base.iloc[:50], again.iloc[:50])


def test_incomplete_window_leaves_the_sleeve_on():
    _idx, close = _close(40)
    allowed = sleeve_allowed(close, 30, 0.08)
    assert bool(allowed.iloc[:29].all())


def test_stop_cuts_a_fast_break_while_the_slow_average_is_still_on():
    idx = pd.bdate_range("2010-01-04", periods=420)
    flat = np.full(300, 100.0)
    high = np.full(100, 150.0)
    # Five sessions, 12% below the trailing high. The 200-day average is still
    # far under the close, so the live band would keep the sleeve on.
    drop = np.linspace(150.0, 132.0, 20)
    close = pd.Series(np.concatenate([flat, high, drop]), index=idx)
    trend = trend_state(close, 200, 0.02, "daily")
    allowed = sleeve_allowed(close, 63, 0.10)
    assert bool(trend.iloc[-1])
    assert not bool(allowed.iloc[-1])
    assert bool((trend & ~allowed).any())


def test_reentry_waits_until_half_the_cut_has_healed():
    idx = pd.bdate_range("2014-01-02", periods=40)
    # Window 10. A drop through 10%, then a hold at 7% (still past the 5% heal),
    # then a recovery to 4% drawdown.
    peak = np.full(15, 100.0)
    down = np.array([90.0, 89.0])
    held = np.full(5, 93.0)
    healed = np.full(5, 96.0)
    pad = np.full(40 - (15 + 2 + 5 + 5), 100.0)
    close = pd.Series(np.concatenate([peak, down, held, healed, pad]), index=idx)
    allowed = sleeve_allowed(close, 10, 0.10)
    assert not bool(allowed.iloc[16])
    assert not bool(allowed.iloc[20])
    assert bool(allowed.iloc[22])


def test_rising_prices_never_cut_and_the_scale_is_binary():
    idx = pd.bdate_range("2016-01-04", periods=80)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    scale = sleeve_scale(close, 20, 0.08)
    assert bool((scale == 1.0).all())
    assert scale.name == "sleeve_scale"


def test_signal_moves_only_the_sleeve_when_the_stop_is_on():
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    scale = pd.Series(1.0, index=idx)
    scale.iloc[-1] = 0.0
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, sleeve_scale=scale,
        dd_info={"drawdown_rule": "dd_w63_c8", "drawdown_stop_on": True},
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    assert payload["diagnostics"]["drawdown_stop_on"] is True
    assert payload["trade_reason"] == "drawdown_stop"

    future = scale.copy()
    future.iloc[-1] = 1.0
    future.iloc[20] = 0.0
    earlier = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, sleeve_scale=future,
        dd_info={"drawdown_rule": "dd_w63_c8", "drawdown_stop_on": False},
    )
    assert earlier["weights"]["UPRO"] == pytest.approx(0.5)
    assert "BIL" not in earlier["weights"]


def test_rejected_stop_does_not_change_the_signal(tmp_path):
    idx, close = _close(80)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": False,
            "status": "tested",
            "overlay": "drawdown_stop",
            "walk_forward_picks": [["2018-01-02", "dd_w63_c8"]],
            "variants": [{"name": "dd_w63_c8", "window": 63, "cut": 0.08, "reentry_fraction": 0.5}],
        }
    ]))
    scale, info = kept_drawdown_scale(path, close, as_of=idx[-1])
    assert scale is None and info is None


def test_kept_stop_uses_the_pick_known_at_as_of(tmp_path):
    idx, close = _close(400, seed=11)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "drawdown_stop",
            "walk_forward_picks": [
                ["2018-01-02", "dd_w63_c12"],
                ["2019-06-03", "dd_off"],
            ],
            "variants": [
                {"name": "dd_w63_c12", "window": 63, "cut": 0.12, "reentry_fraction": 0.5},
            ],
        }
    ]))
    scale, info = kept_drawdown_scale(path, close, as_of=idx[200])
    assert info["drawdown_rule"] == "dd_w63_c12"
    assert info["drawdown_window"] == 63
    assert info["drawdown_cut"] == pytest.approx(0.12)
    assert set(np.unique(scale.to_numpy())).issubset({0.0, 1.0})
    later, later_info = kept_drawdown_scale(path, close, as_of=idx[-1])
    assert later is None and later_info is None
