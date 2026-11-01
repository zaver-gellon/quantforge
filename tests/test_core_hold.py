"""Off-trend core hold. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_core_hold, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.corehold import (
    HOLDS,
    LIVE_HOLD,
    cells,
    is_live_cell,
    kept_core_hold,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _book(n=400, seed=5):
    idx = pd.bdate_range("2010-01-04", periods=n)
    rng = np.random.default_rng(seed)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    close = pd.Series(close_px, index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    sso_c = close * (1.0 + rng.normal(0.0, 0.001, n))
    upro_c = close * (1.0 + rng.normal(0.0, 0.001, n))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(n) > 0.35, index=idx)
    dip = pd.Series(rng.random(n) > 0.85, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    return idx, open_, close, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [(0.0,), (0.25,), (0.5,), (0.75,), (1.0,)]
    assert HOLDS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert LIVE_HOLD == 1.0
    assert len({variant_name(*cell) for cell in found}) == 5
    assert sum(not is_live_cell(*cell) for cell in found) == 4
    assert is_live_cell(1.0)
    assert not is_live_cell(0.75)
    assert variant_name(0.0) == "corehold_f000"
    assert variant_name(0.25) == "corehold_f025"
    assert variant_name(1.0) == "corehold_f100"
    for cell in found:
        neighbors = adjacent_cells(cell, [HOLDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_full_hold_matches_the_trend_core_book():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, trend, dip, mix, rf = _book()
    args = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    base = run_trend_core(**args)
    held = run_core_hold(**args, off_core_fraction=1.0)
    pd.testing.assert_series_equal(held.returns, base.returns)
    pd.testing.assert_series_equal(held.turnover, base.turnover)


def test_zero_hold_drops_the_off_trend_spy_loss_and_ignores_a_future_state():
    idx = pd.bdate_range("2020-01-02", periods=6)
    spy_c = pd.Series([100.0, 100.0, 100.0, 100.0, 80.0, 80.0], index=idx)
    spy_o = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 80.0], index=idx)
    trend = pd.Series(False, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    args = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=spy_o, sso_close=spy_c,
        upro_open=spy_o, upro_close=spy_c, trend_on=trend, dip_on=dip,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    full = run_core_hold(**args, off_core_fraction=1.0)
    half = run_core_hold(**args, off_core_fraction=0.5)
    zero = run_core_hold(**args, off_core_fraction=0.0)
    assert full.returns.iloc[4] == pytest.approx(-0.10)
    assert half.returns.iloc[4] == pytest.approx(-0.05)
    assert zero.returns.iloc[4] == pytest.approx(0.0)
    # Before the drop the strategic mix is still in place. After it, the SPY pile is a smaller share.
    assert zero.exposure.iloc[3] == pytest.approx(0.0)
    assert half.exposure.iloc[3] == pytest.approx(0.25)
    assert full.exposure.iloc[3] == pytest.approx(0.5)

    later = trend.copy()
    later.iloc[4:] = True
    again = run_core_hold(**{**args, "trend_on": later}, off_core_fraction=0.5)
    pd.testing.assert_series_equal(half.returns.iloc[:5], again.returns.iloc[:5])


def test_signal_uses_a_kept_hold_only_while_the_trend_is_off(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(100.0, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "core_hold",
        "walk_forward_picks": [["2018-01-02", "corehold_f000"]],
        "variants": [{"name": "corehold_f000", "hold": 0.0}],
    }]))
    assert kept_core_hold(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "core_hold",
        "walk_forward_picks": [
            ["2018-01-03", "corehold_f050"],
            ["2018-02-01", "corehold_f025"],
        ],
        "variants": [
            {"name": "corehold_f050", "hold": 0.5},
            {"name": "corehold_f025", "hold": 0.25},
            {"name": "corehold_f100", "hold": 1.0},
        ],
    }]))
    series, info = kept_core_hold(path, close, as_of=idx[-1])
    assert info["core_hold_rule"] == "corehold_f025"
    assert info["core_hold_fraction"] == pytest.approx(0.25)
    assert series.loc[pd.Timestamp("2018-01-02")] == pytest.approx(1.0)
    assert series.loc[pd.Timestamp("2018-01-31")] == pytest.approx(0.5)
    assert series.iloc[-1] == pytest.approx(0.25)

    off = pd.Series(False, index=idx)
    on = pd.Series(True, index=idx)
    parked = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, core_hold=series, core_hold_info=info,
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.125)
    assert parked["weights"]["BIL"] == pytest.approx(0.875)
    assert "UPRO" not in parked["weights"]
    assert parked["diagnostics"]["core_hold_fraction"] == pytest.approx(0.25)
    stepped = series.copy()
    stepped.iloc[-1] = 0.0
    resized = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, core_hold=stepped,
    )
    assert resized["weights"]["BIL"] == pytest.approx(1.0)
    assert resized["trade_reason"] == "core_hold"

    invested = build_trend_signal(
        TrendCoreSpec(window=20), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, core_hold=series, core_hold_info=info,
    )
    assert invested["weights"]["SPY"] == pytest.approx(0.5)
    assert invested["weights"]["UPRO"] == pytest.approx(0.5)
    assert invested["diagnostics"]["core_hold_applied"] == pytest.approx(1.0)

    earlier, early_info = kept_core_hold(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["core_hold_rule"] == "corehold_f050"
    assert earlier.iloc[-1] == pytest.approx(0.5)
    # A later pick must not leak into the series visible on the earlier date.
    assert pd.Timestamp("2018-02-01") not in earlier.index or earlier.index[-1] < pd.Timestamp("2018-02-01")
