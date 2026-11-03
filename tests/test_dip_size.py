"""Sized, trend-gated dip. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.engine.backtest import run_dip_size, run_trend_core
from quantforge.engine.schedule import rebalance_mask
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.dipsize import (
    BOOSTS,
    GATES,
    cells,
    dip_boost,
    is_live_cell,
    kept_dip_size,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _down_book():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], index=idx)
    trend = pd.Series([True, True, True, True, False, True], index=idx)
    return idx, close, trend


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [
        ("any", 0.25), ("any", 0.50), ("any", 0.75), ("any", 1.0),
        ("trend_on", 0.25), ("trend_on", 0.50), ("trend_on", 0.75), ("trend_on", 1.0),
    ]
    assert GATES == ("any", "trend_on")
    assert BOOSTS == (0.25, 0.50, 0.75, 1.0)
    assert len({variant_name(*cell) for cell in found}) == 8
    assert variant_name("any", 1.0) == "dipsize_any_b100"
    assert variant_name("trend_on", 0.25) == "dipsize_trend_b025"
    assert variant_name("any", 0.50) == "dipsize_any_b050"
    assert is_live_cell("any", 1.0)
    assert not is_live_cell("trend_on", 1.0)
    assert not is_live_cell("any", 0.75)
    assert sum(is_live_cell(*cell) for cell in found) == 1
    for cell in found:
        neighbors = adjacent_cells(cell, [GATES, BOOSTS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_ungated_full_boost_matches_the_live_dip_and_ignores_trend():
    idx, close, trend = _down_book()
    live = dip_active(close)
    ungated = dip_boost(close, trend, "any", 1.0)
    assert list(ungated) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    pd.testing.assert_series_equal(ungated.gt(0).rename("dip_on"), live.astype(bool), check_names=False)
    flipped = trend.copy()
    flipped.iloc[:] = ~flipped.to_numpy()
    pd.testing.assert_series_equal(dip_boost(close, flipped, "any", 1.0), ungated)
    partial = dip_boost(close, trend, "any", 0.5)
    assert list(partial) == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]


def test_trend_gate_blocks_entry_and_flattens_without_reentering_the_same_close():
    idx, close, trend = _down_book()
    gated = dip_boost(close, trend, "trend_on", 0.5)
    # The third down close enters. The next close is still down, but the band
    # is off, so the dip flattens and does not take the fresh three-down signal
    # on that same close. The following close is a new entry because the band
    # is back on.
    assert list(gated) == [0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
    always = pd.Series(True, index=idx)
    both = dip_boost(close, always, "trend_on", 1.0)
    pd.testing.assert_series_equal(both, dip_boost(close, always, "any", 1.0))
    # An up close still ends the dip while the band stays on.
    bounce = pd.Series([10.0, 9.0, 8.0, 7.0, 8.0], index=idx[:5])
    ended = dip_boost(bounce, pd.Series(True, index=bounce.index), "trend_on", 1.0)
    assert list(ended) == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_boost_uses_no_future_close_or_trend():
    _idx, close, trend = _down_book()
    full = dip_boost(close, trend, "trend_on", 0.5)
    prefix = dip_boost(close.iloc[:5], trend.iloc[:5], "trend_on", 0.5)
    pd.testing.assert_series_equal(prefix, full.iloc[:5])
    later = trend.copy()
    later.iloc[-1] = False
    again = dip_boost(close, later, "trend_on", 0.5)
    pd.testing.assert_series_equal(again.iloc[:5], full.iloc[:5])
    assert again.iloc[-1] == 0.0
    assert full.iloc[-1] == 0.5


def test_boolean_boost_matches_the_trend_core_book():
    idx = pd.bdate_range("2014-01-02", periods=250)
    rng = np.random.default_rng(11)
    close_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    close = pd.Series(close_px, index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    sso_c = close * (1.0 + rng.normal(0.0, 0.002, len(idx)))
    upro_c = close * (1.0 + rng.normal(0.0, 0.002, len(idx)))
    sso_o = sso_c.shift(1)
    upro_o = upro_c.shift(1)
    sso_o.iloc[0] = sso_c.iloc[0]
    upro_o.iloc[0] = upro_c.iloc[0]
    trend = pd.Series(rng.random(len(idx)) > 0.3, index=idx)
    dip = pd.Series(rng.random(len(idx)) > 0.8, index=idx)
    mix = rebalance_mask(idx, "quarterly")
    rf = pd.Series(0.02 / 252.0, index=idx)
    args = dict(
        spy_open=open_, spy_close=close, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend,
        mix_rebalance=mix, core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    base = run_trend_core(**args, dip_on=dip)
    sized = run_dip_size(**args, dip_boost=dip.astype(float))
    pd.testing.assert_series_equal(sized.returns, base.returns)
    pd.testing.assert_series_equal(sized.turnover, base.turnover)
    pd.testing.assert_series_equal(sized.exposure, base.exposure)
    off = run_dip_size(**args, dip_boost=pd.Series(0.0, index=idx))
    plain = run_trend_core(**args, dip_on=pd.Series(False, index=idx))
    pd.testing.assert_series_equal(off.returns, plain.returns)


def test_half_boost_earns_the_blend_and_ignores_a_future_fraction():
    idx = pd.bdate_range("2020-01-02", periods=3)
    spy_o = pd.Series([100.0, 100.0, 100.0], index=idx)
    spy_c = pd.Series([100.0, 100.0, 110.0], index=idx)
    sso_o = pd.Series([100.0, 100.0, 100.0], index=idx)
    sso_c = pd.Series([100.0, 100.0, 120.0], index=idx)
    upro_o = pd.Series([100.0, 100.0, 100.0], index=idx)
    upro_c = pd.Series([100.0, 100.0, 100.0], index=idx)
    trend = pd.Series(True, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    half = pd.Series(0.5, index=idx)
    args = dict(
        spy_open=spy_o, spy_close=spy_c, sso_open=sso_o, sso_close=sso_c,
        upro_open=upro_o, upro_close=upro_c, trend_on=trend, mix_rebalance=mix,
        core_weight=0.5, sleeve_weight=0.5, cash_yield=rf,
        spy_cost=0.0, sso_cost=0.0, upro_cost=0.0,
    )
    blended = run_dip_size(**args, dip_boost=half)
    # 25% SPY at +10%, 25% SSO at +20%, 50% UPRO flat.
    assert blended.returns.iloc[-1] == pytest.approx(0.075)
    full = run_dip_size(**args, dip_boost=pd.Series(1.0, index=idx))
    assert full.returns.iloc[-1] == pytest.approx(0.10)
    later = half.copy()
    later.iloc[-1] = 1.0
    unchanged = run_dip_size(**args, dip_boost=later)
    pd.testing.assert_series_equal(unchanged.returns, blended.returns)
    # A half swap of the 50% core trades 25% of the book each way.
    flat = pd.Series(100.0, index=pd.bdate_range("2020-01-02", periods=4))
    step = pd.Series([0.0, 0.0, 0.5, 0.5], index=flat.index)
    traded = run_dip_size(
        flat, flat, flat, flat, flat, flat,
        pd.Series(True, index=flat.index), step, pd.Series(False, index=flat.index),
        0.5, 0.5, pd.Series(0.0, index=flat.index),
    )
    assert traded.turnover.iloc[3] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        run_dip_size(**args, dip_boost=pd.Series(1.5, index=idx))


def test_kept_size_uses_only_the_selected_cell(tmp_path):
    idx, close, trend = _down_book()
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "dip_size",
    }]))
    assert kept_dip_size(path, close, trend, as_of=idx[-1]) == (None, None)
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_size",
        "walk_forward_picks": [
            ["2020-01-02", "dipsize_trend_b050"],
            ["2020-01-09", "dipsize_any_b100"],
        ],
        "variants": [
            {"name": "dipsize_trend_b050", "gate": "trend_on", "boost": 0.5},
            {"name": "dipsize_any_b100", "gate": "any", "boost": 1.0},
        ],
    }]))
    # 2020-01-09 is the last session, and the second pick starts that day.
    series, info = kept_dip_size(path, close, trend, as_of=idx[-1])
    assert info["dip_size_rule"] == "dipsize_any_b100"
    assert info["dip_size_live"] is True
    assert list(series) == [0.0, 0.0, 0.0, 0.5, 0.0, 1.0]
    earlier, early_info = kept_dip_size(path, close, trend, as_of=pd.Timestamp("2020-01-08"))
    assert early_info["dip_size_rule"] == "dipsize_trend_b050"
    assert list(earlier) == [0.0, 0.0, 0.0, 0.5, 0.0]
    # The ungated cell would still be in the dip on that close. The selected
    # cell has flattened because the band is off.
    ungated = dip_boost(close.loc[: pd.Timestamp("2020-01-08")], trend.loc[: pd.Timestamp("2020-01-08")], "any", 1.0)
    assert ungated.iloc[-1] == 1.0
    signal = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[3], apply_dip=False,
        trend_override=pd.Series(True, index=idx),
        dip_boost=series, dip_size_info=info,
    )
    # As of the entry close the stitched fraction is still the gated half boost.
    # That holds 0.25 SPY and 0.25 SSO.
    assert signal["weights"]["SPY"] == pytest.approx(0.25)
    assert signal["weights"]["SSO"] == pytest.approx(0.25)
    assert signal["weights"]["UPRO"] == pytest.approx(0.5)
    assert signal["diagnostics"]["dip_on"] is True
    assert signal["diagnostics"]["dip_boost"] == pytest.approx(0.5)
    assert signal["trade_reason"] == "dip_overlay"
    outside = {"name": "dipsize_b200", "gate": "any", "boost": 2.0}
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "dip_size",
        "walk_forward_picks": [["2020-01-02", "dipsize_b200"]],
        "variants": [outside],
    }]))
    with pytest.raises(DataError):
        kept_dip_size(path, close, trend, as_of=idx[-1])
