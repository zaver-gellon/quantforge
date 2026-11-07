"""Exponential sleeve filter. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.ema import (
    BANDS,
    OFF_NAME,
    SPANS,
    cells,
    ema_state,
    kept_ema_trend,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found == [
        (100, 0.0), (100, 0.02), (150, 0.0), (150, 0.02), (200, 0.0), (200, 0.02),
    ]
    assert SPANS == (100, 150, 200)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "ema_off"
    assert len({variant_name(*cell) for cell in found}) == 6
    assert variant_name(100, 0.0) == "ema_s100_b0"
    assert variant_name(100, 0.02) == "ema_s100_b2"
    assert variant_name(200, 0.02) == "ema_s200_b2"
    assert OFF_NAME not in {variant_name(*cell) for cell in found}
    for cell in found:
        neighbors = adjacent_cells(cell, [SPANS, BANDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_recursive_average_and_no_lookahead():
    idx = pd.bdate_range("2019-01-02", periods=40)
    prices = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.2, len(idx))), index=idx)
    span = 8
    alpha = 2.0 / (span + 1.0)
    ema = float(prices.iloc[0])
    means = []
    for i, price in enumerate(prices.to_numpy(dtype=float)):
        if i:
            ema = alpha * float(price) + (1.0 - alpha) * ema
        means.append(ema if i + 1 >= span else np.nan)
    expected = pd.Series(means, index=idx)
    pandas_ema = prices.ewm(span=span, min_periods=span, adjust=False).mean()
    pd.testing.assert_series_equal(pandas_ema, expected, check_names=False)

    state = False
    hand = []
    for price, mean in zip(prices.to_numpy(dtype=float), means):
        if np.isfinite(mean):
            if price > mean:
                state = True
            elif price < mean:
                state = False
        hand.append(state)
    pd.testing.assert_series_equal(
        ema_state(prices, span, 0.0),
        pd.Series(hand, index=idx, name="ema_on"),
    )
    # The average is undefined, and the sleeve stays off, until ``span`` closes.
    assert not bool(ema_state(prices, span, 0.0).iloc[span - 2])

    full = ema_state(prices, span, 0.02)
    cut = ema_state(prices.iloc[:25], span, 0.02)
    pd.testing.assert_series_equal(full.iloc[:25], cut)
    with pytest.raises(ValueError, match="span"):
        ema_state(prices, 1, 0.0)
    with pytest.raises(ValueError, match="band"):
        ema_state(prices, span, -0.01)


def test_exponential_filter_is_not_the_simple_average():
    idx = pd.bdate_range("2010-01-04", periods=80)
    ramp = np.linspace(100.0, 160.0, 40)
    crash = np.linspace(160.0, 110.0, 8)
    recover = np.linspace(110.0, 180.0, 32)
    prices = pd.Series(np.concatenate([ramp, crash, recover]), index=idx)
    ema = ema_state(prices, 20, 0.0)
    sma = trend_state(prices, 20, 0.0, "daily")
    assert not ema.equals(sma)
    # Inside the band the previous state is kept.
    flat = pd.Series(np.concatenate([np.linspace(100.0, 140.0, 30), np.repeat(136.0, 20)]), index=idx[:50])
    banded = ema_state(flat, 10, 0.05)
    assert bool(banded.iloc[29])
    assert bool(banded.iloc[-1])


def test_rejected_rule_is_not_reloaded_and_a_kept_rule_ignores_later_prices(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "ema_trend",
        "walk_forward_picks": [["2018-01-02", "ema_s100_b0"]],
        "variants": [{"name": "ema_s100_b0", "span": 100, "band": 0.0}],
    }]))
    assert kept_ema_trend(path, close, as_of=idx[-1]) == (None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "ema_trend",
        "walk_forward_picks": [
            ["2018-01-03", "ema_s100_b0"],
            ["2018-02-01", "ema_off"],
        ],
        "variants": [
            {"name": "ema_s100_b0", "span": 100, "band": 0.0},
            {"name": "ema_off", "span": None, "band": None},
        ],
    }]))
    assert kept_ema_trend(path, close, as_of=idx[-1]) == (None, None)

    earlier, early_info = kept_ema_trend(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["ema_rule"] == "ema_s100_b0"
    assert early_info["ema_span"] == 100
    assert early_info["ema_band"] == 0.0
    assert earlier.index[-1] < pd.Timestamp("2018-02-01")
    bumped = close.copy()
    bumped.iloc[-1] = 50.0
    unchanged, unchanged_info = kept_ema_trend(path, bumped, as_of=pd.Timestamp("2018-01-31"))
    pd.testing.assert_series_equal(unchanged, earlier)
    assert unchanged_info["ema_on"] == early_info["ema_on"]

    # Span 100 is not defined on 40 closes, so the kept state stays off.
    off = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=earlier.index[-1],
        generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=earlier, ema_info=early_info,
    )
    assert off["weights"]["SPY"] == pytest.approx(0.5)
    assert off["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in off["weights"]
    assert off["diagnostics"]["ema_rule"] == "ema_s100_b0"
    assert off["diagnostics"]["trend_on"] is False

    rising = pd.Series(np.linspace(100.0, 180.0, len(idx)), index=idx)
    on_state = ema_state(rising, 10, 0.0)
    assert bool(on_state.iloc[-1])
    held = build_trend_signal(
        TrendCoreSpec(window=20), rising, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on_state,
    )
    assert held["weights"]["SPY"] == pytest.approx(0.5)
    assert held["weights"]["UPRO"] == pytest.approx(0.5)
