"""Bill-yield gear on the UPRO sleeve. Synthetic yields only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.billgear import (
    FLOORS,
    OFF_NAME,
    THRESHOLDS,
    align_bill_yield,
    bill_share,
    cells,
    kept_bill_gear,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == (1.0, 0.0)
    assert found[-1] == (5.0, 0.5)
    assert len(found) == 6
    assert THRESHOLDS == (1.0, 3.0, 5.0)
    assert FLOORS == (0.0, 0.5)
    assert OFF_NAME == "bill_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 6
    assert variant_name(1.0, 0.0) == "bill_t01_f00"
    assert variant_name(3.0, 0.5) == "bill_t03_f50"
    assert variant_name(5.0, 0.0) == "bill_t05_f00"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [THRESHOLDS, FLOORS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_share_uses_only_that_close_and_stays_inside_the_grid():
    idx = pd.bdate_range("2014-01-02", periods=30)
    quotes = pd.Series(np.linspace(0.2, 6.0, len(idx)), index=idx)
    base = bill_share(quotes, 3.0, 0.0)
    future = quotes.copy()
    future.iloc[20:] = 9.0
    again = bill_share(future, 3.0, 0.0)
    pd.testing.assert_series_equal(base.iloc[:20], again.iloc[:20])
    assert float(base.min()) >= 0.0
    assert float(base.max()) <= 1.0
    at_level = pd.Series([3.0], index=idx[:1])
    assert float(bill_share(at_level, 3.0, 0.0).iloc[-1]) == 1.0
    above = pd.Series([3.0 + 1e-6], index=idx[:1])
    assert float(bill_share(above, 3.0, 0.0).iloc[-1]) == 0.0
    assert float(bill_share(above, 3.0, 0.5).iloc[-1]) == pytest.approx(0.5)
    cheap = pd.Series([0.05], index=idx[:1])
    assert float(bill_share(cheap, 1.0, 0.0).iloc[-1]) == 1.0
    missing = quotes.copy()
    missing.iloc[10] = np.nan
    assert float(bill_share(missing, 1.0, 0.0).iloc[10]) == 1.0
    low_threshold = bill_share(quotes, 1.0, 0.0)
    high_threshold = bill_share(quotes, 5.0, 0.0)
    assert bool((high_threshold + 1e-12 >= low_threshold).all())
    low_floor = bill_share(quotes, 3.0, 0.0)
    high_floor = bill_share(quotes, 3.0, 0.5)
    assert bool((high_floor + 1e-12 >= low_floor).all())
    with pytest.raises(ValueError):
        bill_share(quotes, 0.0, 0.0)
    with pytest.raises(ValueError):
        bill_share(quotes, 3.0, 1.0)


def test_align_fills_forward_and_does_not_fill_backward():
    idx = pd.bdate_range("2020-01-02", periods=5)
    raw = pd.Series([1.0, 4.0], index=[idx[1], idx[2]])
    aligned = align_bill_yield(raw, idx)
    assert pd.isna(aligned.iloc[0])
    assert float(aligned.iloc[1]) == pytest.approx(1.0)
    assert float(aligned.iloc[3]) == pytest.approx(4.0)
    later = pd.concat([raw, pd.Series([9.0], index=[idx[4]])])
    again = align_bill_yield(later, idx)
    pd.testing.assert_series_equal(aligned.iloc[:4], again.iloc[:4])
    assert float(again.iloc[4]) == pytest.approx(9.0)


def test_signal_blend_stays_in_equities_and_a_rejected_rule_does_not_load(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    on = pd.Series(True, index=idx)
    share = pd.Series(1.0, index=idx)
    share.iloc[-1] = 0.5
    spec = TrendCoreSpec(window=20)
    blended = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, bill_share=share,
        bill_info={"bill_rule": "bill_t03_f50", "bill_live": False},
    )
    assert blended["weights"]["SPY"] == pytest.approx(0.75)
    assert blended["weights"]["UPRO"] == pytest.approx(0.25)
    assert "BIL" not in blended["weights"]
    assert blended["trade_reason"] == "bill_gear"
    assert blended["diagnostics"]["sleeve_leverage_applied"] == pytest.approx(2.0)
    assert blended["diagnostics"]["bill_share"] == pytest.approx(0.5)
    plain = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on,
    )
    full = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, bill_share=pd.Series(1.0, index=idx),
    )
    assert full["weights"] == plain["weights"]
    assert "bill_share" not in plain["diagnostics"]
    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, bill_share=pd.Series(0.0, index=idx),
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]
    dipped = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, bill_share=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]
    with pytest.raises(ValueError, match="cannot both be kept"):
        build_trend_signal(
            spec, close, generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, trend_override=on,
            cushion_share=share, bill_share=share,
        )

    quotes = pd.Series(4.0, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "id": "2026-11-11-bill-gear",
        "kept": False,
        "status": "tested",
        "overlay": "bill_gear",
    }]))
    assert kept_bill_gear(path, quotes, as_of=idx[-1]) == (None, None)
    path.write_text(json.dumps([{
        "id": "2026-11-11-bill-gear",
        "kept": True,
        "status": "tested",
        "overlay": "bill_gear",
        "variants": [{
            "name": "bill_t03_f00",
            "threshold_pct": 3.0,
            "floor": 0.0,
        }],
        "walk_forward_picks": [["2018-01-02", "bill_off"], ["2018-02-01", "bill_t03_f00"]],
    }]))
    series, info = kept_bill_gear(path, quotes, as_of=idx[-1])
    assert info["bill_rule"] == "bill_t03_f00"
    assert info["bill_threshold_pct"] == pytest.approx(3.0)
    assert info["bill_floor"] == pytest.approx(0.0)
    assert info["bill_live"] is False
    assert info["bill_yield_pct"] == pytest.approx(4.0)
    assert bool(series.loc[:pd.Timestamp("2018-01-31")].eq(1.0).all())
    later = series.loc[pd.Timestamp("2018-02-01"):]
    expected = bill_share(quotes, 3.0, 0.0).reindex(later.index)
    pd.testing.assert_series_equal(later, expected, check_names=False)
    earlier, early_info = kept_bill_gear(path, quotes, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["bill_live"] is True
    assert float(earlier.iloc[-1]) == 1.0
    leaked = quotes.copy()
    leaked.iloc[-1] = 0.1
    truncated, truncated_info = kept_bill_gear(path, leaked, as_of=pd.Timestamp("2018-02-15"))
    assert truncated.index[-1] <= pd.Timestamp("2018-02-15")
    assert float(truncated.iloc[-1]) == 0.0
    assert truncated_info["bill_yield_pct"] == pytest.approx(4.0)
    assert float(leaked.iloc[-1]) == pytest.approx(0.1)
