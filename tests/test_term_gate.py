"""VIX term-structure gate on the UPRO sleeve. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.termgate import (
    BAND,
    FLOORS,
    OFF_NAME,
    THRESHOLDS,
    WINDOW,
    cells,
    kept_term_gate,
    term_ratio,
    term_share,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def _prices(values, start: str = "2018-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == (1.0, 0.0)
    assert found[-1] == (1.10, 0.5)
    assert len(found) == 6
    assert THRESHOLDS == (1.0, 1.05, 1.10)
    assert FLOORS == (0.0, 0.5)
    assert WINDOW == 200
    assert BAND == 0.02
    assert OFF_NAME == "term_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 6
    assert variant_name(1.0, 0.0) == "term_t100_f00"
    assert variant_name(1.05, 0.5) == "term_t105_f50"
    assert variant_name(1.10, 0.0) == "term_t110_f00"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [THRESHOLDS, FLOORS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_ratio_uses_the_same_session_and_does_not_fill_forward():
    index = pd.bdate_range("2020-01-02", periods=4)
    vix = pd.Series([20.0, 30.0, 24.0, 18.0], index=index)
    # The second session has no three-month print. The third has no one-month print.
    vix3m = pd.Series([20.0, 22.0], index=pd.DatetimeIndex([index[0], index[3]]))
    ratio = term_ratio(vix, vix3m, index)
    assert ratio.iloc[0] == pytest.approx(1.0)
    assert pd.isna(ratio.iloc[1])
    assert pd.isna(ratio.iloc[2])
    assert ratio.iloc[3] == pytest.approx(18.0 / 22.0)
    # A later print does not rewrite the earlier ratio.
    shifted = vix.copy()
    shifted.iloc[-1] = 80.0
    again = term_ratio(shifted, vix3m, index)
    pd.testing.assert_series_equal(ratio.iloc[:3], again.iloc[:3])
    # A print that is not on the equity calendar is not moved onto the next session.
    extra = vix3m.copy()
    extra.loc[index[1] + pd.Timedelta(days=1)] = 10.0
    untouched = term_ratio(vix, extra, index)
    assert pd.isna(untouched.iloc[1])
    assert pd.isna(untouched.iloc[2])


def test_share_cuts_only_on_trend_when_the_curve_is_inverted():
    index = pd.bdate_range("2020-01-02", periods=5)
    ratio = pd.Series([0.99, 1.00, 1.04, np.nan, 1.10], index=index)
    on = pd.Series(True, index=index)
    share = term_share(ratio, on, 1.0, 0.0)
    assert list(share.round(6)) == [1.0, 0.0, 0.0, 1.0, 0.0]
    half = term_share(ratio, on, 1.05, 0.5)
    assert list(half.round(6)) == [1.0, 1.0, 1.0, 1.0, 0.5]
    # Equality at 1.05 cuts. 1.04 does not.
    edge = term_share(ratio, on, 1.05, 0.0)
    assert float(edge.iloc[1]) == 1.0
    assert float(edge.iloc[4]) == 0.0
    off = pd.Series(False, index=index)
    assert bool(term_share(ratio, off, 1.0, 0.0).eq(1.0).all())
    mixed = on.copy()
    mixed.iloc[1] = False
    flattened = term_share(ratio, mixed, 1.0, 0.0)
    assert float(flattened.iloc[1]) == 1.0
    assert float(flattened.iloc[2]) == 0.0
    future = ratio.copy()
    future.iloc[-1] = 0.5
    again = term_share(future, on, 1.0, 0.0)
    pd.testing.assert_series_equal(share.iloc[:4], again.iloc[:4])
    assert float(again.iloc[-1]) == 1.0
    with pytest.raises(ValueError):
        term_share(ratio, on, 0.95, 0.0)
    with pytest.raises(ValueError):
        term_share(ratio, on, 1.0, 1.0)
    with pytest.raises(ValueError):
        term_share(pd.Series([np.inf], index=index[:1]), on.iloc[:1], 1.0, 0.0)


def test_signal_blend_stays_in_equities_and_a_rejected_rule_does_not_load(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    on = pd.Series(True, index=idx)
    share = pd.Series(1.0, index=idx)
    share.iloc[-1] = 0.5
    spec = TrendCoreSpec(window=20)
    blended = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, term_share=share,
        term_info={"term_rule": "term_t105_f50", "term_live": False, "term_ratio": 1.08},
    )
    assert blended["weights"]["SPY"] == pytest.approx(0.75)
    assert blended["weights"]["UPRO"] == pytest.approx(0.25)
    assert "BIL" not in blended["weights"]
    assert blended["trade_reason"] == "term_gate"
    assert blended["diagnostics"]["sleeve_leverage_applied"] == pytest.approx(2.0)
    assert blended["diagnostics"]["term_share"] == pytest.approx(0.5)
    assert blended["diagnostics"]["term_ratio"] == pytest.approx(1.08)
    plain = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on,
    )
    full = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, term_share=pd.Series(1.0, index=idx),
    )
    assert full["weights"] == plain["weights"]
    assert "term_share" not in plain["diagnostics"]
    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, term_share=pd.Series(0.0, index=idx),
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]
    dipped = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, term_share=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]
    with pytest.raises(ValueError, match="cannot both be kept"):
        build_trend_signal(
            spec, close, generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, trend_override=on,
            pause_share=share, term_share=share,
        )

    vix = pd.Series(24.0, index=idx)
    vix3m = pd.Series(20.0, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "id": "2026-11-13-term-gate",
        "kept": False,
        "status": "tested",
        "overlay": "term_gate",
    }]))
    assert kept_term_gate(path, close, vix, vix3m, as_of=idx[-1]) == (None, None)

    rising = pd.Series(np.linspace(100.0, 180.0, 260), index=pd.bdate_range("2014-01-02", periods=260))
    high = pd.Series(24.0, index=rising.index)
    calm = pd.Series(20.0, index=rising.index)
    path.write_text(json.dumps([{
        "id": "2026-11-13-term-gate",
        "kept": True,
        "status": "tested",
        "overlay": "term_gate",
        "variants": [
            {"name": "term_t100_f00", "threshold": 1.0, "floor": 0.0},
            {"name": "term_t105_f50", "threshold": 1.05, "floor": 0.5},
        ],
        "walk_forward_picks": [["2014-01-02", "term_off"], ["2014-06-02", "term_t105_f50"]],
    }]))
    series, info = kept_term_gate(path, rising, high, calm, as_of=rising.index[-1])
    assert info["term_rule"] == "term_t105_f50"
    assert info["term_threshold"] == pytest.approx(1.05)
    assert info["term_floor"] == pytest.approx(0.5)
    assert info["term_ratio"] == pytest.approx(1.2)
    assert info["term_live"] is False
    assert info["term_on"] is True
    trend = trend_state(rising, WINDOW, BAND, "daily")
    expected = term_share(term_ratio(high, calm, rising.index), trend, 1.05, 0.5)
    later = series.loc[pd.Timestamp("2014-06-02"):]
    pd.testing.assert_series_equal(later, expected.reindex(later.index), check_names=False)
    assert bool(series.loc[:pd.Timestamp("2014-05-30")].eq(1.0).all())
    earlier, early_info = kept_term_gate(path, rising, high, calm, as_of=pd.Timestamp("2014-05-30"))
    assert early_info["term_live"] is True
    assert float(earlier.iloc[-1]) == 1.0
    leaked_vix = high.copy()
    leaked_vix.iloc[-1] = 80.0
    truncated, _info = kept_term_gate(path, rising, leaked_vix, calm, as_of=rising.index[-5])
    assert truncated.index[-1] == rising.index[-5]
    untouched = kept_term_gate(path, rising, high, calm, as_of=rising.index[-5])[0]
    pd.testing.assert_series_equal(truncated, untouched)
