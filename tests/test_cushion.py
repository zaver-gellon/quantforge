"""Trend-cushion scale of the UPRO sleeve. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.cushion import (
    FLOORS,
    OFF_NAME,
    WIDTHS,
    WINDOW,
    cells,
    cushion_share,
    kept_cushion,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == (0.04, 0.0)
    assert found[-1] == (0.12, 0.5)
    assert len(found) == 6
    assert WIDTHS == (0.04, 0.08, 0.12)
    assert FLOORS == (0.0, 0.5)
    assert WINDOW == 200
    assert OFF_NAME == "cush_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 6
    assert variant_name(0.04, 0.0) == "cush_w04_f00"
    assert variant_name(0.08, 0.5) == "cush_w08_f50"
    assert variant_name(0.12, 0.0) == "cush_w12_f00"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [WIDTHS, FLOORS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_share_does_not_read_a_future_close_and_stays_inside_the_band():
    idx = pd.bdate_range("2014-01-02", periods=260)
    rng = np.random.default_rng(4)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)
    base = cushion_share(close, 0.08, 0.0)
    future = close.copy()
    future.iloc[220:] = close.iloc[220] * 0.5
    again = cushion_share(future, 0.08, 0.0)
    pd.testing.assert_series_equal(base.iloc[:220], again.iloc[:220])
    assert float(base.iloc[WINDOW - 2]) == 1.0
    assert float(base.min()) >= 0.0
    assert float(base.max()) <= 1.0
    flat = pd.Series(100.0, index=idx)
    assert float(cushion_share(flat, 0.08, 0.0).iloc[-1]) == 0.0
    assert float(cushion_share(flat, 0.08, 0.5).iloc[-1]) == pytest.approx(0.5)
    known = pd.Series(100.0, index=pd.bdate_range("2010-01-04", periods=WINDOW))
    known.iloc[-1] = 110.0
    average = (199.0 * 100.0 + 110.0) / WINDOW
    excess = 110.0 / average - 1.0
    scaled = min(1.0, max(0.0, excess / 0.10))
    assert float(cushion_share(known, 0.10, 0.0).iloc[-1]) == pytest.approx(scaled)
    assert float(cushion_share(known, 0.10, 0.5).iloc[-1]) == pytest.approx(0.5 + 0.5 * scaled)
    rich = known.copy()
    rich.iloc[-1] = 130.0
    assert float(cushion_share(rich, 0.08, 0.0).iloc[-1]) == 1.0
    assert float(cushion_share(rich, 0.08, 0.5).iloc[-1]) == 1.0
    weak = known.copy()
    weak.iloc[-1] = 90.0
    assert float(cushion_share(weak, 0.08, 0.0).iloc[-1]) == 0.0
    assert float(cushion_share(weak, 0.08, 0.5).iloc[-1]) == pytest.approx(0.5)
    wide = cushion_share(close, 0.12, 0.0)
    narrow = cushion_share(close, 0.04, 0.0)
    ready = close.index[WINDOW - 1:]
    assert bool((narrow.loc[ready] + 1e-12 >= wide.loc[ready]).all())
    low_floor = cushion_share(close, 0.08, 0.0)
    high_floor = cushion_share(close, 0.08, 0.5)
    assert bool((high_floor.loc[ready] + 1e-12 >= low_floor.loc[ready]).all())
    with pytest.raises(ValueError):
        cushion_share(close, 0.0, 0.0)
    with pytest.raises(ValueError):
        cushion_share(close, 0.08, 1.0)


def test_signal_blend_stays_in_equities_and_a_rejected_rule_does_not_load(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    on = pd.Series(True, index=idx)
    share = pd.Series(1.0, index=idx)
    share.iloc[-1] = 0.5
    spec = TrendCoreSpec(window=20)
    blended = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, cushion_share=share,
        cushion_info={"cushion_rule": "cush_w08_f00", "cushion_live": False},
    )
    assert blended["weights"]["SPY"] == pytest.approx(0.75)
    assert blended["weights"]["UPRO"] == pytest.approx(0.25)
    assert "BIL" not in blended["weights"]
    assert blended["trade_reason"] == "cushion"
    assert blended["diagnostics"]["sleeve_leverage_applied"] == pytest.approx(2.0)
    assert blended["diagnostics"]["cushion_share"] == pytest.approx(0.5)
    plain = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on,
    )
    full = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=on, cushion_share=pd.Series(1.0, index=idx),
    )
    assert full["weights"] == plain["weights"]
    off = pd.Series(False, index=idx)
    parked = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, cushion_share=pd.Series(0.0, index=idx),
    )
    assert parked["weights"]["SPY"] == pytest.approx(0.5)
    assert parked["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in parked["weights"]
    dipped = build_trend_signal(
        spec, close, generated_at="2026-01-01T00:00:00Z",
        trend_override=on, cushion_share=pd.Series(0.0, index=idx),
        dip_boost=pd.Series(1.0, index=idx),
    )
    assert dipped["weights"]["SSO"] == pytest.approx(0.5)
    assert dipped["weights"]["SPY"] == pytest.approx(0.5)
    assert "UPRO" not in dipped["weights"]
    with pytest.raises(ValueError, match="cannot both be kept"):
        build_trend_signal(
            spec, close, generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, trend_override=on,
            sleeve_upro_share=share, cushion_share=share,
        )

    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "id": "2026-11-10-cushion-scale",
        "kept": False,
        "status": "tested",
        "overlay": "cushion",
    }]))
    assert kept_cushion(path, close, as_of=idx[-1]) == (None, None)
    path.write_text(json.dumps([{
        "id": "2026-11-10-cushion-scale",
        "kept": True,
        "status": "tested",
        "overlay": "cushion",
        "variants": [{
            "name": "cush_w08_f00",
            "width": 0.08,
            "floor": 0.0,
        }],
        "walk_forward_picks": [["2018-01-02", "cush_off"], ["2018-02-01", "cush_w08_f00"]],
    }]))
    series, info = kept_cushion(path, close, as_of=idx[-1])
    assert info["cushion_rule"] == "cush_w08_f00"
    assert info["cushion_width"] == pytest.approx(0.08)
    assert info["cushion_floor"] == pytest.approx(0.0)
    assert info["cushion_live"] is False
    assert bool(series.loc[:pd.Timestamp("2018-01-31")].eq(1.0).all())
    later = series.loc[pd.Timestamp("2018-02-01"):]
    expected = cushion_share(close, 0.08, 0.0).reindex(later.index)
    pd.testing.assert_series_equal(later, expected, check_names=False)
    earlier, early_info = kept_cushion(path, close, as_of=pd.Timestamp("2018-01-31"))
    assert early_info["cushion_live"] is True
    assert float(earlier.iloc[-1]) == 1.0
