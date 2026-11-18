"""Credit-regime gate. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.credit import gate_trend
from quantforge.strategies.creditregime import (
    BANDS,
    FALLBACK,
    OFF_NAME,
    PRIMARY,
    TREASURY,
    WINDOWS,
    align_close,
    cells,
    credit_ratio,
    has_kept_credit_regime,
    kept_credit_regime,
    merge_hedge,
    pair_ready_dates,
    regime_state,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _sessions(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def test_grid_is_fixed_and_the_live_book_sorts_first():
    found = cells()
    assert WINDOWS == (100, 150, 200)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "cr_off"
    assert PRIMARY == "HYG"
    assert FALLBACK == "LQD"
    assert TREASURY == "IEF"
    assert found == [
        (100, 0.0), (100, 0.02),
        (150, 0.0), (150, 0.02),
        (200, 0.0), (200, 0.02),
    ]
    names = [variant_name(*cell) for cell in found]
    assert names == [
        "zcr_w100_b00", "zcr_w100_b02",
        "zcr_w150_b00", "zcr_w150_b02",
        "zcr_w200_b00", "zcr_w200_b02",
    ]
    assert len(set(names)) == 6
    assert OFF_NAME < min(names)
    with pytest.raises(ValueError):
        variant_name(126, 0.0)
    with pytest.raises(ValueError):
        variant_name(100, 0.01)
    for window in WINDOWS:
        for band in BANDS:
            neighbors = adjacent_cells((window, band), [WINDOWS, BANDS])
            assert neighbors
            assert (window, band) not in neighbors
    assert adjacent_cells((150, 0.0), [WINDOWS, BANDS]) == [
        (100, 0.0), (200, 0.0), (150, 0.02),
    ]
    assert adjacent_cells((100, 0.02), [WINDOWS, BANDS]) == [(150, 0.02), (100, 0.0)]


def test_alignment_does_not_use_a_later_print():
    prices = pd.Series([10.0, 12.0, 30.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-07"]))
    sessions = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    aligned = align_close(prices, sessions)
    assert float(aligned.iloc[2]) == 12.0
    assert float(aligned.iloc[3]) == 30.0
    assert list(aligned.index) == list(pd.DatetimeIndex(sessions).normalize())


def test_before_either_average_the_gate_does_not_block():
    idx = _sessions(12)
    ief = pd.Series(100.0, index=idx)
    lqd = pd.Series(80.0, index=idx)
    hyg = pd.Series(np.nan, index=idx)
    healthy, stress, pairs, averages = regime_state(hyg, lqd, ief, idx, window=20, band=0.0)
    assert bool(stress.any()) is False
    assert bool(healthy.all()) is True
    assert set(pairs) == {""}
    assert not bool(np.isfinite(averages.to_numpy(dtype=float)).any())


def test_lqd_is_the_fallback_until_hyg_average_exists_then_hyg_takes_over():
    idx = _sessions(30)
    ief = pd.Series(100.0, index=idx)
    lqd = pd.Series(100.0, index=idx)
    lqd.iloc[8:] = 70.0
    hyg = pd.Series(np.nan, index=idx)
    hyg.iloc[15:] = 130.0
    _healthy, stress, pairs, averages = regime_state(hyg, lqd, ief, idx, window=5, band=0.0)
    assert list(pairs.iloc[:4]) == ["", "", "", ""]
    assert pairs.iloc[4] == "LQD/IEF"
    assert pairs.iloc[14] == "LQD/IEF"
    assert bool(stress.iloc[4]) is False
    assert bool(stress.iloc[8]) is True
    assert pairs.iloc[19] == "HYG/IEF"
    assert bool(stress.iloc[19]) is False
    assert np.isfinite(averages.iloc[19])
    ready = pair_ready_dates(hyg, lqd, ief, idx, windows=(5,))
    assert ready["LQD_w5"] == str(idx[4].date())
    assert ready["HYG_w5"] == str(idx[19].date())


def test_a_handoff_does_not_inherit_the_fallback_stress():
    idx = _sessions(20)
    ief = pd.Series(100.0, index=idx)
    lqd = pd.Series(100.0, index=idx)
    lqd.iloc[6:] = 70.0
    hyg = pd.Series(np.nan, index=idx)
    hyg.iloc[10:] = 100.0
    _healthy, stress, pairs, _averages = regime_state(hyg, lqd, ief, idx, window=5, band=0.02)
    assert pairs.iloc[9] == "LQD/IEF"
    assert bool(stress.iloc[9]) is True
    assert pairs.iloc[13] == "LQD/IEF"
    assert pairs.iloc[14] == "HYG/IEF"
    assert bool(stress.iloc[14]) is False


def test_band_holds_the_previous_state_and_a_zero_band_crosses():
    idx = _sessions(20)
    ief = pd.Series(100.0, index=idx)
    hyg = pd.Series(np.nan, index=idx)
    lqd = pd.Series(100.0, index=idx)
    lqd.iloc[10:] = 99.0
    wide, wide_stress, _pairs, _avg = regime_state(hyg, lqd, ief, idx, window=5, band=0.02)
    tight, tight_stress, _pairs2, _avg2 = regime_state(hyg, lqd, ief, idx, window=5, band=0.0)
    assert bool(wide_stress.iloc[10]) is False
    assert bool(tight_stress.iloc[10]) is True
    assert bool(wide.iloc[10]) is True
    lqd.iloc[12:] = 90.0
    deep, deep_stress, _pairs3, _avg3 = regime_state(hyg, lqd, ief, idx, window=5, band=0.02)
    assert bool(deep_stress.iloc[11]) is False
    assert bool(deep_stress.iloc[12]) is True
    assert bool(deep.iloc[12]) is False


def test_state_does_not_read_a_future_close():
    idx = _sessions(40)
    rng = np.random.default_rng(4)
    ief = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0, 0.004, len(idx)))), index=idx)
    lqd = pd.Series(80 * np.exp(np.cumsum(rng.normal(0.0, 0.006, len(idx)))), index=idx)
    hyg = pd.Series(90 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx)))), index=idx)
    base_h, base_s, base_p, base_a = regime_state(hyg, lqd, ief, idx, window=10, band=0.02)
    crashed = hyg.copy()
    crashed.iloc[25:] = hyg.iloc[25] * 0.4
    later_ief = ief.copy()
    later_ief.iloc[25:] = ief.iloc[25] * 2.0
    again_h, again_s, again_p, again_a = regime_state(crashed, lqd, later_ief, idx, window=10, band=0.02)
    pd.testing.assert_series_equal(base_h.iloc[:25], again_h.iloc[:25])
    pd.testing.assert_series_equal(base_s.iloc[:25], again_s.iloc[:25])
    pd.testing.assert_series_equal(base_p.iloc[:25], again_p.iloc[:25])
    pd.testing.assert_series_equal(base_a.iloc[:25], again_a.iloc[:25])


def test_stress_gates_only_by_and_with_the_equity_trend():
    idx = _sessions(8)
    equity = pd.Series([True, True, False, True], index=idx[:4])
    stress = pd.Series([False, True, True, False], index=idx[:4])
    gated = gate_trend(equity, stress)
    assert list(gated.astype(bool)) == [True, False, False, True]


def test_ratio_rejects_a_non_positive_treasury():
    idx = _sessions(3)
    credit = pd.Series([10.0, 10.0, 10.0], index=idx)
    treasury = pd.Series([5.0, 0.0, -1.0], index=idx)
    ratio = credit_ratio(credit, treasury)
    assert float(ratio.iloc[0]) == 2.0
    assert not np.isfinite(ratio.iloc[1])
    assert not np.isfinite(ratio.iloc[2])


def test_kept_regime_ignores_a_rejection_and_does_not_read_a_later_close(tmp_path):
    idx = _sessions(30)
    ief = pd.Series(100.0, index=idx)
    lqd = pd.Series(100.0, index=idx)
    lqd.iloc[12:] = 60.0
    hyg = pd.Series(np.nan, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w100_b00"]],
        "variants": [{"name": "zcr_w100_b00", "window": 100, "band": 0.0}],
    }]))
    assert has_kept_credit_regime(path) is False
    assert kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1]) == (None, None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "cr_off"]],
        "variants": [{"name": "cr_off", "window": None, "band": None}],
    }]))
    assert has_kept_credit_regime(path) is True
    assert kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1]) == (None, None, None)

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w100_b00"]],
        "variants": [{"name": "zcr_w100_b00", "window": 100, "band": 0.0}],
    }]))
    # Window 100 is not full on 30 sessions, so the kept cell does not block.
    idle, rule, info = kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1])
    assert rule == "zcr_w100_b00"
    assert bool(idle.iloc[-1]) is False
    assert info["credit_stress"] is False
    assert info["credit_pair"] is None

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w100_b02"]],
        "variants": [{"name": "zcr_w100_b02", "window": 5, "band": 0.02}],
    }]))
    with pytest.raises(DataError):
        kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1])

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w150_b00"]],
        "variants": [{"name": "zcr_w150_b00", "window": 150, "band": 0.0}],
    }]))
    # The stored window is outside the sessions but inside the grid, so the
    # average is not ready and the gate does not block.
    quiet, quiet_rule, quiet_info = kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1])
    assert quiet_rule == "zcr_w150_b00"
    assert bool(quiet.any()) is False
    assert quiet_info["credit_pair"] is None

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w100_b00"]],
        "variants": [{"name": "zcr_w100_b00", "window": 100, "band": 0.0}],
    }]))
    # Rebuild a short window by writing a grid cell whose parameters match
    # the name. The name check uses the grid, so use window 100 only when
    # the series is long enough. Here the point is the as-of cut.
    long_idx = _sessions(120)
    long_ief = pd.Series(100.0, index=long_idx)
    long_lqd = pd.Series(100.0, index=long_idx)
    long_lqd.iloc[110:] = 50.0
    long_hyg = pd.Series(np.nan, index=long_idx)
    stress, rule, info = kept_credit_regime(path, long_idx, long_hyg, long_lqd, long_ief, as_of=long_idx[-1])
    assert rule == "zcr_w100_b00"
    assert info["credit_pair"] == "LQD/IEF"
    assert info["credit_stress"] is True
    earlier, _rule, early_info = kept_credit_regime(
        path, long_idx, long_hyg, long_lqd, long_ief, as_of=long_idx[100],
    )
    assert bool(earlier.iloc[-1]) is False
    assert early_info["credit_stress"] is False
    assert len(earlier) == 101
    leaked = long_lqd.copy()
    leaked.iloc[101:] = 40.0
    again, _rule2, again_info = kept_credit_regime(
        path, long_idx, long_hyg, leaked, long_ief, as_of=long_idx[100],
    )
    pd.testing.assert_series_equal(earlier, again)
    assert again_info["credit_stress"] is False
    with pytest.raises(DataError):
        kept_credit_regime(path, long_idx, long_hyg, long_lqd, long_ief, as_of=pd.Timestamp("1990-01-01"))


def test_a_kept_gate_moves_the_sleeve_to_cash_and_a_rejection_does_not(tmp_path):
    idx = _sessions(220)
    level = np.full(len(idx), 100.0)
    level[180:] = 140.0
    close = pd.Series(level, index=idx)
    ief = pd.Series(100.0, index=idx)
    lqd = pd.Series(100.0, index=idx)
    lqd.iloc[150:] = 50.0
    hyg = pd.Series(np.nan, index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "credit_regime",
        "walk_forward_picks": [["2010-01-04", "zcr_w100_b02"]],
        "variants": [{"name": "zcr_w100_b02", "window": 100, "band": 0.02}],
    }]))
    stress, rule, info = kept_credit_regime(path, idx, hyg, lqd, ief, as_of=idx[-1])
    assert rule == "zcr_w100_b02"
    assert info["credit_stress"] is True
    merged = merge_hedge((None, None, None), (stress, rule, info))
    payload = build_trend_signal(
        TrendCoreSpec(), close, generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, hedge_on=merged[0], hedge_rule=merged[1], hedge_info=merged[2],
    )
    assert payload["diagnostics"]["trend_on"] is True
    assert payload["diagnostics"]["hedge_on"] is True
    assert payload["diagnostics"]["credit_pair"] == "LQD/IEF"
    assert payload["weights"] == {"SPY": 0.5, "BIL": 0.5}
    with pytest.raises(RuntimeError):
        merge_hedge((stress, "baa", {}), (stress, rule, info))
    assert merge_hedge((stress, "baa", {"k": 1}), (None, None, None))[1] == "baa"
