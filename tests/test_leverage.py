import numpy as np
import pandas as pd
import pytest

from quantforge.engine.leverage import stitch_ohlc, synthesize_leveraged, tracking_stats


def test_close_to_close_matches_the_financing_formula():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series(100 * (1.01 ** np.arange(len(idx))), index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    rf = pd.Series(0.05 / 252, index=idx)
    out = synthesize_leveraged(open_, close, rf, leverage=3, expense_ratio=0.0091, financing_spread=0.004)
    underlying = close.pct_change()
    expected = 3 * underlying - 2 * (rf + 0.004 / 252) - 0.0091 / 252
    got = out["close"].pct_change()
    assert got.iloc[1:].to_numpy() == pytest.approx(expected.iloc[1:].to_numpy())
    # No underlying overnight gap: synthetic open equals the previous synthetic close.
    assert out["open"].iloc[1:].to_numpy() == pytest.approx(out["close"].shift(1).iloc[1:].to_numpy())


def test_overnight_gap_is_levered_and_daily_total_still_matches():
    idx = pd.bdate_range("2020-01-02", periods=4)
    close = pd.Series([100.0, 100.0, 110.0, 110.0], index=idx)
    open_ = pd.Series([100.0, 100.0, 105.0, 110.0], index=idx)
    rf = pd.Series(0.0, index=idx)
    out = synthesize_leveraged(open_, close, rf, leverage=3, expense_ratio=0.0, financing_spread=0.0)
    assert out["open"].iloc[2] / out["close"].iloc[1] - 1 == pytest.approx(3 * (105 / 100 - 1))
    assert out["close"].pct_change().iloc[2] == pytest.approx(3 * (110 / 100 - 1))


def test_stitch_uses_actual_returns_after_listing():
    idx = pd.bdate_range("2020-01-02", periods=6)
    synth_close = pd.Series(np.linspace(100, 160, len(idx)), index=idx)
    synth_open = synth_close.shift(1).fillna(synth_close.iloc[0])
    actual_close = pd.Series([np.nan, np.nan, 50.0, 55.0, 60.5, 60.5], index=idx)
    actual_open = pd.Series([np.nan, np.nan, 49.0, 52.0, 56.0, 60.0], index=idx)
    stitched = stitch_ohlc(synth_open, synth_close, actual_open, actual_close)
    assert stitched["close"].iloc[1] == pytest.approx(synth_close.iloc[1])
    actual_ret = actual_close.pct_change()
    stitched_ret = stitched["close"].pct_change()
    assert stitched_ret.iloc[3:].to_numpy() == pytest.approx(actual_ret.iloc[3:].to_numpy())
    stats = tracking_stats(synth_close, actual_close.dropna())
    assert stats["n_days"] >= 3
