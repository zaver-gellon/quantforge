"""Vol-sleeve timing. No market data."""

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_trend_core, run_vol_sleeve
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.volscale import sleeve_scale


def _prices(n=40, start=100.0, step=0.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = pd.Series(start + step * np.arange(n), index=idx, dtype=float)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    return idx, open_, close


def test_scale_does_not_read_a_future_close():
    idx = pd.bdate_range("2019-01-02", periods=80)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)
    base = sleeve_scale(close, 10, 0.15)
    future = close.copy()
    future.iloc[40:] = close.iloc[40] * 0.5
    again = sleeve_scale(future, 10, 0.15)
    pd.testing.assert_series_equal(base.iloc[:40], again.iloc[:40])
    assert float(base.iloc[5]) == 1.0  # window is not full yet


def test_full_scale_matches_the_unscaled_trend_book():
    idx, open_, close = _prices(120, step=0.2)
    # A dip and a trend flip, plus one mix date, so the allocator is not a straight line.
    trend = pd.Series(True, index=idx)
    trend.iloc[40:55] = False
    dip = pd.Series(False, index=idx)
    dip.iloc[70:80] = True
    mix = pd.Series(False, index=idx)
    mix.iloc[30] = True
    mix.iloc[90] = True
    rf = pd.Series(0.0001, index=idx)
    sso_c = close * 1.5
    upro_c = close * 2.0
    sso_o = open_ * 1.5
    upro_o = open_ * 2.0
    kwargs = dict(
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    plain = run_trend_core(
        open_, close, sso_o, sso_c, upro_o, upro_c,
        trend, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    scaled = run_vol_sleeve(
        open_, close, sso_o, sso_c, upro_o, upro_c,
        trend, dip, mix, pd.Series(1.0, index=idx), 0.5, 0.5, rf, **kwargs,
    )
    pd.testing.assert_series_equal(plain.returns, scaled.returns, check_names=False)


def test_scale_known_at_the_close_applies_on_the_next_session():
    idx, open_, close = _prices(8)
    # Open equals the previous close, so the whole move is open-to-close.
    # Every asset rises 1% from open to close.
    close = pd.Series(100.0 * (1.01 ** np.arange(len(idx))), index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    scale = pd.Series(1.0, index=idx)
    scale.iloc[3] = 0.0
    result = run_vol_sleeve(
        open_, close, open_, close, open_, close,
        trend, dip, mix, scale, 0.5, 0.5, rf,
    )
    # scale[3] is the decision at that close. It changes the holding on session 4.
    assert result.returns.iloc[3] == pytest.approx(0.01)
    assert result.returns.iloc[4] == pytest.approx(0.005)
    assert result.returns.iloc[5] == pytest.approx(0.01)


def test_signal_scales_only_the_sleeve_and_ignores_a_future_scale():
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(80, 120, len(idx)), index=idx)
    scale = pd.Series(1.0, index=idx)
    scale.iloc[-1] = 0.4
    future = scale.copy()
    future.iloc[-1] = 1.0
    future.iloc[20] = 0.0
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, sleeve_scale=scale, vol_info={"vol_rule": "vol_lb21_tv15"},
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["UPRO"] == pytest.approx(0.2)
    assert payload["weights"]["BIL"] == pytest.approx(0.3)
    assert payload["diagnostics"]["sleeve_scale"] == pytest.approx(0.4)
    earlier = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, sleeve_scale=future, vol_info={"vol_rule": "vol_lb21_tv15"},
    )
    assert earlier["weights"]["UPRO"] == pytest.approx(0.5)
    assert "BIL" not in earlier["weights"]
