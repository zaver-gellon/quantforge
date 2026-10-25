import numpy as np
import pandas as pd
import pytest

from quantforge.calendar import next_nyse_session
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal
from quantforge.strategies.dipbuy import build_dip_signal, dip_targets
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def test_default_trend_core_is_the_two_percent_band():
    spec = TrendCoreSpec()
    assert spec.band == pytest.approx(0.02)
    assert spec.window == 200
    assert spec.core_weight == pytest.approx(0.5)
    assert spec.sleeve_weight == pytest.approx(0.5)
    assert spec.sleeve_symbol == "UPRO"
    assert spec.rebalance == "quarterly"


def test_trend_signal_matches_the_bot_schema_and_ignores_the_future():
    idx = pd.bdate_range("2018-01-02", periods=400)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.4, len(idx))), index=idx)
    payload = build_trend_signal(TrendCoreSpec(), close, as_of=idx[300], generated_at="2026-01-01T00:00:00Z")
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert set(payload["weights"]) <= ALLOWED_TREND_TICKERS
    assert all(value > 0 for value in payload["weights"].values())
    assert sum(payload["weights"].values()) <= 1 + 1e-9
    assert payload["date"] == next_nyse_session(idx[300]).strftime("%Y-%m-%d")
    assert payload["strategy"] == "trend_core"
    # Rising market, SMA below price: sleeve is UPRO, core is SPY.
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["UPRO"] == pytest.approx(0.5)

    future = close.copy()
    future.iloc[301:] = 1.0
    untouched = build_trend_signal(TrendCoreSpec(), future, as_of=idx[300], generated_at="2026-01-01T00:00:00Z")
    assert untouched["weights"] == payload["weights"]
    assert untouched["diagnostics"]["trend_on"] == payload["diagnostics"]["trend_on"]


def test_hysteresis_stays_off_until_the_upper_band_and_on_until_the_lower_band():
    idx = pd.bdate_range("2019-01-02", periods=40)
    prices = np.full(len(idx), 100.0)
    close = pd.Series(prices, index=idx)
    # SMA is 100. 101 is inside the 2% band, so a flat book does not enter.
    close.iloc[-3] = 101.0
    assert bool(trend_state(close.iloc[:-2], window=20, band=0.02).iloc[-1]) is False
    entered = close.copy()
    entered.iloc[-2] = 103.0
    assert bool(trend_state(entered.iloc[:-1], window=20, band=0.02).iloc[-1]) is True
    held = entered.copy()
    held.iloc[-1] = 100.0  # back to the SMA, still above 0.98x
    assert bool(trend_state(held, window=20, band=0.02).iloc[-1]) is True
    exited = held.copy()
    exited.iloc[-1] = 97.0
    assert bool(trend_state(exited, window=20, band=0.02).iloc[-1]) is False


def test_dip_overlay_replaces_only_the_spy_half():
    idx = pd.bdate_range("2018-01-02", periods=80)
    level = np.full(len(idx), 100.0)
    level[30:] = 130.0
    level[-3:] = [129.5, 129.0, 128.5]
    close = pd.Series(level, index=idx)
    spec = TrendCoreSpec(window=20, band=0.02, freq="daily", rebalance="quarterly")
    payload = build_trend_signal(spec, close, generated_at="2026-01-01T00:00:00Z")
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert payload["diagnostics"]["trend_on"] is True
    assert payload["diagnostics"]["dip_on"] is True
    assert payload["weights"]["SSO"] == pytest.approx(0.5)
    assert payload["weights"]["UPRO"] == pytest.approx(0.5)
    assert "SPY" not in payload["weights"]
    assert sum(payload["weights"].values()) == pytest.approx(1.0)


def test_trend_band_holds_through_a_small_dip_and_monthly_does_not_flip_midmonth():
    idx = pd.bdate_range("2018-01-02", periods=260)
    prices = np.full(len(idx), 100.0)
    prices[220:] = 101.0  # a small lift, then we poke below the SMA but inside a 2% band
    close = pd.Series(prices, index=idx)
    # Build a series that is well above its SMA, then dips 1% (inside the band).
    level = pd.Series(np.linspace(80, 120, len(idx)), index=idx)
    state = trend_state(level, window=50, band=0.02, freq="daily")
    assert bool(state.iloc[-1]) is True
    dipped = level.copy()
    sma = level.rolling(50).mean().iloc[-1]
    dipped.iloc[-1] = float(sma) * 0.995  # below the SMA, still above SMA * 0.98
    held = trend_state(dipped, window=50, band=0.02, freq="daily")
    assert bool(held.iloc[-1]) is True
    exited = dipped.copy()
    exited.iloc[-1] = float(sma) * 0.97
    assert bool(trend_state(exited, window=50, band=0.02, freq="daily").iloc[-1]) is False

    # June 2019 business days: the 14th is a Friday mid-month, the 28th is the last session.
    june = pd.bdate_range("2019-04-01", "2019-06-28")
    monthly = pd.Series(np.linspace(80, 120, len(june)), index=june)
    crash_at = pd.Timestamp("2019-06-14")
    monthly.loc[crash_at:] = 10.0
    daily = trend_state(monthly, window=20, band=0.0, freq="daily")
    month = trend_state(monthly, window=20, band=0.0, freq="monthly")
    assert bool(daily.loc[crash_at]) is False
    # May's month-end state was on, and June has not ended yet on the 14th.
    assert bool(month.loc[crash_at]) is True
    assert bool(month.loc[pd.Timestamp("2019-06-28")]) is False


def test_credit_hedge_moves_only_the_sleeve_to_bil_and_ignores_a_future_print():
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(80, 120, len(idx)), index=idx)
    stress = pd.Series(False, index=idx)
    stress.iloc[-1] = True
    future = stress.copy()
    future.iloc[-1] = False
    future.iloc[10] = True
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, hedge_on=stress, hedge_rule="baa10y_d20_gt_10bp",
    )
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert payload["diagnostics"]["trend_on"] is True
    assert payload["diagnostics"]["hedge_on"] is True
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    earlier = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, hedge_on=future, hedge_rule="baa10y_d20_gt_10bp",
    )
    assert earlier["diagnostics"]["hedge_on"] is False
    assert earlier["weights"]["UPRO"] == pytest.approx(0.5)


def test_two_times_sleeve_uses_sso_and_flat_trend_uses_bil():
    idx = pd.bdate_range("2018-01-02", periods=400)
    close = pd.Series(np.linspace(100, 40, len(idx)), index=idx)
    spec = TrendCoreSpec(core_weight=0.4, leverage=2, window=50, band=0.0, freq="daily", rebalance="monthly")
    payload = build_trend_signal(
        spec, close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z", apply_dip=False,
    )
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert payload["diagnostics"]["trend_on"] is False
    assert payload["weights"]["SPY"] == pytest.approx(0.4)
    assert payload["weights"]["BIL"] == pytest.approx(0.6)
    assert "SSO" not in payload["weights"]


def test_dip_buy_enters_after_three_down_closes_and_exits_on_the_up_close():
    idx = pd.bdate_range("2020-01-02", periods=8)
    # Three down closes land on index 3. Entry is the next session. The next close is up, so the overlay exits after it.
    close = pd.Series([10, 9, 8, 7, 8, 8, 8, 8], index=idx, dtype=float)
    weights, reb = dip_targets(close, "3_down_closes", max_hold=5)
    assert weights["SPY"].iloc[2] == 1.0
    assert weights["SSO"].iloc[3] == 1.0  # decision at the third down close
    assert weights["SPY"].iloc[4] == 1.0  # up close ends it for the following session
    assert bool(reb.iloc[3]) is True
    signal = build_dip_signal(close.iloc[:4], generated_at="2026-01-01T00:00:00Z")
    validate_signal(signal, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert signal["weights"] == {"SSO": 0.5}
    assert signal["date"] == next_nyse_session(idx[3]).strftime("%Y-%m-%d")
    # Scrambling the future does not change the decision at index 3.
    scrambled = close.copy()
    scrambled.iloc[4:] = [1, 1, 1, 1]
    again, _ = dip_targets(scrambled, "3_down_closes")
    assert again.iloc[3].to_dict() == weights.iloc[3].to_dict()


def test_dip_buy_exits_after_five_sessions_without_an_up_close():
    idx = pd.bdate_range("2020-01-02", periods=12)
    close = pd.Series(np.linspace(20, 9, len(idx)), index=idx)
    weights, _reb = dip_targets(close, "3_down_closes", max_hold=5)
    # First signal at index 3 (three down steps: 0->1, 1->2, 2->3). Held SSO for the next 5 sessions' decisions.
    sso_run = weights["SSO"].iloc[3:8]
    assert sso_run.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert weights["SSO"].iloc[8] == 0.0
