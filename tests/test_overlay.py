import numpy as np
import pandas as pd

from quantforge.engine.overlay import half_kelly_exposure, scale_exposure
from quantforge.strategies.basket import donchian_state, equal_active_weights, momentum_weights, risk_parity_weights


def test_scaling_uses_the_previous_exposure_and_charges_borrow():
    idx = pd.bdate_range("2020-01-02", periods=5)
    returns = pd.Series(0.0, index=idx)
    exposure = pd.Series(2.0, index=idx)
    rf = pd.Series(0.0, index=idx)
    scaled = scale_exposure(returns, exposure, rf, borrow_spread=0.01, cost=0.0)
    # Exposure dated t-1 scales day t. Day 1 is the first scaled day: 2x flat pays one debit of spread.
    assert scaled.iloc[1] == -0.01
    future = returns.copy()
    future.iloc[-1] = 0.5
    again = scale_exposure(future, exposure, rf, borrow_spread=0.01, cost=0.0)
    assert again.iloc[1] == scaled.iloc[1]


def test_half_kelly_does_not_use_the_future_and_stays_at_one_until_it_has_data():
    idx = pd.bdate_range("2018-01-02", periods=400)
    returns = pd.Series(0.001, index=idx)
    returns.iloc[-5:] = -0.2
    rf = pd.Series(0.0, index=idx)
    exposure = half_kelly_exposure(returns, rf, min_obs=252, cap=3.0)
    early = exposure.iloc[200]
    assert early == 1.0
    trimmed = half_kelly_exposure(returns.iloc[:-5], rf.iloc[:-5], min_obs=252, cap=3.0)
    assert exposure.iloc[-6] == trimmed.iloc[-1]


def test_equal_weight_and_momentum_and_risk_parity_are_causal():
    idx = pd.bdate_range("2020-01-02", periods=80)
    active = pd.DataFrame({"A": True, "B": False}, index=idx)
    weights = equal_active_weights(active, gross=1.0)
    assert weights.iloc[-1].to_dict() == {"A": 1.0, "B": 0.0}

    rng = np.random.default_rng(0)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.002, len(idx))))
    loud = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx))))
    noisy = pd.DataFrame({"QUIET": quiet, "LOUD": loud}, index=idx)
    parity = risk_parity_weights(noisy, vol_window=20, gross=1.0)
    assert parity["QUIET"].iloc[-1] > parity["LOUD"].iloc[-1]

    trend = pd.DataFrame(
        {"QUIET": np.full(len(idx), 100.0), "LOUD": 100 * np.exp(np.linspace(0, 0.5, len(idx)))},
        index=idx,
    )
    mom = momentum_weights(trend, lookback=40, skip=5, top_n=1, gross=1.5)
    assert mom["LOUD"].iloc[-1] == 1.5
    assert mom["QUIET"].iloc[-1] == 0.0
    scrambled = trend.copy()
    scrambled.iloc[-1] = [1.0, 1.0]
    mom_past = momentum_weights(scrambled, lookback=40, skip=5, top_n=1, gross=1.5)
    assert mom_past.iloc[-2].to_dict() == mom.iloc[-2].to_dict()


def test_donchian_enters_on_a_prior_high_and_ignores_the_future():
    idx = pd.bdate_range("2020-01-02", periods=30)
    close = pd.Series(np.full(len(idx), 10.0), index=idx)
    close.iloc[10] = 12.0
    close.iloc[20] = 8.0
    state = donchian_state(close, entry=5, exit=5)
    assert bool(state.iloc[9]) is False
    assert bool(state.iloc[10]) is True
    assert bool(state.iloc[20]) is False
    future = close.copy()
    future.iloc[-1] = 100.0
    assert bool(donchian_state(future, entry=5, exit=5).iloc[10]) is True
