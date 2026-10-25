"""Two ways to scale gross exposure. Both are known at the close.

The volatility/trend rule is the default because it has no estimated
transition matrix. The two-state Gaussian HMM is the alternative. Its
parameters are fit on an expanding window at each refit date and then frozen.
The probability used on later days is the forward filter, so day t does not
see day t+1. Smoothed (forward-backward) probabilities are used only inside
the fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_trend_scale(
    close: pd.Series,
    vol_window: int = 21,
    trend_window: int = 200,
    calm: float = 1.0,
    hot: float = 0.60,
    down: float = 0.50,
) -> pd.Series:
    """1 when the trend is up and vol is below its expanding median.

    Hot vol with an uptrend uses ``hot``. A downtrend uses ``down``. Until
    the trend and the vol median both exist, the multiplier is 1.
    """
    series = close.astype(float)
    average = series.rolling(trend_window, min_periods=trend_window).mean()
    vol = series.pct_change().rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252.0)
    median = vol.expanding(min_periods=252).median()
    scale = pd.Series(1.0, index=series.index)
    ready = average.notna() & median.notna() & vol.notna()
    up = series > average
    quiet = vol <= median
    scale = scale.where(~(ready & up & ~quiet), hot)
    scale = scale.where(~(ready & ~up), down)
    scale = scale.where(~(ready & up & quiet), calm)
    return scale.rename("regime_scale")


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    total = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis)


def _emission_logp(obs: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    logp = np.empty((len(obs), len(means)))
    for k in range(len(means)):
        diff = obs - means[k]
        logp[:, k] = -0.5 * np.sum(diff ** 2 / var[k] + np.log(2.0 * np.pi * var[k]), axis=1)
    return logp


def fit_gaussian_hmm(obs: np.ndarray, n_iter: int = 15) -> dict:
    """Two-state diagonal Gaussian HMM, initialized by a vol split."""
    x = np.asarray(obs, dtype=float)
    if len(x) < 30:
        raise ValueError("HMM fit needs at least 30 observations")
    n_states = 2
    labels = (x[:, -1] >= np.median(x[:, -1])).astype(int)
    means = np.zeros((n_states, x.shape[1]))
    var = np.ones((n_states, x.shape[1]))
    for state in range(n_states):
        chunk = x[labels == state]
        if len(chunk) < 5:
            chunk = x
        means[state] = chunk.mean(axis=0)
        var[state] = chunk.var(axis=0) + 1e-3
    start = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    for _ in range(n_iter):
        gamma, xi = _e_step(x, start, trans, means, var)
        start = gamma[0]
        trans = xi.sum(axis=0)
        trans = trans / np.clip(trans.sum(axis=1, keepdims=True), 1e-12, None)
        for state in range(n_states):
            weight = gamma[:, state]
            total = float(weight.sum())
            if total < 1e-8:
                continue
            means[state] = (weight[:, None] * x).sum(axis=0) / total
            diff = x - means[state]
            var[state] = (weight[:, None] * diff ** 2).sum(axis=0) / total + 1e-3
    return {"start": start, "trans": trans, "means": means, "var": var}


def _e_step(obs, start, trans, means, var):
    log_b = _emission_logp(obs, means, var)
    log_a = np.log(trans + 1e-300)
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = np.log(start + 1e-300) + log_b[0]
    for t in range(1, len(obs)):
        mixed = log_alpha[t - 1][:, None] + log_a
        log_alpha[t] = _logsumexp(mixed, axis=0) + log_b[t]
    log_beta = np.zeros_like(log_b)
    for t in range(len(obs) - 2, -1, -1):
        mixed = log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :]
        log_beta[t] = _logsumexp(mixed, axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    xi = np.zeros((len(obs) - 1, len(start), len(start)))
    for t in range(len(obs) - 1):
        mixed = log_alpha[t][:, None] + log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :]
        mixed -= _logsumexp(mixed.ravel(), axis=0)
        xi[t] = np.exp(mixed)
    return gamma, xi


def filter_state_prob(obs: np.ndarray, params: dict, state: int) -> np.ndarray:
    """Forward probability of ``state`` given observations through each date."""
    x = np.asarray(obs, dtype=float)
    log_b = _emission_logp(x, params["means"], params["var"])
    log_a = np.log(params["trans"] + 1e-300)
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = np.log(params["start"] + 1e-300) + log_b[0]
    for t in range(1, len(x)):
        mixed = log_alpha[t - 1][:, None] + log_a
        log_alpha[t] = _logsumexp(mixed, axis=0) + log_b[t]
    # Normalize each row in probability space.
    peak = np.max(log_alpha, axis=1, keepdims=True)
    prob = np.exp(log_alpha - peak)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-300, None)
    return prob[:, state]


def hmm_scale(
    close: pd.Series,
    refit_dates,
    low: float = 1.0,
    mid: float = 0.75,
    high: float = 0.50,
    min_obs: int = 252,
) -> pd.Series:
    """Map the filtered high-vol probability into a gross multiplier.

    Parameters fit at a refit date are applied until the next refit. The
    high-vol state is the one whose fitted mean volatility is higher.
    """
    series = close.astype(float)
    returns = series.pct_change()
    vol = returns.rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252.0)
    obs = pd.DataFrame({"ret": returns, "vol": vol}).dropna()
    scale = pd.Series(1.0, index=series.index, name="hmm_scale")
    refits = [pd.Timestamp(d) for d in refit_dates if pd.Timestamp(d) >= obs.index[0]]
    if not refits:
        return scale
    for i, start in enumerate(refits):
        train = obs.loc[:start]
        if len(train) < min_obs:
            continue
        mu = train.mean()
        sd = train.std(ddof=1).replace(0.0, 1.0)
        params = fit_gaussian_hmm(((train - mu) / sd).to_numpy(), n_iter=12)
        high_state = int(np.argmax(params["means"][:, 1]))
        end = refits[i + 1] if i + 1 < len(refits) else obs.index[-1]
        window = obs.loc[:end]
        standardized = ((window - mu) / sd).to_numpy()
        prob = filter_state_prob(standardized, params, high_state)
        prob_s = pd.Series(prob, index=window.index)
        segment = prob_s.loc[start:end]
        mapped = pd.Series(low, index=segment.index)
        mapped = mapped.where(segment < 0.45, mid)
        mapped = mapped.where(segment < 0.67, high)
        # The refit date's probability uses data through that date. Keep it.
        # Do not overwrite an earlier segment's last point with a refit that
        # has seen it; writing the segment [start, end] is the frozen-parameter path.
        scale.loc[segment.index] = mapped
    return scale
