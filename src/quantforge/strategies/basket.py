"""Cross-asset weights. Every weight at t uses closes through t and is traded next session."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.strategies.trend import trend_state


def trend_panel(closes: pd.DataFrame, window: int, band: float = 0.0, freq: str = "monthly") -> pd.DataFrame:
    """Boolean trend, with NA until the moving average exists. NA is not an off signal."""
    columns = {}
    for name in closes.columns:
        series = closes[name].dropna()
        state = trend_state(series, window, band, freq)
        ready = series.rolling(window, min_periods=window).mean().notna()
        columns[name] = state.where(ready)
    return pd.DataFrame(columns).reindex(closes.index)


def equal_active_weights(active: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Equal weight among the active columns. Inactive and unknown names are zero."""
    flags = active.fillna(False).astype(float)
    count = flags.sum(axis=1)
    weights = flags.mul(float(gross))
    weights = weights.div(count.where(count > 0), axis=0).fillna(0.0)
    return weights


def long_short_weights(active: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Equal absolute weight. Active is long, inactive is short, unknown is flat."""
    known = active.notna()
    sign = active.fillna(False).astype(float) * 2.0 - 1.0
    sign = sign.where(known, 0.0)
    count = known.sum(axis=1).astype(float)
    return sign.mul(float(gross)).div(count.where(count > 0), axis=0).fillna(0.0)


def risk_parity_weights(
    closes: pd.DataFrame,
    vol_window: int = 63,
    gross: float = 1.0,
    trend: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inverse-vol weights from the trailing window, optionally zeroed when the trend is off."""
    vol = closes.pct_change().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    inverse = 1.0 / vol.replace(0.0, np.nan)
    if trend is not None:
        inverse = inverse.where(trend.reindex(inverse.index).fillna(False))
    inverse = inverse.replace([np.inf, -np.inf], np.nan)
    total = inverse.sum(axis=1)
    weights = inverse.div(total.where(total > 0), axis=0).fillna(0.0) * float(gross)
    return weights


def momentum_weights(
    closes: pd.DataFrame,
    lookback: int,
    skip: int,
    top_n: int,
    gross: float = 1.0,
    absolute: bool = True,
) -> pd.DataFrame:
    """Equal weight in the top names by total return from ``lookback`` ago to ``skip`` ago.

    Names with a non-positive score are skipped when ``absolute`` is set. The
    score at t uses closes at t-skip and t-lookback only.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    momentum = closes.shift(skip) / closes.shift(lookback) - 1.0
    values = momentum.to_numpy(dtype=float)
    out = np.zeros(momentum.shape, dtype=float)
    for i in range(len(closes)):
        row = values[i]
        order = np.argsort(np.where(np.isfinite(row), -row, np.inf))
        picked = []
        for col in order:
            score = row[col]
            if not np.isfinite(score):
                continue
            if absolute and score <= 0:
                continue
            picked.append(col)
            if len(picked) >= top_n:
                break
        if picked:
            share = float(gross) / len(picked)
            for col in picked:
                out[i, col] = share
    return pd.DataFrame(out, index=closes.index, columns=closes.columns)


def donchian_state(close: pd.Series, entry: int = 55, exit: int = 20) -> pd.Series:
    """Long after a close above the prior ``entry``-day high, flat after a close below the prior ``exit``-day low."""
    series = close.astype(float)
    prior_high = series.shift(1).rolling(entry, min_periods=entry).max()
    prior_low = series.shift(1).rolling(exit, min_periods=exit).min()
    state = False
    out = np.zeros(len(series), dtype=bool)
    prices = series.to_numpy(dtype=float)
    highs = prior_high.to_numpy(dtype=float)
    lows = prior_low.to_numpy(dtype=float)
    for i in range(len(series)):
        if np.isfinite(highs[i]) and prices[i] > highs[i]:
            state = True
        elif state and np.isfinite(lows[i]) and prices[i] < lows[i]:
            state = False
        out[i] = state
    return pd.Series(out, index=series.index, name="breakout")
