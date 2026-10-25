"""Causal leverage on an existing daily return stream.

The exposure dated t is known at the close of t and scales the return of t+1.
Positive cash earns the cash yield. Exposure above 1 pays that yield plus a
daily borrow spread on the debit. Changing the exposure pays ``cost`` per unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.engine.schedule import rebalance_mask


def scale_exposure(
    returns: pd.Series,
    exposure: pd.Series,
    rf: pd.Series,
    borrow_spread: float = 0.0,
    cost: float = 0.0,
) -> pd.Series:
    """Scale ``returns`` by the previous close's exposure.

    ``borrow_spread`` and the cash yield are daily rates. ``cost`` is a fraction
    charged on the absolute change in exposure.
    """
    series = pd.Series(returns).astype(float)
    rate = pd.Series(rf).reindex(series.index).astype(float).fillna(0.0)
    held = pd.Series(exposure).reindex(series.index).astype(float).shift(1)
    prev = held.shift(1).fillna(1.0)
    turn = (held - prev).abs()
    turn = turn.where(held.notna(), 0.0)
    financing = np.maximum(held - 1.0, 0.0) * float(borrow_spread)
    excess = series - rate
    scaled = rate + held * excess - financing - turn.fillna(0.0) * float(cost)
    scaled = scaled.where(held.notna())
    return scaled.rename("return")


def half_kelly_exposure(
    returns: pd.Series,
    rf: pd.Series,
    min_obs: int = 252,
    cap: float = 3.0,
    fraction: float = 0.5,
) -> pd.Series:
    """Expanding half-Kelly, updated on month-ends and held until the next one.

    The value at a month-end uses returns through that close. Before ``min_obs``
    the exposure is 1 (the unscaled book). A non-positive edge is cash.
    """
    series = pd.Series(returns).astype(float)
    rate = pd.Series(rf).reindex(series.index).astype(float).fillna(0.0)
    excess = (series - rate).to_numpy(dtype=float)
    months = rebalance_mask(series.index, "monthly").to_numpy(dtype=bool)
    out = np.full(len(series), np.nan)
    total = 0.0
    total_sq = 0.0
    count = 0
    last = 1.0
    for i, value in enumerate(excess):
        if np.isfinite(value):
            total += value
            total_sq += value * value
            count += 1
        if not months[i]:
            continue
        if count < min_obs:
            last = 1.0
        else:
            mean = total / count
            var = total_sq / count - mean * mean
            last = 0.0 if var <= 0 or mean <= 0 else float(np.clip(fraction * mean / var, 0.0, cap))
        out[i] = last
    decided = pd.Series(out, index=series.index).ffill()
    return decided.fillna(1.0).rename("exposure")


def vol_target_exposure(
    returns: pd.Series,
    target_annual: float,
    window: int = 63,
    cap: float = 2.0,
    min_obs: int = 63,
) -> pd.Series:
    """Month-end exposure ``target / trailing vol``, capped, held until the next month-end."""
    series = pd.Series(returns).astype(float)
    vol = series.rolling(window, min_periods=min_obs).std(ddof=1) * np.sqrt(252.0)
    raw = (float(target_annual) / vol.replace(0.0, np.nan)).clip(upper=cap)
    months = rebalance_mask(series.index, "monthly")
    decided = raw.where(months).ffill()
    return decided.fillna(1.0).rename("exposure")
