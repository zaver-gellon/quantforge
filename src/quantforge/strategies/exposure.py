"""Gross exposure of a SPY + UPRO mix, and how often a path trails a benchmark.

The lead book is 50% SPY and 50% UPRO when the trend is on, which is 2x gross
equity exposure (0.5 * 1 + 0.5 * 3). These helpers only convert that gross
number into the two weights. They do not choose an exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPOSURES = (1.0, 1.5, 2.0, 2.5)
LIVE_EXPOSURE = 2.0
# Trading days. A calendar month is not a fixed bar count; 21 and 63 are the desk's windows.
ROLL_WINDOWS = {"1m": 21, "3m": 63}


def mix_weights(exposure: float) -> tuple[float, float]:
    """SPY core weight and UPRO sleeve weight for a fully invested book.

    ``core + sleeve = 1`` and ``core + 3 * sleeve = exposure``. Exposure 1 is
    100% SPY. Exposure 2 is the live 50/50 book. Exposure 3 would be 100% UPRO.
    """
    target = float(exposure)
    if not np.isfinite(target) or target < 1.0 - 1e-12 or target > 3.0 + 1e-12:
        raise ValueError("exposure must be between 1 and 3")
    core = (3.0 - target) / 2.0
    sleeve = 1.0 - core
    return float(core), float(sleeve)


def gross_exposure(core: float, sleeve: float, sleeve_leverage: float = 3.0) -> float:
    return float(core) * 1.0 + float(sleeve) * float(sleeve_leverage)


def rolling_total(returns: pd.Series, window: int) -> pd.Series:
    """Compounded simple return over the last ``window`` sessions, inclusive."""
    if window < 1:
        raise ValueError("window must be positive")
    series = pd.Series(returns).astype(float)
    log_ret = np.log1p(series.to_numpy(dtype=float))
    if np.any(~np.isfinite(log_ret) & series.notna().to_numpy()):
        raise ValueError("returns must be finite and greater than -1")
    filled = np.where(np.isfinite(log_ret), log_ret, 0.0)
    cumulative = np.cumsum(filled)
    out = np.full(len(series), np.nan)
    if len(series) >= window:
        shifted = np.concatenate([np.zeros(window), cumulative[:-window]])
        total = cumulative - shifted
        out[window - 1 :] = total[window - 1 :]
    valid = series.notna().to_numpy()
    # A window that swallowed a missing return is not a real window.
    counts = pd.Series(valid).rolling(window, min_periods=window).sum().to_numpy()
    out[counts != window] = np.nan
    return pd.Series(np.expm1(out), index=series.index)


def trail_rates(strategy: pd.Series, benchmark: pd.Series, windows: dict[str, int] | None = None) -> dict:
    """Share of rolling windows where the strategy's total return is strictly below the benchmark."""
    chosen = ROLL_WINDOWS if windows is None else windows
    left = pd.Series(strategy).astype(float).dropna().sort_index()
    right = pd.Series(benchmark).reindex(left.index).astype(float)
    both = left.index[right.notna()]
    left = left.loc[both]
    right = right.loc[both]
    out = {"n_days": int(len(left))}
    for name, window in chosen.items():
        s = rolling_total(left, int(window))
        b = rolling_total(right, int(window))
        mask = s.notna() & b.notna()
        n = int(mask.sum())
        out[name] = {
            "window": int(window),
            "n_windows": n,
            "trail_rate": float((s[mask] < b[mask]).mean()) if n else float("nan"),
        }
    return out
