"""TQQQ while QQQ is in a 200-day uptrend, sized to a realized-vol target.

The weight is ``min(1, target / 20-day realized vol of TQQQ)`` when the trend
is on, and zero otherwise. The cap is 100% of the book: the rule does not
borrow to reach the target. The rest is cash. The weight at t uses the TQQQ
return that ends at t and the QQQ trend state at t, and it is the weight for
the next session. Bands and targets below are the whole grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.strategies.trend import trend_state

BANDS = (0.0, 0.02)
TARGETS = (0.20, 0.30, 0.40)
VOL_WINDOW = 20
TREND_WINDOW = 200


def cells() -> list[tuple[float, float]]:
    return [(float(band), float(target)) for band in BANDS for target in TARGETS]


def cell_name(band: float, target: float) -> str:
    if not any(abs(float(band) - item) < 1e-12 for item in BANDS):
        raise ValueError(f"unknown Nasdaq band {band}")
    if not any(abs(float(target) - item) < 1e-12 for item in TARGETS):
        raise ValueError(f"unknown vol target {target}")
    return f"band{int(round(float(band) * 100)):02d}_vol{int(round(float(target) * 100)):02d}"


def parse_cell(name: str) -> tuple[float, float]:
    for band, target in cells():
        if cell_name(band, target) == name:
            return band, target
    raise ValueError(f"unknown Nasdaq-vol cell {name!r}")


def realized_vol(close: pd.Series, window: int = VOL_WINDOW, periods_per_year: int = 252) -> pd.Series:
    """Annualized sample standard deviation of the last ``window`` close-to-close returns."""
    returns = close.astype(float).pct_change()
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def tqqq_weight(
    qqq_close: pd.Series,
    tqqq_close: pd.Series,
    band: float,
    target: float,
    vol_window: int = VOL_WINDOW,
) -> pd.Series:
    """TQQQ weight for the next session. The residual is cash. Never above 1."""
    if float(target) <= 0.0:
        raise ValueError("vol target must be positive")
    state = trend_state(qqq_close.astype(float), TREND_WINDOW, float(band), "daily")
    vol = realized_vol(tqqq_close, vol_window).reindex(state.index)
    raw = float(target) / vol
    weight = raw.clip(lower=0.0, upper=1.0)
    allowed = state.astype(bool) & vol.notna() & (vol > 0.0) & np.isfinite(vol)
    weight = weight.where(allowed, 0.0)
    return weight.fillna(0.0).astype(float)
