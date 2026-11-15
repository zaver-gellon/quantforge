"""Weekly top-1 rotation across leveraged equity, Treasury, and gold ETFs.

Each week-ending close, rank the risky sleeves by trailing total return.
Hold the winner only when that return beats cash over the same window.
Otherwise hold cash. The pick is filled at the next open and kept until the
next week-ending close. Lookbacks and the equity sleeve are the whole grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.engine.schedule import week_ends

LOOKBACKS = (21, 42, 63)
EQUITIES = ("UPRO", "TQQQ")
BONDS = "TMF"
GOLD = "UGL"
CASH = "BIL"


def cells() -> list[tuple[str, int]]:
    return [(equity, int(lookback)) for equity in EQUITIES for lookback in LOOKBACKS]


def cell_name(equity: str, lookback: int) -> str:
    if equity not in EQUITIES:
        raise ValueError(f"unknown rotation equity {equity!r}")
    if int(lookback) not in LOOKBACKS:
        raise ValueError(f"unknown rotation lookback {lookback}")
    return f"{equity}_L{int(lookback):02d}"


def parse_cell(name: str) -> tuple[str, int]:
    for equity, lookback in cells():
        if cell_name(equity, lookback) == name:
            return equity, lookback
    raise ValueError(f"unknown rotation cell {name!r}")


def trailing_return(close: pd.Series, lookback: int) -> pd.Series:
    """Total return from the close ``lookback`` sessions ago to this close."""
    if int(lookback) < 1:
        raise ValueError("lookback must be positive")
    series = close.astype(float)
    return series / series.shift(int(lookback)) - 1.0


def cash_trailing_return(rf_daily: pd.Series, lookback: int) -> pd.Series:
    """Compounded T-bill yield over the last ``lookback`` sessions, known at this close.

    ``rf_daily`` is the lagged daily yield already used by the backtest, so the
    hurdle does not use a yield that was printed after the close.
    """
    window = int(lookback)
    if window < 1:
        raise ValueError("lookback must be positive")
    growth = (1.0 + rf_daily.astype(float)).rolling(window, min_periods=window).apply(np.prod, raw=True)
    return growth - 1.0


def weekly_pick(
    scores: dict[str, pd.Series],
    cash_score: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Asset to hold on the next session. Week-ends pick; other days carry the pick.

    Ties break alphabetically. A missing score is not eligible. If every risky
    sleeve is missing or fails to beat cash, the pick is BIL.
    """
    idx = pd.DatetimeIndex(index)
    names = sorted(scores)
    frame = pd.DataFrame({name: scores[name].reindex(idx).astype(float) for name in names})
    hurdle = cash_score.reindex(idx).astype(float)
    decisions = week_ends(idx).to_numpy()
    out = np.array([CASH] * len(idx), dtype=object)
    current = CASH
    values = frame.to_numpy(dtype=float)
    hurdles = hurdle.to_numpy(dtype=float)
    for t in range(len(idx)):
        if decisions[t]:
            eligible: list[tuple[float, str]] = []
            floor = hurdles[t]
            for j, name in enumerate(names):
                score = values[t, j]
                # A sleeve must beat cash. A missing hurdle or a missing score stays in cash.
                if np.isfinite(score) and np.isfinite(floor) and score > floor:
                    eligible.append((float(score), name))
            if not eligible:
                current = CASH
            else:
                best = max(score for score, _name in eligible)
                current = min(name for score, name in eligible if score == best)
        out[t] = current
    return pd.Series(out, index=idx, name="pick")
