"""Bitcoin trend: long when the fast average is above the slow average, else cash.

The named rule is 20/50. The two neighbors exist so the materiality bar can
see whether that cell is an isolated peak. This is a daily rule on the NYSE
calendar the study aligns the coin to. It is not an intraday rule.
"""

from __future__ import annotations

import pandas as pd

PAIRS = ((10, 30), (20, 50), (50, 200))
NAMED = (20, 50)


def cells() -> list[tuple[int, int]]:
    return [(int(fast), int(slow)) for fast, slow in PAIRS]


def cell_name(fast: int, slow: int) -> str:
    pair = (int(fast), int(slow))
    if pair not in PAIRS:
        raise ValueError(f"unknown BTC trend pair {pair}")
    return f"sma{int(fast):02d}_{int(slow):03d}"


def parse_cell(name: str) -> tuple[int, int]:
    for fast, slow in cells():
        if cell_name(fast, slow) == name:
            return fast, slow
    raise ValueError(f"unknown BTC trend cell {name!r}")


def trend_on(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """True when the fast average is strictly above the slow average.

    Both averages use the close at t and earlier bars. The flag at t is the
    position for the next session. Before either average exists the flag is off.
    """
    if int(fast) < 1 or int(slow) < 1:
        raise ValueError("windows must be positive")
    if int(fast) >= int(slow):
        raise ValueError("the fast window must be shorter than the slow window")
    series = close.astype(float).sort_index()
    fast_avg = series.rolling(int(fast), min_periods=int(fast)).mean()
    slow_avg = series.rolling(int(slow), min_periods=int(slow)).mean()
    return (fast_avg > slow_avg).fillna(False)
