"""Short-horizon mean-reversion trigger for the SPY half of the trend core.

The live overlay enters after three strictly down closes and leaves at the
first up close or after five sessions. This rule replaces that trigger and
that exit. It does not replace the 50/50 book, the 200-day 2% sleeve, or the
SSO vehicle. The standalone mean-reversion account and the RSI(2) boost are
different books and are not these cells.

Enter at the close when Wilder RSI(2) of SPY is strictly below 5, 10, or 15.
Leave at the close when SPY is strictly above its 5-day average, or when
RSI(2) is strictly above 70. The exit close does not re-enter. There is no
session cap and no 200-day gate on the overlay. The value at t is the book
for the next session and uses the close at t and earlier bars only.

The thresholds were fixed before this sample was scored. Do not add a cell
after seeing the result. The three-down overlay is the current book. It is
already counted and is not a cell on this axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.meanrev import wilder_rsi

# One step on this axis is a neighbor. 5 is stricter than 10, which is
# stricter than 15. The live three-down rule is not on the axis.
THRESHOLDS = (5, 10, 15)
EXIT_RSI = 70.0
EXIT_SMA = 5
RSI_PERIOD = 2
OFF_NAME = "dmr_off"
LIVE_RULE = "3_down_closes"
LIVE_HOLD = 5


def cells() -> list[int]:
    """New trials. The live three-down trigger is not in this list."""
    return [int(threshold) for threshold in THRESHOLDS]


def variant_name(threshold: int) -> str:
    level = int(threshold)
    if level not in THRESHOLDS:
        raise ValueError(f"unknown dip mean-reversion threshold {threshold!r}")
    return f"zdmr_rsi{level:02d}"


def parse_threshold(name: str) -> int:
    for threshold in cells():
        if variant_name(threshold) == name:
            return threshold
    raise ValueError(f"unknown dip mean-reversion cell {name!r}")


def dip_mr_flag(close: pd.Series, threshold: int) -> pd.Series:
    """True when the SPY half should be SSO on the next session.

    Entry is Wilder RSI(2) strictly below ``threshold``. Exit is a close
    strictly above the 5-day average, or RSI(2) strictly above 70. A same-close
    exit is not re-armed. Missing RSI or average values do not enter and do
    not count as an exit.
    """
    level = int(threshold)
    if level not in THRESHOLDS:
        raise ValueError(f"unknown dip mean-reversion threshold {threshold!r}")
    series = close.astype(float).sort_index()
    rsi = wilder_rsi(series, RSI_PERIOD).reindex(series.index)
    average = series.rolling(EXIT_SMA, min_periods=EXIT_SMA).mean()
    entry = (rsi < float(level)).fillna(False).to_numpy(dtype=bool)
    hot = (rsi > EXIT_RSI).fillna(False).to_numpy(dtype=bool)
    above = (series > average).fillna(False).to_numpy(dtype=bool)
    n = len(series)
    out = np.zeros(n, dtype=bool)
    in_trade = False
    for t in range(n):
        if not in_trade:
            nxt = bool(entry[t])
            in_trade = nxt
        else:
            if bool(hot[t] or above[t]):
                in_trade = False
                nxt = False
            else:
                nxt = True
        out[t] = nxt
    return pd.Series(out, index=series.index, name="dip_mr")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "dip_mr"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> int | None:
    """Threshold in force on ``as_of``, or None when the pick is the live dip."""
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        threshold = row.get("threshold")
        if threshold is None:
            raise DataError(f"kept dip mean-reversion rule {chosen} has no threshold")
        level = int(threshold)
        if level not in THRESHOLDS or variant_name(level) != chosen:
            raise DataError(f"kept dip mean-reversion cell {chosen} is outside the grid")
        return level
    raise DataError(f"kept dip mean-reversion rule has no parameters for {chosen}")


def kept_dip_mr(iterations_path, close: pd.Series, as_of=None):
    """SSO flag for a kept RSI trigger, or ``(None, None)``.

    The cell is the walk-forward pick in force on ``as_of``. The flag uses
    closes on or before ``as_of`` only. A pick of the three-down book, and a
    rejected iteration, return ``(None, None)``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the dip mean-reversion trigger")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    threshold = _selection(entry, visible.index[-1])
    if threshold is None:
        return None, None
    flag = dip_mr_flag(visible, threshold)
    # A later close must not change the flag through this session.
    info = {
        "dip_mr_rule": variant_name(threshold),
        "dip_mr_threshold": int(threshold),
        "dip_mr_exit_rsi": EXIT_RSI,
        "dip_mr_exit_sma": EXIT_SMA,
        "dip_mr_live": False,
    }
    return flag, info


def live_dip_flag(close: pd.Series) -> pd.Series:
    """The current three-down overlay. Used to show a cell is not that book."""
    return dip_active(close, LIVE_RULE, LIVE_HOLD)
