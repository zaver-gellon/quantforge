"""Delay only the re-entry of the UPRO sleeve.

The live book turns the sleeve on at the first close above ``SMA * (1 + 0.02)``
and off at the first close below ``SMA * (1 - 0.02)``. The 2026-10-31 test
required N consecutive closes beyond the band before either transition, so
the exit waited as well as the re-entry. The counts below were fixed before
this sample was scored. Do not add a cell after seeing the result.

``confirms=1`` is the live filter. It is already counted. Every longer count
is a new trial. The exit stays immediate. The value at t uses the close at t
and earlier bars only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on this axis is a neighbor. 1 is the live book and is not a new trial.
CONFIRMS = (2, 3, 5, 8, 10)
LIVE_CONFIRM = 1
AXIS = (LIVE_CONFIRM, *CONFIRMS)
OFF_NAME = "re_live"
WINDOW = 200
BAND = 0.02


def cells() -> list[int]:
    """New trials. The immediate re-entry is not in this list."""
    return [int(count) for count in CONFIRMS]


def variant_name(confirms: int) -> str:
    count = int(confirms)
    if count not in AXIS:
        raise ValueError(f"unknown re-entry count {confirms}")
    return f"zre_n{count:02d}"


def reentry_trend_state(
    close: pd.Series,
    window: int = WINDOW,
    band: float = BAND,
    confirms: int = LIVE_CONFIRM,
) -> pd.Series:
    """True when the next session should be long the risky sleeve.

    The sleeve turns off on the first close below ``SMA * (1 - band)``. It
    turns on only after ``confirms`` consecutive closes above
    ``SMA * (1 + band)``. A close that is not above that upper band resets
    the entry count. Inside the band the previous state is kept. The book
    stays off until the first completed entry. Before the SMA exists the
    state is off. ``confirms=1`` is the live hysteresis rule.
    """
    lookback = int(window)
    if lookback < 2:
        raise ValueError("window must be at least 2")
    width = float(band)
    if width < 0.0:
        raise ValueError("band must be non-negative")
    needed = int(confirms)
    if needed < 1:
        raise ValueError("confirms must be at least 1")
    series = close.astype(float).sort_index()
    average = series.rolling(lookback, min_periods=lookback).mean()
    prices = series.to_numpy(dtype=float)
    means = average.to_numpy(dtype=float)
    state = False
    pending = 0
    out = np.zeros(len(series), dtype=bool)
    for i in range(len(series)):
        finite = np.isfinite(means[i]) and np.isfinite(prices[i])
        if not finite:
            if not state:
                pending = 0
        else:
            upper = prices[i] > means[i] * (1.0 + width)
            lower = prices[i] < means[i] * (1.0 - width)
            if state:
                pending = 0
                if lower:
                    state = False
            elif upper:
                pending += 1
                if pending >= needed:
                    state = True
                    pending = 0
            else:
                pending = 0
        out[i] = state
    return pd.Series(out, index=series.index, name="trend_on")


def matches_live_filter(close: pd.Series, window: int = WINDOW, band: float = BAND) -> bool:
    """``confirms=1`` is ``trend_state``, bar for bar."""
    delayed = reentry_trend_state(close, window, band, LIVE_CONFIRM)
    live = trend_state(close, window, band, "daily")
    return bool(delayed.equals(live.reindex(delayed.index)))


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "reentry_confirm"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, int | None]:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == OFF_NAME:
        return None, None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        count = row.get("confirms")
        if count is None:
            raise DataError(f"kept re-entry rule {chosen} has no count")
        needed = int(count)
        if needed == LIVE_CONFIRM:
            return None, None
        if needed not in CONFIRMS:
            raise DataError(f"kept re-entry count {needed} is outside the grid")
        return chosen, needed
    raise DataError(f"kept re-entry rule has no parameters for {chosen}")


def kept_reentry_confirm(iterations_path, close: pd.Series, as_of=None):
    """Delayed re-entry state for a kept rule, or ``(None, None)``.

    The count is the walk-forward pick for ``as_of``. The state is built from
    closes on or before ``as_of`` only. The live cell is not an override.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the re-entry delay")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, needed = _selection(entry, visible.index[-1])
    if rule is None or needed is None:
        return None, None
    state = reentry_trend_state(visible, WINDOW, BAND, needed)
    info = {
        "reentry_rule": rule,
        "reentry_confirms": int(needed),
        "reentry_window": WINDOW,
        "reentry_band": BAND,
        "reentry_on": bool(state.iloc[-1]),
    }
    return state, info
