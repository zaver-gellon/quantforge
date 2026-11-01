"""Park part of the SPY core in T-bills while the 200-day band is off.

The live book turns only the UPRO sleeve to cash when SPY leaves its 200-day
band. The SPY half stays invested, and that residual equity is the drawdown
that remains after the sleeve is already in cash. The fractions below were
fixed before this sample was scored. Do not add a cell after seeing the result.

``hold=1`` is that live book. It is already counted. Every smaller fraction is
a new trial. While the band is on, the fraction is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on this axis is a neighbor. 1.0 is the current book.
HOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)
LIVE_HOLD = 1.0


def cells() -> list[tuple[float]]:
    return [(float(hold),) for hold in HOLDS]


def is_live_cell(hold: float) -> bool:
    """The current book: the core stays fully invested when the trend is off."""
    return abs(float(hold) - LIVE_HOLD) <= 1e-12


def variant_name(hold: float) -> str:
    points = int(round(float(hold) * 100))
    # Zero-padded so an alphabetical tie-break follows the fraction.
    return f"corehold_f{points:03d}"


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "core_hold"
    ]
    if not kept:
        return None
    return kept[-1]


def _fraction(entry: dict, name: str | None) -> float:
    if name is None:
        return LIVE_HOLD
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        if row.get("hold") is None:
            continue
        hold = float(row["hold"])
        if hold < -1e-12 or hold > 1.0 + 1e-12:
            raise DataError(f"kept core-hold fraction {hold} is outside [0, 1]")
        return min(1.0, max(0.0, hold))
    raise DataError(f"kept core-hold rule has no fraction for {name}")


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def kept_core_hold(iterations_path, close: pd.Series, as_of=None):
    """Off-trend core fraction for a kept rule, or ``(None, None)``.

    The value at each session is the walk-forward pick for that session. The
    live cell is 1. The series uses sessions on or before ``as_of`` only. The
    caller applies the fraction only while the equity trend is off.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the core-hold rule")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    picks = []
    for date, name in entry.get("walk_forward_picks") or []:
        decision = pd.Timestamp(date)
        if decision <= visible.index[-1]:
            picks.append((decision, name))
    picks.sort(key=lambda item: item[0])
    values = np.ones(len(visible), dtype=float)
    cursor = 0
    current = LIVE_HOLD
    current_name = None
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current_name = picks[cursor][1]
            current = _fraction(entry, current_name)
            cursor += 1
        values[i] = current
    series = pd.Series(values, index=visible.index, name="core_hold")
    chosen = _selection(entry, visible.index[-1])
    info = {
        "core_hold_rule": chosen,
        "core_hold_fraction": float(series.iloc[-1]),
        "core_hold_live": bool(is_live_cell(series.iloc[-1])),
    }
    return series, info
