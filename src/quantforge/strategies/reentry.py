"""Re-enter the unlevered core on a shorter trend while the 200-day band is off.

The live book leaves the SPY half invested for the whole time the sleeve is in
T-bills. Parking that half for the entire off-trend episode cut the drawdown
and was rejected, because the cash stake also sat out the rebound. The grid
below is the rebound rule: the core is invested while a shorter filter is on
and in T-bills only while that filter is off. The windows and bands were fixed
before this sample was scored. Do not add a cell after seeing the result.

The current book, which never parks the core, is not a cell. It is already
counted. Every shorter filter is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on either axis is a neighbor. Windows are sessions. Bands match the
# live hysteresis: enter above SMA * (1 + band), exit below SMA * (1 - band).
WINDOWS = (20, 50, 100)
BANDS = (0.0, 0.02)
OFF_NAME = "reentry_off"


def cells() -> list[tuple[int, float]]:
    return [(int(window), float(band)) for window in WINDOWS for band in BANDS]


def variant_name(window: int, band: float) -> str:
    points = int(round(float(band) * 100))
    # Zero-padded window so an alphabetical tie-break follows the window.
    return f"reentry_w{int(window):03d}_b{points}"


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "fast_reentry"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _params_for(entry: dict, name: str) -> tuple[int, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        window = row.get("window")
        band = row.get("band")
        if window is None or band is None:
            raise DataError(f"kept re-entry rule {name} has no window or band")
        found = (int(window), float(band))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept re-entry parameters {found} are outside the grid")
        return found
    raise DataError(f"kept re-entry rule has no parameters for {name}")


def kept_fast_reentry(iterations_path, close: pd.Series, as_of=None):
    """Core fraction for a kept re-entry rule, or ``(None, None)``.

    The value is the share of the core to leave invested while the 200-day
    band is off. It is 1 when the walk-forward pick is the current book or
    that pick's shorter filter is on, and 0 when the shorter filter is off.
    The series uses sessions on or before ``as_of`` only. The caller ignores
    the fraction while the 200-day band is on.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the fast re-entry rule")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    states = {
        variant_name(window, band): trend_state(visible, window, band, "daily")
        for window, band in cells()
    }
    picks = []
    for date, name in entry.get("walk_forward_picks") or []:
        decision = pd.Timestamp(date)
        if decision <= visible.index[-1]:
            picks.append((decision, name))
    picks.sort(key=lambda item: item[0])
    values = np.ones(len(visible), dtype=float)
    cursor = 0
    current_name = OFF_NAME
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current_name = picks[cursor][1]
            _params_for(entry, current_name)
            cursor += 1
        if current_name == OFF_NAME:
            values[i] = 1.0
        else:
            values[i] = 1.0 if bool(states[current_name].loc[ts]) else 0.0
    series = pd.Series(values, index=visible.index, name="reentry_hold")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    fast_on = None if params is None else bool(states[variant_name(*params)].iloc[-1])
    info = {
        "reentry_rule": chosen,
        "reentry_window": None if params is None else int(params[0]),
        "reentry_band": None if params is None else float(params[1]),
        "reentry_fast_on": fast_on,
        "reentry_live": params is None,
    }
    return series, info
