"""Hold the UPRO sleeve in SPY when a shorter trend is off.

The live book keeps the 3x sleeve on until SPY closes through the 200-day
band. Exits that replace the sleeve with T-bills were rejected: the cash
stake also sat out the rebound. This grid does not sell the sleeve into
T-bills and does not touch the core. While the 200-day band is on, the
sleeve is UPRO only when a shorter filter is on, and SPY when that filter
is off. While the 200-day band is off, the sleeve stays in T-bills and the
shorter filter is ignored.

The windows and bands were fixed before this sample was scored. Do not add
a cell after seeing the result. The current book, which never moves the
sleeve from UPRO to SPY, is not a cell. It is already counted. Every shorter
filter is a new trial.
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
OFF_NAME = "gear_off"


def cells() -> list[tuple[int, float]]:
    return [(int(window), float(band)) for window in WINDOWS for band in BANDS]


def variant_name(window: int, band: float) -> str:
    points = int(round(float(band) * 100))
    # Zero-padded window so an alphabetical tie-break follows the window.
    return f"gear_w{int(window):03d}_b{points}"


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "sleeve_gear"
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
            raise DataError(f"kept sleeve-gear rule {name} has no window or band")
        found = (int(window), float(band))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept sleeve-gear parameters {found} are outside the grid")
        return found
    raise DataError(f"kept sleeve-gear rule has no parameters for {name}")


def kept_sleeve_gear(iterations_path, close: pd.Series, as_of=None):
    """Fast-trend flag for a kept sleeve gear, or ``(None, None)``.

    True means the sleeve may be UPRO while the 200-day band is on. False
    means that sleeve is SPY. The series is 1 for every session when the
    walk-forward pick is the current book. The caller ignores the flag while
    the 200-day band is off. Sessions after ``as_of`` are not used.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the sleeve-gear rule")
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
    values = np.ones(len(visible), dtype=bool)
    cursor = 0
    current_name = OFF_NAME
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current_name = picks[cursor][1]
            _params_for(entry, current_name)
            cursor += 1
        if current_name == OFF_NAME:
            values[i] = True
        else:
            values[i] = bool(states[current_name].loc[ts])
    series = pd.Series(values, index=visible.index, name="sleeve_fast")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    fast_on = None if params is None else bool(states[variant_name(*params)].iloc[-1])
    info = {
        "sleeve_gear_rule": chosen,
        "sleeve_gear_window": None if params is None else int(params[0]),
        "sleeve_gear_band": None if params is None else float(params[1]),
        "sleeve_gear_fast_on": fast_on,
        "sleeve_gear_live": params is None,
    }
    return series, info
