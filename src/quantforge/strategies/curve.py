"""Yield-curve recession clock for the whole trend-core book.

The spread is the 10-year Treasury yield minus the 13-week bill yield, both
in percentage points. A crossing below a depth starts a fixed run of sessions
in T-bills. The depths and lengths below were fixed before this sample was
scored. Do not add a cell after seeing the result.

The live book is the cell that never leaves equities for this clock. It is
already counted. Every depth and length pair is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on either axis is a neighbor. Depths are percentage points of
# inversion. Lengths are sessions, including the session of the crossing.
DEPTHS = (0.0, 0.25, 0.50)
LENGTHS = (126, 252, 504)
OFF_NAME = "curve_off"


def cells() -> list[tuple[float, int]]:
    return [(float(depth), int(length)) for depth in DEPTHS for length in LENGTHS]


def variant_name(depth: float, length: int) -> str:
    bps = int(round(float(depth) * 100))
    return f"curve_d{bps:03d}_k{int(length)}"


def treasury_spread(tnx_close: pd.Series, irx_close: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """10-year minus 13-week, in percentage points, on the equity calendar.

    Yahoo quotes ``^TNX`` and ``^IRX`` as percent yields. A series whose
    median or 99th percentile is outside (0, 20) is not that quote (a yield
    times ten sits near 40) and is refused. Prints are carried forward at most
    five equity sessions. A later print is never used. A session with either
    yield still missing stays missing.
    """
    sessions = pd.DatetimeIndex(calendar)

    def _percent(series: pd.Series, name: str) -> pd.Series:
        clean = series.dropna().astype(float).sort_index()
        clean = clean[~clean.index.duplicated(keep="last")]
        if clean.empty:
            raise DataError(f"{name} has no yields")
        median = float(clean.median())
        tail = float(clean.quantile(0.99))
        if not (0.0 < median < 20.0 and 0.0 < tail < 20.0):
            raise DataError(
                f"{name} does not look like a percent yield "
                f"(median {median:.4f}, 99th percentile {tail:.4f}). "
                "The clock expects Yahoo ^TNX and ^IRX in percent, not a price "
                "and not a yield times ten."
            )
        return clean

    tnx = _percent(tnx_close, "^TNX").reindex(sessions).ffill(limit=5)
    irx = _percent(irx_close, "^IRX").reindex(sessions).ffill(limit=5)
    spread = (tnx - irx).astype(float)
    spread.index = sessions
    spread.name = "curve_spread_pct"
    return spread


def recession_clock(spread: pd.Series, depth: float, length: int) -> pd.Series:
    """True when the next session is in T-bills.

    A crossing is a finite spread strictly below ``-depth`` after a finite
    spread that was at or above ``-depth``. The first print, and a print that
    arrives already below the depth, is not a crossing. The close of the
    crossing starts a run of ``length`` sessions, including that close. A
    crossing while the run is underway does not extend it and does not queue
    another run. A missing spread is not a crossing and does not change the
    side of the last finite print. The run still counts that session.
    """
    level = float(depth)
    if level < 0.0:
        raise ValueError("depth must be non-negative")
    sessions = int(length)
    if sessions < 1:
        raise ValueError("length must be positive")
    values = spread.astype(float).to_numpy()
    out = np.zeros(len(values), dtype=bool)
    last_below: bool | None = None
    remaining = 0
    for i, value in enumerate(values):
        finite = bool(np.isfinite(value))
        below = bool(finite and value < -level)
        if remaining > 0:
            out[i] = True
            remaining -= 1
            if finite:
                last_below = below
            continue
        crossed = last_below is False and below
        if finite:
            last_below = below
        if crossed:
            out[i] = True
            remaining = sessions - 1
    return pd.Series(out, index=spread.index, name="curve_on")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "yield_curve"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_curve(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _params(entry: dict, name: str) -> tuple[float, int] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        if row.get("depth") is None or row.get("length") is None:
            raise DataError(f"kept yield-curve rule {name} has no depth and length")
        depth = float(row["depth"])
        length = int(row["length"])
        if depth < 0.0 or length < 1:
            raise DataError(f"kept yield-curve rule {name} is outside the grid")
        return depth, length
    raise DataError(f"kept yield-curve rule has no parameters for {name}")


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def kept_curve_clock(iterations_path, spread: pd.Series, as_of=None):
    """Cash flag for a kept clock, or ``(None, None)``.

    The flag at each session is the full-history clock of the walk-forward
    cell selected for that session. The off cell is never in cash. Sessions
    after ``as_of`` are not used.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    series = spread.astype(float).sort_index()
    if series.empty:
        raise DataError("no yield spread for the recession clock")
    stamp = series.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = series.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the yield spread")
    picks = []
    for date, name in entry.get("walk_forward_picks") or []:
        decision = pd.Timestamp(date)
        if decision <= visible.index[-1]:
            picks.append((decision, name))
    picks.sort(key=lambda item: item[0])
    clocks: dict[str, pd.Series] = {}
    values = np.zeros(len(visible), dtype=bool)
    cursor = 0
    current = OFF_NAME
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current = picks[cursor][1]
            cursor += 1
        if current not in clocks:
            params = _params(entry, current)
            if params is None:
                clocks[current] = pd.Series(False, index=visible.index)
            else:
                clocks[current] = recession_clock(visible, params[0], params[1])
        values[i] = bool(clocks[current].iloc[i])
    flag = pd.Series(values, index=visible.index, name="curve_on")
    chosen = _selection(entry, visible.index[-1])
    if chosen is None:
        chosen = OFF_NAME
    params = _params(entry, chosen)
    last_spread = float(visible.iloc[-1])
    info = {
        "curve_rule": chosen,
        "curve_depth": None if params is None else float(params[0]),
        "curve_length": None if params is None else int(params[1]),
        "curve_spread": None if not np.isfinite(last_spread) else last_spread,
        "curve_live": params is None,
    }
    return flag, info
