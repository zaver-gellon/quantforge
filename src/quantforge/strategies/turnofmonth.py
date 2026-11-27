"""Turn-of-month tilt on the UPRO sleeve of the trend core.

The live book is half SPY (or SSO while the three-down dip is on) and half
UPRO while the 200-day 2% band is on. This rule does not replace that filter.
While the band is on, and the next session is one of the last N NYSE sessions
of its month or one of the first M, the target weights become 30% core and
70% UPRO. The 20 percentage points come out of the core, so the book stays
fully invested. While the band is off the targets stay 50/50 and the sleeve
stays in cash. The window is a calendar fact. It does not use prices.

N and M are each 1, 2, or 3. That grid was fixed before this sample was
scored. Do not add a cell after seeing the result. The un-tilted book is the
current book. It is already counted and is not a cell on these axes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

from quantforge.calendar import next_nyse_session, nyse_holidays
from quantforge.data.base import DataError

# One step on either axis is a neighbor. 1 is the shortest window, 3 the longest.
LASTS = (1, 2, 3)
FIRSTS = (1, 2, 3)
BASE_CORE = 0.50
BASE_SLEEVE = 0.50
RAISED_CORE = 0.30
RAISED_SLEEVE = 0.70
OFF_NAME = "tom_off"

_SESSIONS: pd.DatetimeIndex | None = None


def cells() -> list[tuple[int, int]]:
    """New trials. The live 50/50 book is not in this list."""
    return [(int(n_last), int(m_first)) for n_last in LASTS for m_first in FIRSTS]


def variant_name(n_last: int, m_first: int) -> str:
    pair = (int(n_last), int(m_first))
    if pair not in cells():
        raise ValueError(f"unknown turn-of-month cell {pair!r}")
    return f"ztom_n{pair[0]}_m{pair[1]}"


def parse_cell(name: str) -> tuple[int, int]:
    for n_last, m_first in cells():
        if variant_name(n_last, m_first) == name:
            return n_last, m_first
    raise ValueError(f"unknown turn-of-month cell {name!r}")


def _sessions() -> pd.DatetimeIndex:
    global _SESSIONS
    if _SESSIONS is None:
        holidays = nyse_holidays()
        _SESSIONS = pd.DatetimeIndex(
            pd.bdate_range("1990-01-01", "2040-12-31", freq=CustomBusinessDay(holidays=holidays))
        ).normalize()
    return _SESSIONS


def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    return stamp.normalize()


def window_sessions(n_last: int, m_first: int, start, end) -> set[pd.Timestamp]:
    """NYSE sessions in the last ``n_last`` or first ``m_first`` of their month.

    The month is taken from the holiday calendar, which is known before the
    close. A half-day is a session. Rank uses every calendar session in the
    month, including ones the price file has not reached yet.
    """
    n_last = int(n_last)
    m_first = int(m_first)
    if (n_last, m_first) not in cells():
        raise ValueError(f"unknown turn-of-month cell {(n_last, m_first)!r}")
    first = _stamp(start).to_period("M")
    last = _stamp(end).to_period("M")
    sessions = _sessions()
    periods = sessions.to_period("M")
    found: set[pd.Timestamp] = set()
    for period in pd.period_range(first, last, freq="M"):
        group = sessions[periods == period]
        count = len(group)
        for offset in range(min(m_first, count)):
            found.add(pd.Timestamp(group[offset]).normalize())
        for offset in range(min(n_last, count)):
            found.add(pd.Timestamp(group[-(offset + 1)]).normalize())
    return found


def turn_of_month_flag(index, n_last: int, m_first: int) -> pd.Series:
    """True at close t when the session that follows t should carry the tilt.

    For every bar but the last, the following session is the next date in
    ``index``. That is the bar whose return the backtest earns with this
    decision. The last bar has no following price, so it uses the next NYSE
    session, which is the session a published signal is for. The value does
    not read prices.
    """
    source = index if isinstance(index, pd.Index) else pd.Index(index)
    if len(source) == 0:
        return pd.Series(dtype=bool, index=source, name="turn_of_month")
    stamps = [_stamp(value) for value in source]
    if stamps != sorted(stamps):
        raise ValueError("turn-of-month dates must be sorted")
    horizon = next_nyse_session(stamps[-1])
    window = window_sessions(n_last, m_first, stamps[0], horizon)
    flags = np.zeros(len(stamps), dtype=bool)
    for i in range(len(stamps) - 1):
        flags[i] = stamps[i + 1] in window
    flags[-1] = _stamp(horizon) in window
    return pd.Series(flags, index=source, name="turn_of_month")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "turn_of_month"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[int, int] | None:
    """Cell in force on ``as_of``, or None when the pick is the live book."""
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
        n_last = row.get("n_last")
        m_first = row.get("m_first")
        if n_last is None or m_first is None:
            raise DataError(f"kept turn-of-month rule {chosen} has no window")
        pair = (int(n_last), int(m_first))
        if pair not in cells() or variant_name(*pair) != chosen:
            raise DataError(f"kept turn-of-month cell {chosen} is outside the grid")
        return pair
    raise DataError(f"kept turn-of-month rule has no parameters for {chosen}")


def kept_turn_of_month(iterations_path, close: pd.Series, as_of=None):
    """Window flag for a kept tilt, or ``(None, None)``.

    The flag is true when the following session is inside the kept window.
    Applying it still requires the 200-day band to be on. A pick of the live
    book, and a rejected iteration, return ``(None, None)``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the turn-of-month tilt")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    pair = _selection(entry, visible.index[-1])
    if pair is None:
        return None, None
    n_last, m_first = pair
    flag = turn_of_month_flag(visible.index, n_last, m_first)
    info = {
        "turn_rule": variant_name(n_last, m_first),
        "turn_n_last": int(n_last),
        "turn_m_first": int(m_first),
        "turn_core_weight": RAISED_CORE,
        "turn_sleeve_weight": RAISED_SLEEVE,
        "turn_live": False,
    }
    return flag, info
