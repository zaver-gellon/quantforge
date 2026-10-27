"""Multi-lookback vote for the leveraged sleeve.

The live book gates UPRO with one 200-day average and a 2% band. The grid
already scores that rule, and it scores SMA 100 and SMA 150 as replacements.
It does not score a vote across several lookbacks. The windows, the vote
counts, and the bands below were fixed before this sample was scored. Do not
add a cell after seeing the result.

Each lookback is the same hysteresis rule as ``trend_state``: it turns on only
above ``SMA * (1 + band)`` and off only below ``SMA * (1 - band)``. The sleeve
is on when at least ``min_votes`` of those states are on. The value at t uses
the close at t and earlier bars only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on one axis is a neighbor. The long window stays 200.
# short=50 uses (50, 100, 200). short=100 uses (100, 150, 200).
SHORT_WINDOWS = (50, 100)
MIN_VOTES = (2, 3)
BANDS = (0.0, 0.02)
LONG_WINDOW = 200
OFF_NAME = "vote_off"


def windows_for(short_window: int) -> tuple[int, int, int]:
    short = int(short_window)
    if short not in SHORT_WINDOWS:
        raise ValueError(f"short window must be one of {SHORT_WINDOWS}")
    return (short, short + 50, LONG_WINDOW)


def cells() -> list[tuple[int, int, float]]:
    return [
        (short, votes, band)
        for short in SHORT_WINDOWS
        for votes in MIN_VOTES
        for band in BANDS
    ]


def variant_name(short_window: int, min_votes: int, band: float) -> str:
    points = int(round(float(band) * 100))
    return f"vote_s{int(short_window)}_k{int(min_votes)}_b{points}"


def vote_state(
    close: pd.Series,
    windows: tuple[int, ...] | list[int],
    band: float,
    min_votes: int,
) -> pd.Series:
    """True when at least ``min_votes`` lookbacks are in an uptrend."""
    lookbacks = tuple(int(window) for window in windows)
    if len(lookbacks) < 1 or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("windows must be distinct and non-empty")
    if any(window < 2 for window in lookbacks):
        raise ValueError("each window must be at least 2")
    votes = int(min_votes)
    if votes < 1 or votes > len(lookbacks):
        raise ValueError("min_votes must be between 1 and the number of windows")
    if band < 0:
        raise ValueError("band must be non-negative")
    count = None
    for window in lookbacks:
        state = trend_state(close, window, float(band), "daily").astype(int)
        count = state if count is None else count + state
    on = count >= votes
    on.name = "vote_on"
    return on


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "trend_vote"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, int | None, int | None, float | None]:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == OFF_NAME:
        return chosen, None, None, None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        if row.get("short_window") is None or row.get("min_votes") is None or row.get("band") is None:
            continue
        return chosen, int(row["short_window"]), int(row["min_votes"]), float(row["band"])
    raise DataError(f"kept trend vote has no parameters for {chosen}")


def kept_vote_trend(iterations_path, close: pd.Series, as_of=None):
    """Vote series for a kept multi-lookback rule, or ``(None, None)``.

    The parameters are the walk-forward pick for ``as_of``. The state is built
    from closes on or before ``as_of`` only.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the trend vote")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, short, votes, band = _selection(entry, visible.index[-1])
    if rule is None or short is None or votes is None or band is None:
        return None, None
    windows = windows_for(short)
    state = vote_state(visible, windows, band, votes)
    info = {
        "vote_rule": rule,
        "vote_windows": list(windows),
        "vote_min": votes,
        "vote_band": band,
        "vote_on": bool(state.iloc[-1]),
    }
    return state, info
