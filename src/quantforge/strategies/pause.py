"""Pause the on-trend UPRO sleeve after a short run of down closes.

The 3x sleeve stays fully invested until SPY closes through its 200-day band.
A down streak is a few sessions, not a new average and not a volatility, bill,
or cushion state. After K strictly down closes, and only while that band is on,
the sleeve's UPRO share drops to a floor and the rest of the sleeve is SPY.
The pause ends on the first up close or after five sessions. Five is the live
dip's cap and is not a searched axis. The core's three-down-close dip is a
separate clock.

The streaks and floors were fixed before this sample was scored. Do not add a
cell after seeing the result. The current book, which keeps the share at 1
whenever the band is on, is not a cell. It is already counted. Every pause
cell is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on either axis is a neighbor. Two down closes is a short streak.
# Three matches the live dip's entry count but is applied to the sleeve.
# Four is the next longer streak. A floor of 0.5 keeps the sleeve at 2x.
STREAKS = (2, 3, 4)
FLOORS = (0.0, 0.5)
MAX_HOLD = 5
WINDOW = 200
BAND = 0.02
OFF_NAME = "pause_off"


def cells() -> list[tuple[int, float]]:
    return [(int(streak), float(floor)) for streak in STREAKS for floor in FLOORS]


def variant_name(streak: int, floor: float) -> str:
    floor_points = int(round(float(floor) * 100))
    return f"pause_k{int(streak):02d}_f{floor_points:02d}"


def pause_share(
    close: pd.Series,
    trend_on: pd.Series,
    streak: int,
    floor: float,
    max_hold: int = MAX_HOLD,
) -> pd.Series:
    """UPRO share of the sleeve known at the close.

    The value at t uses the close and the trend state through t only. After
    ``streak`` strictly down closes, and only when the trend is on at t, the
    share for the next session is ``floor``. It stays there until the close
    of the first up session, until ``max_hold`` sessions have been decided,
    or until the trend is off, whichever comes first. A same-close exit is
    not re-armed. A missing trend row is an error. The share is 1 when no
    pause is on, which is the current sleeve.
    """
    needed = int(streak)
    if needed < 2:
        raise ValueError("streak must be at least 2")
    base = float(floor)
    if not np.isfinite(base) or base < 0.0 or base >= 1.0:
        raise ValueError("floor must be in [0, 1)")
    hold_cap = int(max_hold)
    if hold_cap < 1:
        raise ValueError("max_hold must be positive")
    series = close.astype(float).sort_index()
    trend = trend_on.reindex(series.index)
    if trend.isna().any():
        raise ValueError("trend state is missing sessions")
    trend_flag = trend.astype(bool).to_numpy()
    down = series.diff() < 0
    mask = down.copy()
    for lag in range(1, needed):
        mask = mask & down.shift(lag)
    signal = (mask.fillna(False).to_numpy(dtype=bool)) & trend_flag
    prices = series.to_numpy(dtype=float)
    n = len(series)
    held = np.zeros(n, dtype=bool)
    decided = np.ones(n, dtype=float)
    in_trade = False
    hold_count = 0
    for t in range(n):
        if not in_trade:
            if signal[t]:
                in_trade = True
                hold_count = 0
                nxt = base
            else:
                nxt = 1.0
        elif held[t]:
            hold_count += 1
            up = (
                t > 0
                and np.isfinite(prices[t])
                and np.isfinite(prices[t - 1])
                and prices[t] > prices[t - 1]
            )
            if (not trend_flag[t]) or (up and hold_count >= 1) or hold_count >= hold_cap:
                in_trade = False
                nxt = 1.0
            else:
                nxt = base
        else:
            in_trade = False
            nxt = 1.0
        if nxt <= 1e-12:
            nxt = 0.0
        elif nxt >= 1.0 - 1e-12:
            nxt = 1.0
        decided[t] = nxt
        if t + 1 < n:
            held[t + 1] = nxt < 1.0 - 1e-12
    out = pd.Series(decided, index=series.index, name="pause_share")
    return out


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "streak_pause"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_streak_pause(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


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
        streak = row.get("streak")
        floor = row.get("floor")
        if streak is None or floor is None:
            raise DataError(f"kept down-streak pause {name} has no streak or floor")
        found = (int(streak), float(floor))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept down-streak parameters {found} are outside the grid")
        return found
    raise DataError(f"kept down-streak pause has no parameters for {name}")


def kept_streak_pause(iterations_path, close: pd.Series | None = None, as_of=None):
    """UPRO share of the sleeve for a kept down-streak pause, or ``(None, None)``.

    The share is 1 when the walk-forward pick is the current book. The caller
    ignores the share while the 200-day band is off. Sessions after ``as_of``
    are not used. The band is the live 200-day 2% state of ``close``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    if close is None:
        raise DataError("a kept down-streak pause needs the SPY close")
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the down-streak pause")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    trend = trend_state(visible, WINDOW, BAND, "daily")
    states = {
        variant_name(streak, floor): pause_share(visible, trend, streak, floor)
        for streak, floor in cells()
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
            values[i] = float(states[current_name].loc[ts])
    series = pd.Series(values, index=visible.index, name="pause_share")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    share_now = 1.0 if params is None else float(states[variant_name(*params)].iloc[-1])
    info = {
        "pause_rule": chosen,
        "pause_streak": None if params is None else int(params[0]),
        "pause_floor": None if params is None else float(params[1]),
        "pause_max_hold": MAX_HOLD,
        "pause_share": share_now,
        "pause_on": bool(share_now < 1.0 - 1e-8),
        "pause_live": params is None,
    }
    return series, info
