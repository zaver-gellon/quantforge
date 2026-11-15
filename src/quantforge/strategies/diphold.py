"""Exit rule for the three-down-close dip on the SPY half.

The live overlay enters after three strictly down closes and leaves at the
first up close or after five sessions. An up close ends the hold, so a down
close is what extends it. The exits and the caps below were fixed before this
sample was scored. Do not add a cell after seeing the result.

``exit="up"`` and ``max_hold=5`` is that live overlay. It is already counted.
Every other cell is a new trial. The three-down entry is not searched. An
exit close does not re-enter on that same close. The value at t uses the
close at t and earlier bars only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.dipbuy import rule_mask

# One step on either axis is a neighbor. "down" leaves on a down close.
# "fixed" holds the full cap and ignores the sign. "up" is the live exit.
EXITS = ("down", "fixed", "up")
HOLDS = (3, 5, 10)
LIVE_EXIT = "up"
LIVE_HOLD = 5
OFF_NAME = "hold_live"
RULE = "3_down_closes"


def cells() -> list[tuple[str, int]]:
    """New trials. The live exit is not in this list."""
    return [
        (exit_mode, int(hold))
        for exit_mode in EXITS
        for hold in HOLDS
        if not is_live_cell(exit_mode, hold)
    ]


def is_live_cell(exit_mode: str, hold: int) -> bool:
    """The current book: leave on the first up close, or after five sessions."""
    return exit_mode == LIVE_EXIT and int(hold) == LIVE_HOLD


def variant_name(exit_mode: str, hold: int) -> str:
    if exit_mode not in EXITS:
        raise ValueError(f"unknown dip exit {exit_mode!r}")
    if int(hold) not in HOLDS:
        raise ValueError(f"unknown dip hold {hold}")
    # Prefixed so the live book's name sorts first and wins an alphabetical tie.
    return f"zhold_{exit_mode}_n{int(hold):02d}"


def cell_name(exit_mode: str, hold: int) -> str:
    """Name used in the walk-forward, including the current book."""
    if is_live_cell(exit_mode, hold):
        return OFF_NAME
    return variant_name(exit_mode, hold)


def dip_hold_flag(
    close: pd.Series,
    exit_mode: str = LIVE_EXIT,
    max_hold: int = LIVE_HOLD,
    rule: str = RULE,
) -> pd.Series:
    """True when the SPY half should be SSO on the next session.

    The entry is ``rule``. ``exit_mode="up"`` leaves at the first up close or
    after ``max_hold`` sessions, and that is ``dip_active`` when the cap is 5.
    ``exit_mode="down"`` leaves at the first down close of the hold, or after
    the cap. ``exit_mode="fixed"`` holds for ``max_hold`` sessions and ignores
    the sign of the close. A same-close exit is not re-armed.
    """
    if exit_mode not in EXITS:
        raise ValueError(f"unknown dip exit {exit_mode!r}")
    cap = int(max_hold)
    if cap < 1:
        raise ValueError("max_hold must be positive")
    series = close.astype(float).sort_index()
    signal = rule_mask(series, rule).to_numpy(dtype=bool)
    prices = series.to_numpy(dtype=float)
    n = len(series)
    held = np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    in_trade = False
    hold_count = 0
    for t in range(n):
        if not in_trade:
            if signal[t]:
                in_trade = True
                hold_count = 0
                nxt = True
            else:
                nxt = False
        elif held[t]:
            hold_count += 1
            up = (
                t > 0
                and np.isfinite(prices[t])
                and np.isfinite(prices[t - 1])
                and prices[t] > prices[t - 1]
            )
            down = (
                t > 0
                and np.isfinite(prices[t])
                and np.isfinite(prices[t - 1])
                and prices[t] < prices[t - 1]
            )
            if exit_mode == "up":
                leave = (up and hold_count >= 1) or hold_count >= cap
            elif exit_mode == "down":
                leave = (down and hold_count >= 1) or hold_count >= cap
            else:
                leave = hold_count >= cap
            if leave:
                in_trade = False
                nxt = False
            else:
                nxt = True
        else:
            in_trade = False
            nxt = False
        out[t] = nxt
        if t + 1 < n:
            held[t + 1] = nxt
    return pd.Series(out, index=series.index, name="dip_hold")


def episode_lengths(flag: pd.Series) -> list[int]:
    """Lengths of consecutive holds. A true value is one session held next."""
    lengths: list[int] = []
    run = 0
    for bit in flag.fillna(False).astype(bool).to_numpy():
        if bit:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "dip_hold"
    ]
    if not kept:
        return None
    return kept[-1]


def _params(entry: dict, name: str | None) -> tuple[str, int] | None:
    if name is None:
        return None
    if name == OFF_NAME:
        return LIVE_EXIT, LIVE_HOLD
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        exit_mode = row.get("exit")
        hold = row.get("max_hold")
        if exit_mode not in EXITS or hold is None:
            raise DataError(f"kept dip-hold rule {name} has no parameters")
        cap = int(hold)
        if cap not in HOLDS:
            raise DataError(f"kept dip-hold cap {cap} is outside the grid")
        if is_live_cell(exit_mode, cap):
            raise DataError(f"kept dip-hold cell {name} is stored as a new trial of the live exit")
        return exit_mode, cap
    raise DataError(f"kept dip-hold rule has no parameters for {name}")


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def kept_dip_hold(iterations_path, close: pd.Series, as_of=None):
    """Hold flag for a kept dip exit, or ``(None, None)``.

    The flag at each session is the full-history rule of the walk-forward
    cell selected for that session. Sessions after ``as_of`` are not used.
    Before the first pick, the flag is the live overlay. The live pick returns
    the live flag and marks ``dip_hold_live``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the dip hold")
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
    rules: dict[str, pd.Series] = {}
    values = np.zeros(len(visible), dtype=bool)
    cursor = 0
    current = OFF_NAME
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current = picks[cursor][1]
            cursor += 1
        if current not in rules:
            params = _params(entry, current)
            if params is None:
                raise DataError(f"kept dip-hold pick {current} is not a cell")
            rules[current] = dip_hold_flag(visible, params[0], params[1])
        values[i] = bool(rules[current].iloc[i])
    series = pd.Series(values, index=visible.index, name="dip_hold")
    chosen = _selection(entry, visible.index[-1])
    if chosen is None:
        chosen = OFF_NAME
    params = _params(entry, chosen)
    if params is None:
        raise DataError(f"kept dip-hold pick {chosen} is not a cell")
    info = {
        "dip_hold_rule": chosen,
        "dip_hold_exit": params[0],
        "dip_hold_max": int(params[1]),
        "dip_hold_live": bool(is_live_cell(params[0], params[1])),
    }
    return series, info
