"""Absolute-momentum ensemble for the leveraged sleeve.

The live book gates UPRO with one 200-day average and a 2% band. The
2026-10-27 vote already scores agreement among SMA hysteresis states at
windows 50, 100, 150, and 200. This rule does not rescore that vote and it
does not average those averages. A horizon is positive when the close is
strictly above the close that many sessions earlier. The sleeve is on when
enough horizons are positive. The short leg, the vote count, and the fixed
6- and 12-month legs below were fixed before this sample was scored. Do not
add a cell after seeing the result.

The value at t uses closes at t and earlier only. Before every horizon has a
prior close the state is off. A zero return is not positive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on one axis is a neighbor. 21 sessions is the month.
# The center short leg is 63: the 3/6/12-month set. 42 and 84 are one month away.
MONTH = 21
SHORTS = (2 * MONTH, 3 * MONTH, 4 * MONTH)
MID = 6 * MONTH
LONG = 12 * MONTH
MIN_POSITIVE = (2, 3)
OFF_NAME = "mom_off"


def horizons_for(short: int) -> tuple[int, int, int]:
    leg = int(short)
    if leg not in SHORTS:
        raise ValueError(f"short horizon must be one of {SHORTS}")
    return (leg, MID, LONG)


def cells() -> list[tuple[int, int]]:
    """New trials. The live 200-day band is not in this list."""
    return [(int(short), int(votes)) for short in SHORTS for votes in MIN_POSITIVE]


def variant_name(short: int, min_positive: int) -> str:
    leg = int(short)
    votes = int(min_positive)
    if leg not in SHORTS or votes not in MIN_POSITIVE:
        raise ValueError(f"unknown absolute-momentum cell {(short, min_positive)}")
    return f"zmom_s{leg:03d}_k{votes}"


def absolute_momentum_state(
    close: pd.Series,
    horizons: tuple[int, ...] | list[int],
    min_positive: int,
) -> pd.Series:
    """True when at least ``min_positive`` trailing returns are strictly positive.

    The return at t over horizon h is ``close[t] / close[t - h] - 1``. It does
    not use a later close. The state is off until every horizon is defined.
    """
    series = close.astype(float).sort_index()
    lookbacks = tuple(int(horizon) for horizon in horizons)
    if len(lookbacks) < 1 or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("horizons must be distinct and non-empty")
    if any(horizon < 1 for horizon in lookbacks):
        raise ValueError("each horizon must be at least 1")
    votes = int(min_positive)
    if votes < 1 or votes > len(lookbacks):
        raise ValueError("min_positive must be between 1 and the number of horizons")
    prices = series.to_numpy(dtype=float)
    count = np.zeros(len(series), dtype=float)
    ready = np.ones(len(series), dtype=bool)
    for horizon in lookbacks:
        lagged = np.full(len(series), np.nan)
        lagged[horizon:] = prices[:-horizon]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = prices / lagged - 1.0
        known = np.isfinite(ret) & np.isfinite(prices)
        count += (known & (ret > 0.0)).astype(float)
        ready &= known
    on = ready & (count + 1e-12 >= votes)
    return pd.Series(on, index=series.index, name="momentum_on")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "abs_momentum"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, int | None, int | None]:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == OFF_NAME:
        return chosen, None, None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        short = row.get("short")
        votes = row.get("min_positive")
        if short is None or votes is None:
            raise DataError(f"kept absolute momentum {chosen} has no short horizon or vote")
        found = (int(short), int(votes))
        if found not in cells():
            raise DataError(f"kept absolute-momentum parameters {found} are outside the grid")
        return chosen, found[0], found[1]
    raise DataError(f"kept absolute momentum has no parameters for {chosen}")


def kept_abs_momentum(iterations_path, close: pd.Series, as_of=None):
    """Sleeve state for a kept absolute-momentum rule, or ``(None, None)``.

    The parameters are the walk-forward pick in force on ``as_of``. The state
    is built from closes on or before ``as_of`` only. A pick of the current
    200-day book returns ``(None, None)``. A rejected iteration also returns
    ``(None, None)``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the absolute-momentum filter")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, short, votes = _selection(entry, visible.index[-1])
    if rule is None or short is None or votes is None:
        return None, None
    horizons = horizons_for(short)
    state = absolute_momentum_state(visible, horizons, votes)
    info = {
        "momentum_rule": rule,
        "momentum_horizons": list(horizons),
        "momentum_min_positive": int(votes),
        "momentum_on": bool(state.iloc[-1]),
    }
    return state, info
