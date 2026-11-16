"""Hold the on-trend sleeve in SPY when the 13-week bill is expensive.

The 3x sleeve pays financing of two times the bill plus 40 basis points. While
the 200-day band is on and that session's ``^IRX`` close, in percent, is
strictly above a threshold, the sleeve's UPRO share is the floor and the rest
is SPY. At or below the threshold the share is 1, which is the current sleeve.
While the band is off, the sleeve is T-bills and this series is ignored.

The thresholds and floors were fixed before this sample was scored. Do not add
a cell after seeing the result. The current book, which keeps the share at 1
whenever the band is on, is not a cell. It is already counted. Every bill cell
is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on either axis is a neighbor. 1% separates a near-zero bill from a
# positive one. 3% is a mid-cycle bill. 5% is a restrictive bill. A floor of
# 0.5 keeps the sleeve at 2x instead of moving it all the way to SPY.
THRESHOLDS = (1.0, 3.0, 5.0)
FLOORS = (0.0, 0.5)
OFF_NAME = "bill_off"


def cells() -> list[tuple[float, float]]:
    return [(float(threshold), float(floor)) for threshold in THRESHOLDS for floor in FLOORS]


def variant_name(threshold: float, floor: float) -> str:
    points = int(round(float(threshold)))
    floor_points = int(round(float(floor) * 100))
    return f"bill_t{points:02d}_f{floor_points:02d}"


def align_bill_yield(yield_pct: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Percent ``^IRX`` on ``calendar``. Gaps fill forward. Nothing fills backward."""
    quotes = yield_pct.astype(float).sort_index()
    quotes = quotes[~quotes.index.duplicated(keep="last")]
    aligned = quotes.reindex(pd.DatetimeIndex(calendar)).ffill()
    aligned.name = "bill_yield_pct"
    return aligned


def bill_share(yield_pct: pd.Series, threshold: float, floor: float) -> pd.Series:
    """UPRO share of the sleeve known at the close. A missing yield stays at 1.

    The quote is that session's bill yield in percent. A quote strictly above
    ``threshold`` sets the share to ``floor``. A quote at or below it leaves
    the share at 1.
    """
    level = float(threshold)
    base = float(floor)
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError("threshold must be positive")
    if not np.isfinite(base) or base < 0.0 or base >= 1.0:
        raise ValueError("floor must be in [0, 1)")
    quotes = yield_pct.astype(float).sort_index()
    hot = quotes > level
    share = pd.Series(1.0, index=quotes.index, dtype=float)
    share = share.mask(hot, base)
    share = share.where(quotes.notna(), 1.0)
    share = share.mask(share <= 1e-12, 0.0)
    share = share.mask(share >= 1.0 - 1e-12, 1.0)
    share.name = "bill_share"
    return share


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "bill_gear"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_bill(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _params_for(entry: dict, name: str) -> tuple[float, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        threshold = row.get("threshold_pct")
        floor = row.get("floor")
        if threshold is None or floor is None:
            raise DataError(f"kept bill-yield rule {name} has no threshold or floor")
        found = (float(threshold), float(floor))
        if not any(
            abs(found[0] - cell[0]) <= 1e-12 and abs(found[1] - cell[1]) <= 1e-12
            for cell in cells()
        ):
            raise DataError(f"kept bill-yield parameters {found} are outside the grid")
        return found
    raise DataError(f"kept bill-yield rule has no parameters for {name}")


def kept_bill_gear(iterations_path, yield_pct: pd.Series | None = None, as_of=None):
    """UPRO share of the sleeve for a kept bill-yield gear, or ``(None, None)``.

    The share is 1 when the walk-forward pick is the current book. The caller
    ignores the share while the 200-day band is off. Sessions after ``as_of``
    are not used. ``yield_pct`` is the percent bill yield on the decision dates.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    if yield_pct is None:
        raise DataError("a kept bill-yield gear needs the ^IRX series")
    quotes = yield_pct.astype(float).sort_index()
    if quotes.empty:
        raise DataError("no bill yields for the bill-yield rule")
    stamp = quotes.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = quotes.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the bill-yield history")
    states = {
        variant_name(threshold, floor): bill_share(visible, threshold, floor)
        for threshold, floor in cells()
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
    series = pd.Series(values, index=visible.index, name="bill_share")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    share_now = 1.0 if params is None else float(states[variant_name(*params)].iloc[-1])
    quote_now = float(visible.iloc[-1]) if np.isfinite(visible.iloc[-1]) else None
    info = {
        "bill_rule": chosen,
        "bill_threshold_pct": None if params is None else float(params[0]),
        "bill_floor": None if params is None else float(params[1]),
        "bill_yield_pct": quote_now,
        "bill_share": share_now,
        "bill_live": params is None,
    }
    return series, info
