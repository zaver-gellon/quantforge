"""Scale the UPRO sleeve by the cushion above the 200-day average.

While the 200-day band is on, the sleeve stays in equities. Its UPRO share is
``floor + (1 - floor) * clip((close / SMA - 1) / width, 0, 1)``. A close at or
below the average leaves the share at the floor. A close at least ``width``
above the average leaves the share at 1, which is the current sleeve. The rest
of the sleeve is SPY. While the band is off, the sleeve is T-bills and this
series is ignored.

The widths and floors were fixed before this sample was scored. Do not add a
cell after seeing the result. The current book, which keeps the share at 1
whenever the band is on, is not a cell. It is already counted. Every cushion
cell is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# The same average the live band uses. This grid does not replace that filter.
WINDOW = 200
# One step on either axis is a neighbor. A 4% width is full UPRO once the
# close is 4% above the average. A floor of 0.5 keeps the sleeve at least at 2x.
WIDTHS = (0.04, 0.08, 0.12)
FLOORS = (0.0, 0.5)
OFF_NAME = "cush_off"


def cells() -> list[tuple[float, float]]:
    return [(float(width), float(floor)) for width in WIDTHS for floor in FLOORS]


def variant_name(width: float, floor: float) -> str:
    width_points = int(round(float(width) * 100))
    floor_points = int(round(float(floor) * 100))
    return f"cush_w{width_points:02d}_f{floor_points:02d}"


def cushion_share(close: pd.Series, width: float, floor: float) -> pd.Series:
    """UPRO share of the sleeve known at the close. A missing average stays at 1.

    The gap is the close divided by the 200-day average, minus one. It uses
    that close and earlier closes only.
    """
    span = float(width)
    base = float(floor)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("width must be positive")
    if not np.isfinite(base) or base < 0.0 or base >= 1.0:
        raise ValueError("floor must be in [0, 1)")
    prices = close.astype(float).sort_index()
    values = prices.to_numpy(dtype=float)
    if len(values) and (not np.isfinite(values).all() or np.any(values <= 0.0)):
        raise ValueError("closes must be finite and positive")
    average = prices.rolling(WINDOW, min_periods=WINDOW).mean()
    excess = prices / average - 1.0
    scaled = (excess / span).clip(lower=0.0, upper=1.0)
    share = base + (1.0 - base) * scaled
    share = share.where(average.notna(), 1.0).clip(lower=0.0, upper=1.0)
    share = share.mask(share <= 1e-12, 0.0)
    share = share.mask(share >= 1.0 - 1e-12, 1.0)
    share.name = "cushion_share"
    return share


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "cushion"
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


def _params_for(entry: dict, name: str) -> tuple[float, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        width = row.get("width")
        floor = row.get("floor")
        if width is None or floor is None:
            raise DataError(f"kept cushion rule {name} has no width or floor")
        found = (float(width), float(floor))
        if not any(
            abs(found[0] - cell[0]) <= 1e-12 and abs(found[1] - cell[1]) <= 1e-12
            for cell in cells()
        ):
            raise DataError(f"kept cushion parameters {found} are outside the grid")
        return found
    raise DataError(f"kept cushion rule has no parameters for {name}")


def kept_cushion(iterations_path, close: pd.Series, as_of=None):
    """UPRO share of the sleeve for a kept cushion scale, or ``(None, None)``.

    The share is 1 when the walk-forward pick is the current book. The caller
    ignores the share while the 200-day band is off. Sessions after ``as_of``
    are not used.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the cushion rule")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    states = {
        variant_name(width, floor): cushion_share(visible, width, floor)
        for width, floor in cells()
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
    series = pd.Series(values, index=visible.index, name="cushion_share")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    share_now = 1.0 if params is None else float(states[variant_name(*params)].iloc[-1])
    info = {
        "cushion_rule": chosen,
        "cushion_width": None if params is None else float(params[0]),
        "cushion_floor": None if params is None else float(params[1]),
        "cushion_share": share_now,
        "cushion_live": params is None,
    }
    return series, info
