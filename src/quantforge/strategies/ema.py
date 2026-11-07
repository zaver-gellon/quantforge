"""Exponential trend filter for the leveraged sleeve.

The live book gates UPRO with an equal-weighted 200-day average. A simple
average of a long rally sits well below the price, so the sleeve stays on
until the close travels all the way back through that average. An exponential
average of the same span puts more weight on the recent closes: the average
sits closer to the price, and a decline crosses the lower band after a smaller
move. On the way back up the new closes dominate, so the re-entry does not
wait for a block of old prices to leave a rectangular window.

The spans and bands were fixed before this sample was scored. Do not add a
cell after seeing the result. The current simple-average book is not a cell.
It is already counted. Every exponential filter is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on either axis is a neighbor. Spans are sessions. Bands match the
# live hysteresis: enter above EMA * (1 + band), exit below EMA * (1 - band).
SPANS = (100, 150, 200)
BANDS = (0.0, 0.02)
OFF_NAME = "ema_off"


def cells() -> list[tuple[int, float]]:
    return [(int(span), float(band)) for span in SPANS for band in BANDS]


def variant_name(span: int, band: float) -> str:
    points = int(round(float(band) * 100))
    return f"ema_s{int(span):03d}_b{points}"


def ema_state(close: pd.Series, span: int, band: float = 0.0) -> pd.Series:
    """True when the next session should be long the risky sleeve.

    The average at t is the recursive exponential filter of closes up to t,
    ``alpha = 2 / (span + 1)``, ``adjust=False``. It is undefined until
    ``span`` closes exist, and the state stays off until then. With a band,
    the sleeve turns on only above ``EMA * (1 + band)`` and off only below
    ``EMA * (1 - band)``. Inside the band the previous state is kept. The
    value at t does not use a later close.
    """
    series = close.astype(float).sort_index()
    length = int(span)
    if length < 2:
        raise ValueError("span must be at least 2")
    width = float(band)
    if width < 0.0 or not np.isfinite(width):
        raise ValueError("band must be finite and non-negative")
    average = series.ewm(span=length, min_periods=length, adjust=False).mean()
    prices = series.to_numpy(dtype=float)
    means = average.to_numpy(dtype=float)
    state = False
    out = np.zeros(len(series), dtype=bool)
    for i in range(len(series)):
        if np.isfinite(means[i]) and np.isfinite(prices[i]):
            upper = prices[i] > means[i] * (1.0 + width)
            lower = prices[i] < means[i] * (1.0 - width)
            if upper:
                state = True
            elif lower:
                state = False
        out[i] = state
    return pd.Series(out, index=series.index, name="ema_on")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "ema_trend"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, int | None, float | None]:
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
        span = row.get("span")
        band = row.get("band")
        if span is None or band is None:
            raise DataError(f"kept exponential filter {chosen} has no span or band")
        found = (int(span), float(band))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept exponential-filter parameters {found} are outside the grid")
        return chosen, found[0], found[1]
    raise DataError(f"kept exponential filter has no parameters for {chosen}")


def kept_ema_trend(iterations_path, close: pd.Series, as_of=None):
    """EMA sleeve state for a kept filter, or ``(None, None)``.

    The parameters are the walk-forward pick in force on ``as_of``. The state
    is built from closes on or before ``as_of`` only. A pick of the current
    simple-average book returns ``(None, None)`` so the live signal stays that
    book. A rejected iteration also returns ``(None, None)``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the exponential filter")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, span, band = _selection(entry, visible.index[-1])
    if rule is None or span is None or band is None:
        return None, None
    state = ema_state(visible, span, band)
    info = {
        "ema_rule": rule,
        "ema_span": int(span),
        "ema_band": float(band),
        "ema_on": bool(state.iloc[-1]),
    }
    return state, info
