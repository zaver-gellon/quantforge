"""N-day confirmation of the 200-day hysteresis band.

The live book already turns the UPRO sleeve on only above ``SMA * (1 + 0.02)``
and off only below ``SMA * (1 - 0.02)``. A single close through that band
flips the sleeve, and the new state then sticks until the other side. The
bands and the confirmation counts below were fixed before this sample was
scored. Do not add a cell after seeing the result.

``band=0.02`` and ``confirm=1`` is that live filter. It is already counted.
Every other cell is a new trial. The value at t uses the close at t and
earlier bars only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on one axis is a neighbor. The window stays 200.
BANDS = (0.005, 0.01, 0.02, 0.03)
CONFIRMS = (1, 2, 3, 4, 5)
WINDOW = 200
LIVE_BAND = 0.02
LIVE_CONFIRM = 1


def cells() -> list[tuple[float, int]]:
    return [(float(band), int(confirm)) for band in BANDS for confirm in CONFIRMS]


def is_live_cell(band: float, confirm: int) -> bool:
    """The current book: the 2% band with an immediate flip."""
    return abs(float(band) - LIVE_BAND) <= 1e-12 and int(confirm) == LIVE_CONFIRM


def variant_name(band: float, confirm: int) -> str:
    bps = int(round(float(band) * 10_000))
    # Zero-padded so an alphabetical tie-break follows the band, then the count.
    return f"confirm_b{bps:04d}_n{int(confirm)}"


def confirmed_trend_state(
    close: pd.Series,
    window: int = WINDOW,
    band: float = LIVE_BAND,
    confirm: int = LIVE_CONFIRM,
) -> pd.Series:
    """True when the next session should be long the risky sleeve.

    The sleeve turns on only after ``confirm`` consecutive closes above
    ``SMA * (1 + band)``, and off only after ``confirm`` consecutive closes
    below ``SMA * (1 - band)``. A close that is not beyond the band in the
    flip direction resets the count. Inside an unfinished count the previous
    state is kept. The book stays off until the first completed entry. Before
    the SMA exists the state is off. ``confirm=1`` is the live hysteresis rule.
    """
    lookback = int(window)
    if lookback < 2:
        raise ValueError("window must be at least 2")
    width = float(band)
    if width < 0.0:
        raise ValueError("band must be non-negative")
    needed = int(confirm)
    if needed < 1:
        raise ValueError("confirm must be at least 1")
    series = close.astype(float).sort_index()
    average = series.rolling(lookback, min_periods=lookback).mean()
    prices = series.to_numpy(dtype=float)
    means = average.to_numpy(dtype=float)
    state = False
    pending = 0
    out = np.zeros(len(series), dtype=bool)
    for i in range(len(series)):
        if np.isfinite(means[i]) and np.isfinite(prices[i]):
            upper = prices[i] > means[i] * (1.0 + width)
            lower = prices[i] < means[i] * (1.0 - width)
            want_on = (not state) and upper
            want_off = state and lower
            if want_on or want_off:
                pending += 1
                if pending >= needed:
                    state = not state
                    pending = 0
            else:
                pending = 0
        out[i] = state
    return pd.Series(out, index=series.index, name="trend_on")


def matches_live_filter(close: pd.Series, window: int = WINDOW) -> bool:
    """``confirm=1`` on the live band is ``trend_state``, bar for bar."""
    confirmed = confirmed_trend_state(close, window, LIVE_BAND, LIVE_CONFIRM)
    live = trend_state(close, window, LIVE_BAND, "daily")
    return bool(confirmed.equals(live.reindex(confirmed.index)))


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "trend_confirm"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, float | None, int | None]:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None:
        return None, None, None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        if row.get("band") is None or row.get("confirm") is None:
            continue
        if is_live_cell(row["band"], row["confirm"]):
            return chosen, None, None
        return chosen, float(row["band"]), int(row["confirm"])
    raise DataError(f"kept confirmation rule has no parameters for {chosen}")


def kept_confirm_trend(iterations_path, close: pd.Series, as_of=None):
    """Confirmed trend for a kept rule, or ``(None, None)``.

    The parameters are the walk-forward pick for ``as_of``. The state is built
    from closes on or before ``as_of`` only. The live cell is not an override.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the confirmation rule")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, band, confirm = _selection(entry, visible.index[-1])
    if rule is None or band is None or confirm is None:
        return None, None
    state = confirmed_trend_state(visible, WINDOW, band, confirm)
    info = {
        "confirm_rule": rule,
        "confirm_band": band,
        "confirm_days": confirm,
        "confirm_window": WINDOW,
        "confirm_on": bool(state.iloc[-1]),
    }
    return state, info
