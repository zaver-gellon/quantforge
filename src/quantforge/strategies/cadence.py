"""Calendar cadence for the live 200-day sleeve filter.

The live book evaluates the 200-day average, and its 2% band, every session.
One close through the band flips the UPRO sleeve. This rule does not change
the average or the band. It updates that same state only on the last NYSE
session of the ISO week, or only on the last session of the month, and it
carries the previous state forward on every other close. A close through the
band between checks does not trade.

The cadences were fixed before this sample was scored. Do not add a cell
after seeing the result. Daily evaluation is not a cell. It is already
counted. Weekly and monthly evaluation of this 2% band are the new trials.
The original grid's month-end SMA 200 uses a 0% band and is not this cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.engine.schedule import rebalance_mask
from quantforge.strategies.trend import trend_state

# One step on this axis is the other cadence. Daily is the live book.
FREQS = ("weekly", "monthly")
OFF_NAME = "cad_off"
WINDOW = 200
BAND = 0.02


def cells() -> list[str]:
    """New trials. The live daily filter is not in this list."""
    return list(FREQS)


def variant_name(freq: str) -> str:
    if freq not in FREQS:
        raise ValueError(f"unknown evaluation cadence {freq!r}")
    return f"zcad_{freq}"


def cadence_state(close: pd.Series, freq: str, window: int = WINDOW, band: float = BAND) -> pd.Series:
    """200-day 2% state updated only on the named calendar.

    ``weekly`` uses the last session of each ISO week. ``monthly`` uses the
    last session of each month. The value at t uses the close at t and earlier
    bars only. Between checks the previous state is kept.
    """
    if freq not in FREQS:
        raise ValueError(f"unknown evaluation cadence {freq!r}")
    if int(window) != WINDOW or abs(float(band) - BAND) > 1e-12:
        raise ValueError("the cadence test uses the live 200-day 2% band only")
    return trend_state(close, WINDOW, BAND, freq).rename("cadence_on")


def evaluation_mask(index, freq: str) -> pd.Series:
    """Sessions on which ``freq`` is allowed to update the sleeve state."""
    if freq not in FREQS:
        raise ValueError(f"unknown evaluation cadence {freq!r}")
    return rebalance_mask(index, freq)


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "eval_cadence"
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
    if chosen is None or chosen == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        freq = row.get("freq")
        if freq not in FREQS:
            raise DataError(f"kept evaluation cadence {chosen} is outside the grid")
        if variant_name(freq) != chosen:
            raise DataError(f"kept evaluation cadence {chosen} does not match {freq}")
        return freq
    raise DataError(f"kept evaluation cadence has no parameters for {chosen}")


def kept_eval_cadence(iterations_path, close: pd.Series, as_of=None):
    """Sleeve state for a kept cadence, or ``(None, None)``.

    The cell is the walk-forward pick in force on ``as_of``. The state uses
    closes on or before ``as_of`` only. A pick of the daily book, and a
    rejected iteration, return ``(None, None)``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the evaluation cadence")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    freq = _selection(entry, visible.index[-1])
    if freq is None:
        return None, None
    state = cadence_state(visible, freq)
    info = {
        "cadence_rule": variant_name(freq),
        "cadence": freq,
        "cadence_window": WINDOW,
        "cadence_band": BAND,
        "cadence_on": bool(state.iloc[-1]),
    }
    return state, info
