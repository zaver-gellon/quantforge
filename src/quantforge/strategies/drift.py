"""Reset the 50/50 mix only when the UPRO sleeve has drifted above its budget.

The live book resets both piles on a quarterly calendar, which sells strength
and buys weakness on that date, and which also trades when the drift is small.
The bands below replace that calendar with a one-sided ceiling. They were
fixed before this sample was scored. Do not add a cell after seeing the result.

The quarterly calendar is the live book. It is already counted. Every band is
a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.engine.backtest import run_drift_band
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.trend import trend_state

# One step on this axis is a neighbor. Widths are fractions of the book.
BANDS = (0.02, 0.04, 0.06, 0.08, 0.10)
OFF_NAME = "drift_off"
REPLAY_COST = 0.0005
_TREND_WINDOW = 200
_TREND_BAND = 0.02


def cells() -> list[float]:
    return [float(band) for band in BANDS]


def variant_name(band: float) -> str:
    points = int(round(float(band) * 100))
    return f"drift_b{points:02d}"


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "drift_band"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_drift(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def lagged_tbill(irx_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Lagged daily T-bill yield. ``^IRX`` is a percent yield, lagged one session."""
    series = irx_close.dropna().astype(float)
    if series.empty:
        raise DataError("^IRX returned no yields")
    if float(series.median()) < 0.5:
        series = series * 100.0
    elif float(series.median()) > 30:
        raise DataError(f"^IRX median {float(series.median()):.2f} does not look like a percent yield")
    rf = (series.reindex(index).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    rf.index = index
    return rf


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _band_for(entry: dict, name: str) -> float | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        band = row.get("band")
        if band is None:
            raise DataError(f"kept drift rule {name} has no band")
        width = float(band)
        if not any(abs(width - item) <= 1e-12 for item in BANDS):
            raise DataError(f"kept drift band {width} is outside the grid")
        return width
    raise DataError(f"kept drift rule has no parameters for {name}")


def _slice_to(series: pd.Series, stamp: pd.Timestamp) -> pd.Series:
    visible = series.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    return visible


def kept_drift_mix(
    iterations_path,
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    cash_yield: pd.Series,
    as_of=None,
):
    """Mix flag for a kept drift band, or ``(None, None)``.

    The flag at each session is the constant-band rule selected for that
    session, replayed on prices up to ``as_of`` only. The quarterly cell
    returns info and no series, so the caller keeps the calendar. Before the
    first pick the replay is that calendar as well.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    close = spy_close.dropna().astype(float).sort_index()
    if close.empty:
        raise DataError("no closes for the drift band")
    stamp = close.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible_close = _slice_to(close, stamp)
    stamp = visible_close.index[-1]
    chosen = _selection(entry, stamp)
    if chosen is None:
        chosen = OFF_NAME
    width = _band_for(entry, chosen)
    info = {
        "drift_rule": chosen,
        "drift_calendar": width is None,
        "drift_band": None if width is None else float(width),
    }
    if width is None:
        return None, info
    idx = visible_close.index
    frames = {
        "spy_open": spy_open,
        "sso_open": sso_open,
        "sso_close": sso_close,
        "upro_open": upro_open,
        "upro_close": upro_close,
        "cash_yield": cash_yield,
    }
    aligned = {}
    for name, series in frames.items():
        column = series.reindex(idx).astype(float)
        if column.isna().any():
            raise DataError(f"{name} is missing sessions at or before as_of")
        aligned[name] = column
    trend = trend_state(visible_close, _TREND_WINDOW, _TREND_BAND, "daily").reindex(idx)
    if trend.isna().any():
        raise DataError("trend is missing sessions at or before as_of")
    dip = dip_active(visible_close).reindex(idx).fillna(False)
    _result, mix = run_drift_band(
        aligned["spy_open"], visible_close,
        aligned["sso_open"], aligned["sso_close"],
        aligned["upro_open"], aligned["upro_close"],
        trend.astype(bool), dip.astype(bool), float(width),
        0.5, 0.5, aligned["cash_yield"],
        spy_cost=REPLAY_COST, sso_cost=REPLAY_COST, upro_cost=REPLAY_COST,
    )
    return mix, info
