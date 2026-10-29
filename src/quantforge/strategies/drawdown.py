"""Trailing-peak stop for the leveraged sleeve.

The live book gates UPRO with one 200-day average and a 2% band. That average
stays long through the first leg of a crash. A trailing peak cuts the sleeve
only after SPY itself has fallen a set amount from its own high. It does not
scale the sleeve by realized volatility, and it does not replace the average
with another lookback or a vote. The windows, the cuts, and the re-entry
fraction below were fixed before this sample was scored. Do not add a cell
after seeing the result.

The peak at a close is the maximum close over the trailing window, including
that close. The stop turns the sleeve off when the close is at least ``cut``
below the peak, and turns it back on when the drawdown has healed to
``reentry_fraction * cut`` or better. Inside that band the previous state is
kept. Before the window is full the stop has not fired. The value at t uses
the close at t and earlier bars only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on one axis is a neighbor. Re-entry is half the cut, not a third axis.
WINDOWS = (63, 126, 252)
CUTS = (0.08, 0.12, 0.16)
REENTRY_FRACTION = 0.5
OFF_NAME = "dd_off"


def cells() -> list[tuple[int, float]]:
    return [(window, cut) for window in WINDOWS for cut in CUTS]


def variant_name(window: int, cut: float) -> str:
    points = int(round(float(cut) * 100))
    return f"dd_w{int(window)}_c{points}"


def sleeve_allowed(
    close: pd.Series,
    window: int,
    cut: float,
    reentry_fraction: float = REENTRY_FRACTION,
) -> pd.Series:
    """True when the trailing-peak stop leaves the UPRO sleeve invested.

    False means the sleeve budget earns T-bills on the next session even if
    the 200-day trend is still on. The 200-day gate is applied by the caller.
    """
    lookback = int(window)
    if lookback < 2:
        raise ValueError("window must be at least 2")
    threshold = float(cut)
    if not 0.0 < threshold < 1.0:
        raise ValueError("cut must be in (0, 1)")
    fraction = float(reentry_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("reentry_fraction must be in [0, 1)")
    prices = close.astype(float).sort_index()
    peak = prices.rolling(lookback, min_periods=lookback).max()
    heal = threshold * fraction
    allowed = True
    out = np.empty(len(prices), dtype=bool)
    px = prices.to_numpy(dtype=float)
    pk = peak.to_numpy(dtype=float)
    for i in range(len(prices)):
        if np.isfinite(pk[i]) and pk[i] > 0.0 and np.isfinite(px[i]):
            drawdown = px[i] / pk[i] - 1.0
            if allowed and drawdown <= -threshold:
                allowed = False
            elif (not allowed) and drawdown >= -heal:
                allowed = True
        out[i] = allowed
    return pd.Series(out, index=prices.index, name="dd_allowed")


def sleeve_scale(
    close: pd.Series,
    window: int,
    cut: float,
    reentry_fraction: float = REENTRY_FRACTION,
) -> pd.Series:
    """Invested fraction of the sleeve budget: 1 when the stop is idle, else 0."""
    allowed = sleeve_allowed(close, window, cut, reentry_fraction)
    scale = allowed.astype(float)
    scale.name = "sleeve_scale"
    return scale


def trailing_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Close divided by the trailing peak, minus one. Missing until the window is full."""
    prices = close.astype(float).sort_index()
    peak = prices.rolling(int(window), min_periods=int(window)).max()
    drawdown = prices / peak - 1.0
    drawdown.name = "trailing_drawdown"
    return drawdown


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "drawdown_stop"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, int | None, float | None, float | None]:
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
        if row.get("window") is None or row.get("cut") is None:
            continue
        fraction = row.get("reentry_fraction", REENTRY_FRACTION)
        return chosen, int(row["window"]), float(row["cut"]), float(fraction)
    raise DataError(f"kept drawdown stop has no parameters for {chosen}")


def kept_drawdown_scale(iterations_path, close: pd.Series, as_of=None):
    """Scale series for a kept trailing-peak stop, or ``(None, None)``.

    The parameters are the walk-forward pick for ``as_of``. The scale is built
    from closes on or before ``as_of`` only.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the drawdown stop")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, window, cut, fraction = _selection(entry, visible.index[-1])
    if rule is None or window is None or cut is None or fraction is None:
        return None, None
    scale = sleeve_scale(visible, window, cut, fraction)
    drawdown = trailing_drawdown(visible, window)
    peak = visible.rolling(window, min_periods=window).max()
    last_dd = drawdown.iloc[-1]
    last_peak = peak.iloc[-1]
    info = {
        "drawdown_rule": rule,
        "drawdown_window": window,
        "drawdown_cut": cut,
        "drawdown_reentry": cut * fraction,
        "drawdown_stop_on": bool(scale.iloc[-1] == 0.0),
        "trailing_peak": None if not np.isfinite(last_peak) else float(last_peak),
        "trailing_drawdown": None if not np.isfinite(last_dd) else float(last_dd),
    }
    return scale, info
