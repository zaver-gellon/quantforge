"""Scale the leveraged sleeve by trailing realized volatility.

The scale at a close uses that close and earlier closes only. It is the fraction
of the sleeve budget left in the leveraged fund on the next session. The rest of
the sleeve budget earns the cash yield. The scale is capped at 1, so the overlay
only cuts exposure. A missing window, or a zero volatility, leaves the sleeve
unscaled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# Fixed before any vol-sleeve path was scored. Neighbors are one step on one axis.
LOOKBACKS = (10, 21, 63)
TARGET_VOLS = (0.10, 0.15, 0.20, 0.25)
OFF_NAME = "vol_off"


def variant_name(lookback: int, target_vol: float) -> str:
    points = int(round(float(target_vol) * 100))
    return f"vol_lb{int(lookback)}_tv{points}"


def sleeve_scale(close: pd.Series, lookback: int, target_vol: float) -> pd.Series:
    """Invested fraction of the sleeve, known at the close.

    Realized volatility is the sample standard deviation of close-to-close
    simple returns over ``lookback`` sessions, annualized with ``sqrt(252)``.
    The fraction is ``min(1, target_vol / realized_vol)``.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not 0.0 < float(target_vol) <= 1.0:
        raise ValueError("target_vol must be in (0, 1]")
    prices = close.astype(float).sort_index()
    returns = prices.pct_change()
    realized = returns.rolling(int(lookback), min_periods=int(lookback)).std(ddof=1) * np.sqrt(252.0)
    raw = float(target_vol) / realized
    scale = raw.clip(upper=1.0)
    scale = scale.where(realized > 0.0, 1.0).fillna(1.0)
    scale = scale.clip(lower=0.0, upper=1.0)
    scale.name = "sleeve_scale"
    return scale


def _kept_vol_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "vol_sleeve"
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
        if row.get("name") == chosen and row.get("lookback") is not None and row.get("target_vol") is not None:
            return chosen, int(row["lookback"]), float(row["target_vol"])
    raise DataError(f"kept vol sleeve has no parameters for {chosen}")


def kept_sleeve_scale(iterations_path, close: pd.Series, as_of=None):
    """Scale series for a kept vol sleeve, or ``(None, None)``.

    The parameters are the walk-forward pick for ``as_of``. The scale is built
    from closes on or before ``as_of`` only.
    """
    entry = _kept_vol_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the vol sleeve")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, lookback, target = _selection(entry, visible.index[-1])
    if rule is None or lookback is None or target is None:
        return None, None
    scale = sleeve_scale(visible, lookback, target)
    realized = visible.pct_change().rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(252.0)
    last_vol = realized.iloc[-1]
    info = {
        "vol_rule": rule,
        "vol_lookback": lookback,
        "vol_target": target,
        "sleeve_scale": float(scale.iloc[-1]),
        "realized_vol": None if not np.isfinite(last_vol) else float(last_vol),
    }
    return scale, info
