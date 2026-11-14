"""Hold a fixed gold or Treasury slice inside the on-trend UPRO sleeve.

The live book is half SPY and half UPRO while the 200-day band is on. The
slice below is taken from that UPRO half only. It is not a second trend
filter, and it is not held while the band is off. The assets and weights
were fixed before this sample was scored. Do not add a cell after seeing
the result.

The current book, which holds no slice, is not a cell. It is already
counted. Every asset and weight below is a new trial. Variant names are
prefixed so ``slice_off`` sorts first and wins a tie in the trailing
excess Sharpe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.idle import align_ohlc

# One step on either axis is a neighbor. Alphabetical asset order is the axis.
ASSETS = ("GLD", "IEF", "TLT")
SLICES = (0.10, 0.20, 0.30)
OFF_NAME = "slice_off"
SLEEVE_WEIGHT = 0.50


def cells() -> list[tuple[str, float]]:
    return [(asset, float(weight)) for asset in ASSETS for weight in SLICES]


def variant_name(asset: str, weight: float) -> str:
    points = int(round(float(weight) * 100))
    return f"zslice_{str(asset).lower()}_w{points:02d}"


def listing_ready(close: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """True once a positive close exists, including gaps of up to five sessions.

    Leading misses stay missing. Nothing is filled backward. The value on a
    session uses that session's close and earlier closes only.
    """
    _open, aligned = align_ohlc(close, close, calendar)
    ready = aligned.notna() & (aligned > 0)
    return ready.rename("slice_ready")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "ontrend_slice"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_ontrend_slice(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _params_for(entry: dict, name: str) -> tuple[str, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        asset = row.get("asset")
        weight = row.get("slice_weight")
        if asset is None or weight is None:
            raise DataError(f"kept on-trend slice {name} has no asset or weight")
        found = (str(asset), float(weight))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept on-trend slice parameters {found} are outside the grid")
        return found
    raise DataError(f"kept on-trend slice has no parameters for {name}")


def kept_ontrend_slice(
    iterations_path,
    closes: dict[str, pd.Series] | None = None,
    as_of=None,
    calendar: pd.DatetimeIndex | None = None,
):
    """Current diversifier slice for a kept rule, or four Nones when none is kept.

    The return is ``(symbol, weight, ready, info)``. The current book, and a
    walk-forward pick of that book, returns a null symbol and a zero weight.
    ``ready`` is aligned to ``calendar`` when one is given, and uses closes on
    or before ``as_of`` only.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None, None, None
    stamp = pd.Timestamp.today().normalize() if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    chosen = _selection(entry, stamp) or OFF_NAME
    params = _params_for(entry, chosen)
    if params is None:
        info = {
            "slice_rule": chosen,
            "slice_symbol": None,
            "slice_weight": 0.0,
            "slice_on": False,
            "slice_live": True,
        }
        return None, None, None, info
    asset, weight = params
    if not closes or asset not in closes:
        raise DataError(f"a kept on-trend slice needs {asset} closes")
    close = pd.Series(closes[asset]).dropna().astype(float).sort_index()
    close = close.loc[:stamp]
    if close.empty:
        raise DataError(f"as_of is before the {asset} history")
    if calendar is None:
        cal = close.index
    else:
        cal = pd.DatetimeIndex(pd.to_datetime(calendar))
        if cal.tz is not None:
            cal = cal.tz_localize(None)
        cal = cal[cal <= stamp]
        if len(cal) == 0:
            raise DataError("as_of is before the equity calendar")
    ready = listing_ready(close, cal)
    listed = bool(ready.iloc[-1]) if len(ready) else False
    info = {
        "slice_rule": chosen,
        "slice_symbol": asset,
        "slice_weight": float(weight),
        "slice_on": bool(listed),
        "slice_live": False,
    }
    return asset, float(weight), ready, info
