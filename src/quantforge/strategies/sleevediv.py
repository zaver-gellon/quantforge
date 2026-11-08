"""Park the off-trend sleeve in a bond or gold trend. The SPY core stays put.

The idle test moved the whole book into GLD, IEF, or TLT when the 200-day
band was off, which sold the SPY half along with the sleeve. This rule never
sells that half. While the equity band is on, the book is the unchanged trend
core. While it is off, only the sleeve — the pile the live book already holds
in T-bills — holds the diversifier if that asset's own filter is on, and
T-bills if the filter is off or not yet defined.

The assets, windows, and bands were fixed before this sample was scored.
Assets are in alphabetical order so a neighbor is one step on that list.
Do not add a cell after seeing the result. The current book is not a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.idle import align_ohlc, idle_state

# One step on one axis is a neighbor. Alphabetical asset order is the axis.
ASSETS = ("GLD", "IEF", "TLT")
WINDOWS = (100, 200)
BANDS = (0.0, 0.02)
OFF_NAME = "sdiv_off"


def cells() -> list[tuple[str, int, float]]:
    return [(asset, int(window), float(band)) for asset in ASSETS for window in WINDOWS for band in BANDS]


def variant_name(asset: str, window: int, band: float) -> str:
    points = int(round(float(band) * 100))
    return f"sdiv_{str(asset).lower()}_w{int(window)}_b{points}"


def sleeve_cash_state(
    close: pd.Series,
    window: int,
    band: float,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Diversifier trend and whether that filter is defined, on ``calendar``.

    The value on a session uses that session's close and earlier closes only.
    Both series are false before the window is complete.
    """
    on, ready = idle_state(close, int(window), float(band), calendar)
    return on.rename("sleeve_cash_on"), ready.rename("sleeve_cash_ready")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "sleeve_cash"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, str | None, int | None, float | None]:
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
        symbol = row.get("symbol")
        window = row.get("window")
        band = row.get("band")
        if symbol is None or window is None or band is None:
            raise DataError(f"kept sleeve diversifier {chosen} has no asset, window, or band")
        found = (str(symbol), int(window), float(band))
        if not any(
            found[0] == cell[0] and found[1] == cell[1] and abs(found[2] - cell[2]) <= 1e-12
            for cell in cells()
        ):
            raise DataError(f"kept sleeve-diversifier parameters {found} are outside the grid")
        return chosen, found[0], found[1], found[2]
    raise DataError(f"kept sleeve diversifier has no parameters for {chosen}")


def kept_sleeve_cash_choice(iterations_path, as_of) -> dict | None:
    """Parameters of the kept walk-forward pick, or None when the book is unchanged."""
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None
    rule, symbol, window, band = _selection(entry, as_of)
    if symbol is None or window is None or band is None:
        return None
    return {"rule": rule, "symbol": symbol, "window": window, "band": band}


def sleeve_cash_signal_kwargs(iterations_path, spy_close: pd.Series, as_of, closes_by_symbol: dict | None) -> dict:
    """Keyword arguments for ``build_trend_signal``, or ``{}`` when nothing is kept.

    ``closes_by_symbol`` is consulted only when a kept pick names an asset.
    The state series stops at ``as_of``.
    """
    visible = spy_close.dropna().astype(float).sort_index()
    stamp = visible.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    choice = kept_sleeve_cash_choice(iterations_path, stamp)
    if choice is None:
        return {}
    symbol = choice["symbol"]
    if not closes_by_symbol or symbol not in closes_by_symbol:
        raise DataError(f"kept sleeve diversifier needs {symbol} closes")
    visible = visible.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    on, ready = sleeve_cash_state(
        pd.Series(closes_by_symbol[symbol]).loc[:stamp],
        choice["window"], choice["band"], visible.index,
    )
    info = {
        "sleeve_cash_rule": choice["rule"],
        "sleeve_cash_symbol": symbol,
        "sleeve_cash_window": int(choice["window"]),
        "sleeve_cash_band": float(choice["band"]),
        "sleeve_cash_filter_on": bool(on.iloc[-1]),
        "sleeve_cash_ready": bool(ready.iloc[-1]),
    }
    return {
        "sleeve_cash_on": on,
        "sleeve_cash_ready": ready,
        "sleeve_cash_symbol": symbol,
        "sleeve_cash_info": info,
    }
