"""Risk-off diversifier for the whole book when the equity trend is off.

The live book leaves the SPY half invested after the 200-day band has already
moved the UPRO sleeve to cash. The sleeves tested before this one (a credit
spread, a volatility scale, a multi-lookback vote, a trailing peak) only
changed that UPRO half, so the bear-market drawdown stayed in the core. This
rule, once the diversifier's own average exists, takes the core and the sleeve
out of equities together. It holds the diversifier while that asset's filter
is on, and T-bills while it is off. Until the average exists the book is the
unchanged trend core, so the years before the ETF lists are not a cash-timing
rule of their own.

The assets, windows, and bands below were fixed before this sample was scored.
Assets are in alphabetical order so a neighbor is one step on that list.
Do not add a cell after seeing the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on one axis is a neighbor. Alphabetical asset order is the axis.
ASSETS = ("GLD", "IEF", "TLT")
WINDOWS = (100, 200)
BANDS = (0.0, 0.02)
OFF_NAME = "idle_off"


def cells() -> list[tuple[str, int, float]]:
    return [(asset, window, band) for asset in ASSETS for window in WINDOWS for band in BANDS]


def variant_name(asset: str, window: int, band: float) -> str:
    points = int(round(float(band) * 100))
    return f"idle_{str(asset).lower()}_w{int(window)}_b{points}"


def align_ohlc(open_: pd.Series, close: pd.Series, calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Reindex onto ``calendar``. Gaps of up to five sessions after the first print are filled.

    Leading misses, the pre-listing stretch, stay missing. Nothing is filled
    backward. A missing open on a session that has a close uses that close,
    which is a zero gap rather than a made-up overnight move.
    """
    op = pd.Series(open_).astype(float).sort_index().reindex(calendar)
    cl = pd.Series(close).astype(float).sort_index().reindex(calendar)
    op = op.ffill(limit=5)
    cl = cl.ffill(limit=5)
    op = op.where(op.notna(), cl)
    op.name = "div_open"
    cl.name = "div_close"
    return op, cl


def idle_state(
    close: pd.Series,
    window: int,
    band: float,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """Diversifier trend and whether that filter is defined, on ``calendar``.

    Both series are false before the first completed window and wherever no
    price is available. The value on a session uses that session's close and
    earlier closes only. A hole of at most five sessions carries the last
    state; it does not look ahead.
    """
    native = pd.Series(close).astype(float).dropna().sort_index()
    if native.empty:
        raise DataError("diversifier close series is empty")
    state = trend_state(native, int(window), float(band), "daily")
    ready = native.rolling(int(window), min_periods=int(window)).mean().notna()
    state = state.reindex(calendar).ffill(limit=5).fillna(False).astype(bool)
    ready = ready.reindex(calendar).ffill(limit=5).fillna(False).astype(bool)
    priced = pd.Series(close).astype(float).sort_index().reindex(calendar).ffill(limit=5).notna()
    state = (state & priced).rename("idle_on")
    ready = (ready & priced).rename("idle_ready")
    return state, ready


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "idle_diversifier"
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
        if row.get("symbol") is None or row.get("window") is None or row.get("band") is None:
            continue
        return chosen, str(row["symbol"]), int(row["window"]), float(row["band"])
    raise DataError(f"kept idle diversifier has no parameters for {chosen}")


def kept_idle_choice(iterations_path, as_of) -> dict | None:
    """Parameters of the kept walk-forward pick, or None when the book is unchanged."""
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None
    rule, symbol, window, band = _selection(entry, as_of)
    if symbol is None or window is None or band is None:
        return None
    return {"rule": rule, "symbol": symbol, "window": window, "band": band}


def idle_signal_kwargs(iterations_path, spy_close: pd.Series, as_of, closes_by_symbol: dict | None) -> dict:
    """Keyword arguments for ``build_trend_signal``, or ``{}`` when nothing is kept.

    ``closes_by_symbol`` is consulted only when a kept pick names an asset.
    The state series stops at ``as_of``.
    """
    choice = kept_idle_choice(iterations_path, spy_close.index[-1] if as_of is None else as_of)
    if choice is None:
        return {}
    symbol = choice["symbol"]
    if not closes_by_symbol or symbol not in closes_by_symbol:
        raise DataError(f"kept idle diversifier needs {symbol} closes")
    visible = spy_close.dropna().astype(float).sort_index()
    stamp = visible.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = visible.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    on, ready = idle_state(
        pd.Series(closes_by_symbol[symbol]).loc[:stamp],
        choice["window"], choice["band"], visible.index,
    )
    info = {
        "idle_rule": choice["rule"],
        "idle_symbol": symbol,
        "idle_window": choice["window"],
        "idle_band": choice["band"],
        "idle_on": bool(on.iloc[-1]),
        "idle_ready": bool(ready.iloc[-1]),
    }
    return {
        "idle_on": on,
        "idle_ready": ready,
        "idle_symbol": symbol,
        "idle_info": info,
    }
