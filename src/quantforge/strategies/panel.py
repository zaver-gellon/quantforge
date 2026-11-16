"""One NYSE panel for the lead book and the faster books.

SPY sessions are the clock. Other NYSE funds are placed on that clock and a
gap of up to five sessions is filled with the last price, which is known
before the gap. Leveraged funds are the daily-reset model until the listed
fund has a close-to-close return, then the listed return.
"""

from __future__ import annotations

import pandas as pd

from quantforge.strategies.drift import lagged_tbill
from quantforge.strategies.vehicles import stitched_letf

UNDERLYINGS = ("SPY", "QQQ", "TLT", "GLD")
LETFS = {
    "SSO": "SPY",
    "UPRO": "SPY",
    "QLD": "QQQ",
    "TQQQ": "QQQ",
    "TMF": "TLT",
    "UGL": "GLD",
}


def clean_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if "open" not in out.columns:
        out["open"] = out["close"]
    keep = [col for col in ("open", "high", "low", "close") if col in out.columns]
    out = out[keep].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["open", "close"])
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    return out


def _on_clock(frame: pd.DataFrame, clock: pd.DatetimeIndex) -> pd.DataFrame:
    aligned = clean_ohlc(frame).reindex(clock)
    return aligned.ffill(limit=5)


def align_close(close: pd.Series, clock: pd.DatetimeIndex) -> pd.Series:
    """Last print on or before each clock date, at most five sessions stale."""
    series = close.dropna().astype(float).sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(series.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    series = series[~series.index.duplicated(keep="last")]
    return series.reindex(clock).ffill(limit=5)


def build_panel(frames: dict[str, pd.DataFrame]) -> dict:
    spy = clean_ohlc(frames["SPY"])
    clock = spy.index
    rf = lagged_tbill(frames["^IRX"]["close"], clock)
    under = {symbol: _on_clock(frames[symbol], clock) for symbol in UNDERLYINGS}
    under["SPY"] = spy.reindex(clock)
    for column in ("high", "low"):
        if column not in under["SPY"].columns:
            under["SPY"][column] = under["SPY"]["close"]
        else:
            under["SPY"][column] = under["SPY"][column].fillna(under["SPY"]["close"])
    stitched = {}
    for symbol, parent in LETFS.items():
        parent_px = under[parent].dropna(subset=["open", "close"])
        listed_open = None
        listed_close = None
        if symbol in frames:
            listed = clean_ohlc(frames[symbol]).reindex(parent_px.index)
            listed_open = listed["open"]
            listed_close = listed["close"]
        path = stitched_letf(
            symbol,
            parent_px["open"],
            parent_px["close"],
            rf.reindex(parent_px.index),
            listed_open,
            listed_close,
        )
        stitched[symbol] = path.reindex(clock)
    btc = None
    if "BTC-USD" in frames and frames["BTC-USD"] is not None and not frames["BTC-USD"].empty:
        btc = _on_clock(frames["BTC-USD"], clock)
    return {
        "clock": clock,
        "spy": under["SPY"],
        "rf": rf,
        "under": under,
        "stitched": stitched,
        "btc": btc,
    }
