"""Yahoo Finance daily bars via yfinance, with the public chart API as fallback.

Stocks and ETFs are total-return adjusted (OHLC scaled by the close adjustment
factor). ``^IRX`` is the 13-week T-bill yield in percent, not a price.
"""

from __future__ import annotations

import urllib.parse

import pandas as pd

from quantforge.data.base import DataError
from quantforge.data.http import get_json

_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case an OHLC(V) frame and drop rows with no close."""
    df = frame.copy()
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]
    rename = {"adjclose": "adjclose"}
    df = df.rename(columns=rename)
    if "close" not in df.columns and "adjclose" in df.columns:
        df["close"] = df["adjclose"]
    if "close" not in df.columns:
        raise DataError(f"no close column in {list(frame.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = pd.NA
    out = df[_COLUMNS].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    out.index.name = "date"
    out = out.dropna(subset=["close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    """Parse a ``/v8/finance/chart`` payload into adjusted OHLCV."""
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not chart:
        raise DataError("Yahoo chart payload has no chart object")
    if chart.get("error"):
        err = chart["error"]
        raise DataError(f"Yahoo chart error: {err.get('description') or err}")
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise DataError("Yahoo chart payload has no timestamps")
    quote = (result.get("indicators") or {}).get("quote") or [{}]
    q = quote[0]
    adj_block = (result.get("indicators") or {}).get("adjclose") or [{}]
    adj = (adj_block[0] or {}).get("adjclose") if adj_block else None
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    df = pd.DataFrame(
        {
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
        },
        index=idx,
    )
    df.index = df.index.normalize().tz_localize(None)
    df.index.name = "date"
    if adj is not None:
        factor = pd.Series(adj, index=df.index) / df["close"].replace(0, pd.NA)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    return normalize_ohlcv(df)


def _extract_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol in set(frame.columns.get_level_values(0)):
            frame = frame[symbol]
        elif symbol in set(frame.columns.get_level_values(-1)):
            frame = frame.xs(symbol, axis=1, level=-1)
        else:
            raise DataError(f"yfinance frame has no {symbol}")
    return normalize_ohlcv(frame)


class YahooSource:
    name = "yahoo"

    def __init__(self, session=None):
        self.session = session

    def fetch_bars(self, symbols, start, end, interval="1d"):
        if interval != "1d":
            raise DataError("YahooSource currently fetches daily bars only")
        start = pd.Timestamp(start).tz_localize(None).normalize()
        end = pd.Timestamp(end).tz_localize(None).normalize()
        out: dict[str, pd.DataFrame] = {}
        errors: list[str] = []
        for symbol in symbols:
            try:
                out[symbol] = self._fetch_one(symbol, start, end)
            except Exception as exc:  # noqa: BLE001 — one bad symbol should not hide the rest
                errors.append(f"{symbol}: {exc}")
        if errors:
            raise DataError("Yahoo fetch failed for " + "; ".join(errors))
        return out

    def _fetch_one(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        yf_error: Exception | None = None
        try:
            return self._fetch_yfinance(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            yf_error = exc
        try:
            return self._fetch_chart(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            raise DataError(f"yfinance ({yf_error}); chart API ({exc})") from exc

    def _fetch_yfinance(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
            interval="1d",
        )
        if raw is None or len(raw) == 0:
            raise DataError("empty yfinance frame")
        frame = _extract_download(raw, symbol)
        if frame.empty:
            raise DataError("yfinance frame had no closes")
        return frame

    def _fetch_chart(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start_utc = int(pd.Timestamp(start).tz_localize("UTC").timestamp())
        end_utc = int((pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
        quoted = urllib.parse.quote(symbol, safe="")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{quoted}"
            f"?period1={start_utc}&period2={end_utc}&interval=1d&events=div%7Csplit"
        )
        return parse_yahoo_chart(get_json(url))
