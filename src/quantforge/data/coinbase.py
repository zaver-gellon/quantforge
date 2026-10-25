"""Coinbase Exchange public candles. No API key.

Rows are ``[time, low, high, open, close, volume]`` with at most 300 bars per
request. Timestamps are UTC.
"""

from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd

from quantforge.data.base import DataError
from quantforge.data.http import get_json

API = "https://api.exchange.coinbase.com"
GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}


def parse_coinbase_candles(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    frame["ts"] = pd.to_datetime(frame["ts"], unit="s", utc=True).dt.tz_localize(None)
    frame = frame.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    out = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    out.index.name = "date"
    return out.dropna(subset=["close"])


class CoinbaseSource:
    name = "coinbase"

    def __init__(self, pause_seconds: float = 0.15, max_tries: int = 5):
        self.pause_seconds = pause_seconds
        self.max_tries = max_tries

    def fetch_bars(self, symbols, start, end, interval="1d"):
        if interval not in GRANULARITY:
            raise DataError(f"unsupported Coinbase interval {interval}")
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        out = {}
        errors = []
        for symbol in symbols:
            try:
                out[symbol] = self._fetch_product(symbol, GRANULARITY[interval], start, end)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: {exc}")
        if errors:
            raise DataError("Coinbase fetch failed for " + "; ".join(errors))
        return out

    def _fetch_product(self, product: str, granularity: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows: list = []
        cursor = start
        span = timedelta(seconds=granularity * 300)
        while cursor < end:
            stop = min(cursor + span, end)
            params = (
                f"granularity={granularity}"
                f"&start={_iso(cursor)}&end={_iso(stop)}"
            )
            url = f"{API}/products/{product}/candles?{params}"
            payload = self._get(url)
            if not isinstance(payload, list):
                raise DataError(f"unexpected Coinbase payload for {product}: {payload}")
            rows.extend(payload)
            cursor = stop
            time.sleep(self.pause_seconds)
        frame = parse_coinbase_candles(rows)
        if frame.empty:
            raise DataError(f"no Coinbase candles for {product}")
        return frame[(frame.index >= start) & (frame.index <= end)]

    def _get(self, url: str):
        last: Exception | None = None
        for attempt in range(self.max_tries):
            try:
                return get_json(url)
            except DataError as exc:
                last = exc
                time.sleep(self.pause_seconds * (attempt + 1))
        raise DataError(str(last))


def _iso(stamp: pd.Timestamp) -> str:
    return pd.Timestamp(stamp).strftime("%Y-%m-%dT%H:%M:%SZ")
