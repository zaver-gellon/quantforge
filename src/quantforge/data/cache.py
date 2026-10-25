"""Parquet cache in front of any DataSource."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError, DataSource


def default_cache_dir() -> Path:
    env = os.environ.get("QUANTFORGE_CACHE")
    if env:
        return Path(env)
    return Path.cwd() / ".cache" / "quantforge"


def _safe_symbol(symbol: str) -> str:
    out = []
    for ch in symbol:
        out.append(ch if ch.isalnum() or ch in "-_" else "_")
    return "".join(out)


def cache_path(root: Path, source: str, symbol: str, interval: str) -> Path:
    return root / source / interval / f"{_safe_symbol(symbol)}.parquet"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = "date"
    return frame


def read_cache(root: Path, source: str, symbol: str, interval: str) -> pd.DataFrame | None:
    path = cache_path(root, source, symbol, interval)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return _normalize_index(frame)


def write_cache(root: Path, source: str, symbol: str, interval: str, frame: pd.DataFrame) -> None:
    path = cache_path(root, source, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_index(frame).to_parquet(path)


def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    new = _normalize_index(new)
    if old is None or old.empty:
        return new
    both = pd.concat([old, new])
    return _normalize_index(both)


def _covers(cached: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if cached.empty:
        return False
    return cached.index[0] <= start + pd.Timedelta(days=7) and cached.index[-1] >= end - pd.Timedelta(days=5)


def load_bars(
    source: DataSource,
    symbols: list[str],
    start,
    end=None,
    interval: str = "1d",
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return cached bars, fetching only symbols whose cache does not cover ``[start, end]``."""
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize() if end is not None else pd.Timestamp.today().normalize()
    if end_ts < start_ts:
        raise DataError("end is before start")
    found: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for symbol in symbols:
        cached = None if refresh else read_cache(root, source.name, symbol, interval)
        if cached is not None and _covers(cached, start_ts, end_ts):
            found[symbol] = cached[(cached.index >= start_ts) & (cached.index <= end_ts)]
        else:
            need.append(symbol)
    if need:
        fetched = source.fetch_bars(need, start_ts, end_ts, interval)
        missing = [symbol for symbol in need if symbol not in fetched or fetched[symbol] is None or fetched[symbol].empty]
        if missing:
            raise DataError(f"{source.name} returned no bars for {', '.join(missing)}")
        for symbol in need:
            merged = _merge(read_cache(root, source.name, symbol, interval), fetched[symbol])
            write_cache(root, source.name, symbol, interval, merged)
            sliced = merged[(merged.index >= start_ts) & (merged.index <= end_ts)]
            if sliced.empty:
                raise DataError(f"{source.name} has no {symbol} bars between {start_ts.date()} and {end_ts.date()}")
            found[symbol] = sliced
    return found
