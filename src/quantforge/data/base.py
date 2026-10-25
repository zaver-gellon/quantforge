"""Source interface. Prices are never invented here; missing data is an error."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataError(RuntimeError):
    """A source failed or returned something that is not usable OHLCV."""


class DataSource(Protocol):
    """Fetch adjusted OHLCV.

    ``fetch_bars`` returns one frame per symbol. The index is a
    ``DatetimeIndex`` (timezone-naive dates for daily equity bars, UTC
    timestamps for intraday crypto). Columns are ``open, high, low, close,
    volume``. Equity frames are total-return adjusted when the source supports
    it. Yield series such as ``^IRX`` keep the published close; they are not
    prices.
    """

    name: str

    def fetch_bars(
        self,
        symbols: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        ...
