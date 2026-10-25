"""Plug-in points for paid vendors. Neither adapter fetches data.

Implement ``fetch_bars`` and register the class::

    from quantforge.data import register_source
    from quantforge.data.adapters import AlpacaSource

    class MyAlpaca(AlpacaSource):
        def fetch_bars(self, symbols, start, end, interval="1d"):
            ...

    register_source("alpaca", MyAlpaca)

Keep credentials out of the repository. The backtest only sees OHLCV frames.
"""

from __future__ import annotations

from quantforge.data.base import DataError


class _PaidStub:
    vendor = "paid"

    def fetch_bars(self, symbols, start, end, interval="1d"):
        raise DataError(
            f"{type(self).__name__} is not implemented. "
            "Subclass it, implement fetch_bars, and register_source()."
        )


class AlpacaSource(_PaidStub):
    name = "alpaca"
    vendor = "alpaca"


class PolygonSource(_PaidStub):
    name = "polygon"
    vendor = "polygon"
