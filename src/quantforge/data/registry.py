"""Named sources. Paid vendors register here without changing callers."""

from __future__ import annotations

from quantforge.data.base import DataError, DataSource
from quantforge.data.coinbase import CoinbaseSource
from quantforge.data.yahoo import YahooSource

_SOURCES: dict[str, type] = {
    "yahoo": YahooSource,
    "coinbase": CoinbaseSource,
}


def register_source(name: str, factory: type) -> None:
    """Register a ``DataSource`` implementation under ``name``."""
    _SOURCES[name] = factory


def get_source(name: str, **kwargs) -> DataSource:
    try:
        factory = _SOURCES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SOURCES))
        raise DataError(f"unknown source {name!r}; known sources: {known}") from exc
    return factory(**kwargs)
