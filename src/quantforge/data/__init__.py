"""Market data: free sources, parquet cache, and a registry for later paid adapters."""

from quantforge.data.base import DataError, DataSource
from quantforge.data.cache import default_cache_dir, load_bars
from quantforge.data.coinbase import CoinbaseSource
from quantforge.data.registry import get_source, register_source
from quantforge.data.yahoo import YahooSource

__all__ = [
    "CoinbaseSource",
    "DataError",
    "DataSource",
    "YahooSource",
    "default_cache_dir",
    "get_source",
    "load_bars",
    "register_source",
]
