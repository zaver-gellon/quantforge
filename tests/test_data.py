import pandas as pd
import pytest

from quantforge.data.base import DataError
from quantforge.data.cache import load_bars
from quantforge.data.coinbase import parse_coinbase_candles
from quantforge.data.registry import get_source, register_source
from quantforge.data.yahoo import normalize_ohlcv, parse_yahoo_chart


class _MemorySource:
    name = "memory"

    def __init__(self, frames, calls):
        self.frames = frames
        self.calls = calls

    def fetch_bars(self, symbols, start, end, interval="1d"):
        self.calls.append(list(symbols))
        out = {}
        for symbol in symbols:
            frame = self.frames[symbol]
            out[symbol] = frame[(frame.index >= start) & (frame.index <= end)]
        return out


def test_cache_skips_a_second_fetch(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=30)
    frame = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100.0},
        index=idx,
    )
    calls = []
    source = _MemorySource({"SPY": frame}, calls)
    first = load_bars(source, ["SPY"], start="2020-01-02", end="2020-02-14", cache_dir=tmp_path)
    second = load_bars(source, ["SPY"], start="2020-01-06", end="2020-02-10", cache_dir=tmp_path)
    assert calls == [["SPY"]]
    assert len(first["SPY"]) == 30
    assert second["SPY"].index[0] >= pd.Timestamp("2020-01-06")
    assert list(second["SPY"].columns) == ["open", "high", "low", "close", "volume"]


def test_yahoo_chart_parser_adjusts_ohlc():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577836800, 1577923200],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 110.0],
                                "high": [111.0, 120.0],
                                "low": [99.0, 108.0],
                                "close": [110.0, 100.0],
                                "volume": [10, 11],
                            }
                        ],
                        "adjclose": [{"adjclose": [55.0, 50.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    frame = parse_yahoo_chart(payload)
    # Adjustment factor on the first bar is 55/110 = 0.5, so the open is 50.
    assert frame["close"].iloc[0] == pytest.approx(55.0)
    assert frame["open"].iloc[0] == pytest.approx(50.0)
    assert frame.index.tz is None


def test_normalize_and_coinbase_parser():
    raw = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1.5], "Close": [1.5, None], "Volume": [5, 6]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    frame = normalize_ohlcv(raw)
    assert len(frame) == 1
    assert frame["close"].iloc[0] == 1.5
    candles = parse_coinbase_candles(
        [[1577836800, 9, 12, 10, 11, 100], [1577836800, 8, 13, 10, 12, 50]]
    )
    assert len(candles) == 1
    assert candles["close"].iloc[0] == 12
    assert list(candles.columns) == ["open", "high", "low", "close", "volume"]


def test_registry_rejects_unknown_and_accepts_a_plugin():
    with pytest.raises(DataError):
        get_source("polygon-live")

    class _Dummy:
        name = "dummy"

        def fetch_bars(self, symbols, start, end, interval="1d"):
            return {}

    register_source("dummy", _Dummy)
    assert get_source("dummy").name == "dummy"
