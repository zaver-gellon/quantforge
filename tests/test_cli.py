import json

import numpy as np
import pandas as pd
import pytest

from quantforge.cli import main
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_trend_core_cli_writes_a_valid_signal(tmp_path):
    idx = pd.bdate_range("2015-01-02", periods=500)
    level = 100 * np.exp(np.linspace(0, 0.3, len(idx)))
    frame = pd.DataFrame(
        {"open": level, "high": level * 1.01, "low": level * 0.99, "close": level, "volume": 1.0},
        index=idx,
    )
    frame.index.name = "date"
    dest_dir = tmp_path / "yahoo" / "1d"
    dest_dir.mkdir(parents=True)
    frame.to_parquet(dest_dir / "SPY.parquet")
    out = tmp_path / "signals.json"
    code = main([
        "signals", "--strategy", "trend-core",
        "--cache", str(tmp_path),
        "--start", "2015-01-02",
        "--end", "2016-06-01",
        "--out", str(out),
    ])
    assert code == 0
    payload = json.loads(out.read_text())
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert payload["strategy"] == "trend_core"
    assert set(payload["weights"]) <= ALLOWED_TREND_TICKERS
    assert sum(payload["weights"].values()) <= 1
