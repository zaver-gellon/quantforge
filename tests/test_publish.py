"""Signal publisher shape. No network."""

import importlib.util
import json
from pathlib import Path

from quantforge.publish import annotate_lead, check_selection_year, write_bundle
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal


def _render():
    path = Path("studies/faster/report.py")
    spec = importlib.util.spec_from_file_location("faster_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_report


def test_bundle_keeps_the_lead_shape_and_writes_each_strategy(tmp_path):
    lead = {
        "date": "2026-11-16",
        "weights": {"SPY": 0.5, "UPRO": 0.5},
        "strategy": "trend_core",
        "signal_date": "2026-11-13",
        "diagnostics": {"trend_on": True, "dip_on": False, "reference_close": 754.0, "sma": 729.0, "band": 0.02},
    }
    lead = annotate_lead(lead, {"lead": {"validated": True, "validated_reason": "approved_live_book", "oos": {"cagr": 0.12}}})
    boost = {
        "date": "2026-11-16",
        "as_of": "2026-11-13",
        "strategy": "spy-rsi-boost",
        "strategy_id": "spy-rsi-boost",
        "weights": {"SPY": 1.0},
        "validated": False,
        "levels": {"rsi2": 42.0, "boost_on": False, "stop": "none", "exit_level": 750.0},
        "oos": {"cagr": 0.1, "sharpe": 0.5, "max_drawdown": -0.2, "worst_year": 2022},
    }
    combined = write_bundle(tmp_path, lead, {"spy-rsi-boost": boost})
    stored = json.loads((tmp_path / "trend-core.json").read_text())
    validate_signal(stored, allowed_tickers=ALLOWED_TREND_TICKERS)
    assert stored["date"] == "2026-11-16"
    assert stored["weights"] == {"SPY": 0.5, "UPRO": 0.5}
    assert stored["diagnostics"]["trend_on"] is True
    assert stored["validated"] is True
    assert "spy-rsi-boost" in combined["strategies"]
    assert json.loads((tmp_path / "spy-rsi-boost.json").read_text())["levels"]["boost_on"] is False
    assert set(combined["strategies"]) == {"trend-core", "spy-rsi-boost"}


def test_selection_year_rejects_a_later_close():
    try:
        check_selection_year({"selection_year": 2026}, "2027-01-04")
    except Exception as exc:
        assert "2027" in str(exc)
    else:
        raise AssertionError("a close in a later year must be refused")


def test_report_prints_the_planted_boost_cagr():
    text = _render()({
        "verdict": "Bill's since-2016 numbers are not replicated on daily bars.",
        "recent_note": "A recent note.",
        "exposure": [{
            "name": "2.0x",
            "oos": {"cagr": 0.11, "sharpe": 0.4, "max_drawdown": -0.2, "worst_year": 2008, "worst_year_return": -0.3},
            "trail": {"1m": {"trail_rate": 0.42}, "3m": {"trail_rate": 0.37}},
        }],
        "rebalance": {"decision": "rejected", "note": "Monthly."},
        "qqq": {"decision": "rejected", "note": "QQQ."},
        "strategies": [],
        "caveats": ["A caveat."],
        "replication": {
            "named_cell": "rsi10_f33_n10",
            "walk_forward_pick": "rsi10_f33_n10",
            "vs_spy": {"dsr": 0.51},
            "vs_lead": {"dsr": 0.22},
            "windows": {
                "since_2016": {
                    "next_open": {
                        "stats": {"cagr": 0.171, "sharpe": 0.8, "max_drawdown": -0.33, "worst_year": 2022, "worst_year_return": -0.2},
                        "spy": {"cagr": 0.154, "max_drawdown": -0.34},
                        "episodes": {"win_rate": 0.7, "avg_pnl": 0.004, "worst_pnl": -0.08, "per_year": 8.0, "avg_days": 3.2},
                    },
                },
            },
        },
    })
    assert "17.1%" in text
    assert "Bill's SPY + RSI(2) boost" in text
    assert "not replicated" in text
