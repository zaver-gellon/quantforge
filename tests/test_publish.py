"""Signal publisher shape. No network."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quantforge.data.base import DataError
from quantforge.publish import (
    annotate_lead,
    assess_risk,
    check_selection_year,
    load_positions,
    write_bundle,
    write_risk,
)
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


def _lead(spy_close, exit_level):
    return {
        "date": "2026-11-16",
        "as_of": "2026-11-13",
        "signal_date": "2026-11-13",
        "levels": {
            "exit": f"SPY close below {exit_level:.4f}",
            "exit_level": exit_level,
            "reference_close": spy_close,
        },
        "diagnostics": {"reference_close": spy_close, "trend_on": True},
    }


def test_example_upro_stop_is_too_tight_and_total_risk_is_about_5_83(tmp_path):
    """The 2026-11-16 book, with SPY near 738.8 and the exit near 715."""
    rows = json.loads(Path("positions.example.json").read_text())
    assert [row["ticker"] for row in rows] == ["SSO", "UPRO"]
    assert rows[0]["qty"] == pytest.approx(0.2952)
    assert rows[1]["qty"] == pytest.approx(0.1460)
    assert rows[0]["equity"] == pytest.approx(100)
    path = tmp_path / "positions.json"
    path.write_text(json.dumps(rows))
    report = write_risk(tmp_path, _lead(738.8, 715.0), path)
    stored = json.loads((tmp_path / "risk.json").read_text())
    assert stored["spy_close"] == pytest.approx(738.8)
    assert stored["exit_level"] == pytest.approx(715)
    upro = next(row for row in report["positions"] if row["ticker"] == "UPRO")
    sso = next(row for row in report["positions"] if row["ticker"] == "SSO")
    upro_move = ((130.07 - 113.55) / 130.07) / 3
    assert upro["dollar_risk"] == pytest.approx(0.1460 * (130.07 - 113.55))
    assert upro["leverage"] == 3
    assert upro["implied_spy"] == pytest.approx(738.8 * (1.0 - upro_move))
    assert upro["too_tight"] is True
    assert sso["leverage"] == 2
    assert sso["too_tight"] is False
    assert any(flag.startswith("UPRO ") for flag in report["flags"])
    assert not any(flag.startswith("SSO ") for flag in report["flags"])
    expected = 0.2952 * (64.34 - 52.76) + 0.1460 * (130.07 - 113.55)
    assert report["accounts"][0]["account"] == "live"
    assert report["accounts"][0]["dollar_risk"] == pytest.approx(5.83, abs=0.01)
    assert report["accounts"][0]["dollar_risk"] == pytest.approx(expected)
    assert report["accounts"][0]["equity"] == pytest.approx(100)
    assert report["accounts"][0]["risk_pct"] == pytest.approx(expected / 100, abs=1e-6)
    assert stored["positions"][1]["too_tight"] is True
    assert not (tmp_path / "trend-core.json").exists()


def test_stop_on_the_two_percent_boundary_is_too_tight():
    on_band = assess_risk(
        [{"account": "live", "ticker": "SPY", "qty": 1, "fill": 100, "stop": 98}],
        100.0,
        100.0,
    )
    past_band = assess_risk(
        [{"account": "live", "ticker": "SPY", "qty": 1, "fill": 100, "stop": 97}],
        100.0,
        100.0,
    )
    other = assess_risk(
        [{"account": "live", "ticker": "QQQ", "qty": 2, "fill": 50, "stop": 45}],
        100.0,
        100.0,
    )
    assert on_band["positions"][0]["too_tight"] is True
    assert on_band["positions"][0]["implied_spy"] == pytest.approx(98)
    assert on_band["flags"]
    assert past_band["positions"][0]["too_tight"] is False
    assert past_band["flags"] == []
    # QQQ is not a SPY multiple, so the whole stop percent is the SPY move.
    assert other["positions"][0]["leverage"] == 1
    assert other["positions"][0]["implied_spy_move"] == pytest.approx(0.1)
    assert other["positions"][0]["dollar_risk"] == pytest.approx(10)
    assert other["accounts"][0]["risk_pct"] is None


def test_account_totals_keep_each_book_and_one_equity():
    report = assess_risk(
        [
            {"account": "a", "ticker": "SPY", "qty": 2, "fill": 10, "stop": 8, "equity": 100},
            {"account": "b", "ticker": "SPY", "qty": 1, "fill": 10, "stop": 6},
        ],
        100.0,
        50.0,
    )
    assert [row["account"] for row in report["accounts"]] == ["a", "b"]
    assert report["accounts"][0]["dollar_risk"] == pytest.approx(4)
    assert report["accounts"][0]["risk_pct"] == pytest.approx(0.04)
    assert report["accounts"][1]["equity"] is None
    assert report["accounts"][1]["risk_pct"] is None
    assert report["accounts"][1]["dollar_risk"] == pytest.approx(4)


def test_conflicting_equity_is_refused():
    try:
        assess_risk(
            [
                {"account": "a", "ticker": "SPY", "qty": 1, "fill": 10, "stop": 9, "equity": 100},
                {"account": "a", "ticker": "SSO", "qty": 1, "fill": 10, "stop": 9, "equity": 50},
            ],
            100.0,
            50.0,
        )
    except DataError as exc:
        assert "equity" in str(exc)
    else:
        raise AssertionError("conflicting equity must be refused")


def test_missing_positions_file_writes_an_empty_list_and_a_note(tmp_path):
    lead = {
        "date": "2026-11-16",
        "diagnostics": {"reference_close": 738.8},
        "levels": {"exit_level": 715.0},
    }
    report = write_risk(tmp_path, lead, tmp_path / "missing.json")
    stored = json.loads((tmp_path / "risk.json").read_text())
    assert stored["positions"] == []
    assert stored["accounts"] == []
    assert stored["flags"] == []
    assert "no positions file" in stored["note"]
    assert report["spy_close"] == pytest.approx(738.8)
    assert report["exit_level"] == pytest.approx(715)


def test_broken_positions_file_is_refused_and_writes_nothing(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text("[")
    lead = _lead(738.8, 715.0)
    try:
        write_risk(tmp_path, lead, path)
    except DataError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError("broken positions JSON must be refused")
    assert not (tmp_path / "risk.json").exists()
    path.write_text('{"account": "live"}')
    try:
        load_positions(path)
    except DataError as exc:
        assert "list" in str(exc)
    else:
        raise AssertionError("a positions object must be refused")


def test_risk_file_does_not_change_the_signal_schema(tmp_path):
    lead = {
        "date": "2026-11-16",
        "weights": {"SPY": 0.5, "UPRO": 0.5},
        "strategy": "trend_core",
        "signal_date": "2026-11-13",
        "diagnostics": {
            "trend_on": True,
            "dip_on": False,
            "reference_close": 738.8,
            "sma": 729.0,
            "band": 0.02,
        },
    }
    lead = annotate_lead(lead, {"lead": {"validated": True}})
    write_bundle(tmp_path, lead, {})
    before = (tmp_path / "trend-core.json").read_text()
    combined_before = (tmp_path / "signals.json").read_text()
    write_risk(tmp_path, lead, tmp_path / "missing.json")
    assert (tmp_path / "trend-core.json").read_text() == before
    assert (tmp_path / "signals.json").read_text() == combined_before
    validate_signal(json.loads(before), allowed_tickers=ALLOWED_TREND_TICKERS)
    assert "too_tight" not in combined_before


def test_publisher_help_documents_positions():
    proc = subprocess.run(
        [sys.executable, "-m", "quantforge.publish", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--positions" in proc.stdout
