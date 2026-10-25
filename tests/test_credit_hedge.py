"""Causality of the credit-spread gate. No market data."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _hedge():
    path = Path("studies/spy_upro_trend/credit_hedge.py")
    spec = importlib.util.spec_from_file_location("credit_hedge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_align_spread_does_not_use_a_later_print():
    hedge = _hedge()
    spread = pd.Series(
        [1.0, 1.2, 3.0],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-07"]),
    )
    sessions = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    aligned = hedge.align_spread(spread, sessions)
    assert float(aligned.iloc[2]) == 1.2
    assert float(aligned.iloc[3]) == 3.0


def test_stress_is_the_trailing_change_only():
    hedge = _hedge()
    aligned = pd.Series(
        [1.0, 1.0, 1.1, 1.6],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
    )
    change = hedge.twenty_day_change(aligned, lookback=1)
    stress = hedge.stress_on(change, 0.4)
    assert bool(stress.loc["2020-01-06"]) is False
    assert bool(stress.loc["2020-01-07"]) is True
    assert bool(hedge.stress_on(change, 0.5).loc["2020-01-07"]) is False
    gated = hedge.gate_trend(pd.Series(True, index=aligned.index), stress)
    assert bool(gated.loc["2020-01-06"]) is True
    assert bool(gated.loc["2020-01-07"]) is False


def test_keep_requires_both_sharpe_and_deflated_sharpe():
    hedge = _hedge()
    baseline = {"sharpe": 0.50, "dsr": 0.40}
    assert hedge.keep_candidate({"sharpe": 0.55, "dsr": 0.45}, baseline) is True
    assert hedge.keep_candidate({"sharpe": 0.55, "dsr": 0.40}, baseline) is False
    assert hedge.keep_candidate({"sharpe": 0.50, "dsr": 0.50}, baseline) is False
    assert hedge.keep_candidate({"sharpe": float("nan"), "dsr": 0.9}, baseline) is False


def test_kept_hedge_reads_the_snapshot_and_does_not_fetch(tmp_path, monkeypatch):
    from quantforge.strategies import credit

    def boom(*_args, **_kwargs):
        raise AssertionError("FRED should not be contacted when the snapshot exists")

    monkeypatch.setattr(credit, "fetch_fred_csv", boom)
    sessions = pd.bdate_range("2020-01-02", periods=25)
    values = [1.0] * 24 + [1.4]
    lines = ["observation_date,BAA10Y"]
    lines.extend(f"{day.date()},{value}" for day, value in zip(sessions, values))
    lines.append("2020-03-02,9.99")
    (tmp_path / "BAA10Y.csv").write_text("\n".join(lines) + "\n")
    entry = {
        "kept": True,
        "status": "tested",
        "data": {"series": "BAA10Y", "lookback": 20},
        "walk_forward_picks": [["2020-01-02", "baa10y_d20_gt_10bp"]],
        "variants": [{"name": "baa10y_d20_gt_10bp", "threshold": 0.10}],
    }
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([entry]))
    stress, rule, info = credit.kept_hedge_stress(path, sessions, as_of=sessions[-1])
    assert rule == "baa10y_d20_gt_10bp"
    assert bool(stress.iloc[-1]) is True
    assert info["spread_value"] == 1.4
    assert info["spread_date"] == str(sessions[-1].date())
    assert info["spread_change"] == pytest.approx(0.4)


def test_fred_parser_drops_missing_cells():
    hedge = _hedge()
    text = "observation_date,BAA10Y\n1986-01-02,2.34\n1986-01-03,\n1986-01-06,2.29\n"
    series = hedge.parse_fred_csv(text)
    assert list(series.index.strftime("%Y-%m-%d")) == ["1986-01-02", "1986-01-06"]
    assert float(series.iloc[0]) == 2.34
