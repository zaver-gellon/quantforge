import importlib.util
from pathlib import Path


def _report():
    path = Path("studies/spy_upro_trend/report.py")
    spec = importlib.util.spec_from_file_location("spy_upro_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(cagr):
    return {
        "cagr": cagr, "vol": 0.1, "sharpe": 0.5, "max_drawdown": -0.2,
        "worst_year": 2022, "worst_year_return": -0.11, "turnover_per_year": 0.4,
        "ir_t": 1.2, "psr": 0.8, "dsr": 0.4, "maxdd_trough": "2009-03-09",
    }


def test_report_prints_numbers_from_the_payload():
    report = _report()
    planted = 0.127
    metrics = _metrics(planted)
    results = {
        "protocol_md": "Protocol text.",
        "caveats": ["A caveat."],
        "sample": {"start": "2000-01-03", "end": "2026-10-23", "n_days": 6000, "years": 23.8},
        "costs_one_way_bps": 5.0,
        "financing_spread": 0.004,
        "irx_unit": "percent",
        "n_trials": 10,
        "multiple_testing_bar_t": 2.5,
        "sharpe_dispersion_daily": 0.01,
        "validation": [{
            "etf": "UPRO", "start": "2009-06-26", "n_days": 4000, "years": 16.0,
            "actual_cagr": 0.30, "synthetic_cagr": 0.31, "daily_corr": 0.998,
            "tracking_error": 0.03, "trend_synth_cagr": 0.19, "trend_stitched_cagr": 0.18,
        }],
        "headline": [
            ("BH_SPY", _metrics(0.08)),
            ("BH_SPY_walk_window", _metrics(0.081)),
            ("desk_50_50", metrics),
            ("desk_50_50_stitched", _metrics(0.126)),
            ("walk_forward", {**_metrics(0.09), "start": "2006-01-03", "end": "2026-10-23"}),
        ],
        "walk_forward": {"picks": [("2006-01-03", "cell_a")], "frequency": [("cell_a", 1)]},
        "subperiod_labels": ["2022"],
        "subperiods": {"BH_SPY": {"2022": {"excess": 0.0}}},
        "spy_subperiods": [("2022", -0.18)],
        "full_sample_top": [{
            "name": "cell_a", "cagr": planted, "sharpe": 0.5, "max_drawdown": -0.2,
            "ir_t": 1.2, "excess_2013": 0.02,
        }],
        "bootstrap_block": 63, "bootstrap_paths": 100, "bootstrap_seed": 7,
        "bootstrap": {"BH_SPY": {"1y": {
            "median": 5515, "p10": 4000, "p90": 7000, "p_loss": 0.2,
            "p_beats_benchmark": 0.0, "p_down_30": 0.01,
        }, "5y": {
            "median": 8000, "p10": 5000, "p90": 12000, "p_loss": 0.1,
            "p_beats_benchmark": 0.0, "p_down_30": 0.02,
        }}},
        "stress": {"desk_50_50": {
            "cost_0": 0.13, "cost_3": 0.128, "cost_5": planted, "cost_10": 0.12,
            "next_open": planted, "next_close": 0.125,
        }},
        "dip": [("dip_3_down_closes", _metrics(0.07))],
        "recommendation": {
            "action": "unproven_walk_forward",
            "edge_survives": False,
            "live_variant": "cell_a",
            "variant": "cell_a",
        },
    }
    text = report.render_report(results)
    assert "12.7%" in text
    assert "does not survive" in text
    assert "Protocol text." in text
    assert "signals.json" in text
    assert "## Iteration log" not in text


def test_iteration_log_prints_payload_numbers():
    report = _report()
    results = {
        "protocol_md": "Protocol text.",
        "caveats": ["A caveat."],
        "sample": {"start": "2000-01-03", "end": "2026-10-23", "n_days": 10, "years": 1.0},
        "costs_one_way_bps": 5.0,
        "financing_spread": 0.004,
        "irx_unit": "percent",
        "n_trials": 10,
        "multiple_testing_bar_t": 2.5,
        "sharpe_dispersion_daily": 0.01,
        "validation": [],
        "headline": [
            ("BH_SPY", _metrics(0.08)),
            ("BH_SPY_walk_window", _metrics(0.081)),
            ("desk_50_50", _metrics(0.12)),
            ("desk_50_50_stitched", _metrics(0.12)),
            ("walk_forward", {**_metrics(0.09), "start": "2006-01-03", "end": "2026-10-23"}),
        ],
        "walk_forward": {"picks": [], "frequency": []},
        "subperiod_labels": [],
        "subperiods": {},
        "spy_subperiods": [],
        "full_sample_top": [],
        "bootstrap_block": 63,
        "bootstrap_paths": 10,
        "bootstrap_seed": 7,
        "bootstrap": {},
        "stress": {},
        "dip": [],
        "recommendation": {
            "action": "hold_spy",
            "edge_survives": False,
            "live_variant": "cell_a",
            "variant": "SPY",
        },
        "iterations": [{
            "date": "2026-10-25",
            "title": "Credit-spread sleeve hedge",
            "hypothesis": "Move the sleeve to T-bills when the spread widens.",
            "decision": "rejected",
            "oos_window": {"start": "2006-01-03", "end": "2026-10-23", "n_days": 100},
            "oos": {
                "current": {"cagr": 0.111, "sharpe": 0.42, "dsr": 0.31, "max_drawdown": -0.22},
                "candidate": {"cagr": 0.105, "sharpe": 0.40, "dsr": 0.28, "max_drawdown": -0.18},
            },
        }],
    }
    text = report.render_report(results)
    assert "## Iteration log" in text
    assert "11.10%" in text
    assert "10.50%" in text
    assert "Decision: rejected." in text
