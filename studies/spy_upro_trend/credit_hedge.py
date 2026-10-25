#!/usr/bin/env python3
"""One walk-forward test of a credit-spread hedge on the trend-core sleeve.

The threshold grid, the series rule, and the keep rule below are fixed.
Do not add a threshold after seeing this sample.

Run from the repo root:

    python studies/spy_upro_trend/credit_hedge.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.strategies.credit import (
    align_spread,
    fetch_fred_csv,
    gate_trend,
    parse_fred_csv,
    stress_on,
    twenty_day_change,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
RESULTS_PATH = HERE / "results.json"
ITERATIONS_PATH = HERE / "iterations.json"

ITERATION_ID = "2026-10-25-credit-spread-sleeve-hedge"
ITERATION_DATE = "2026-10-25"
LOOKBACK = 20
# Percentage points of spread widening over LOOKBACK sessions.
# Set for a spread whose level is a few percent, before any hedged return was scored.
THRESHOLDS = (0.10, 0.20, 0.35, 0.50)
COVERAGE_START = pd.Timestamp("1999-01-01")
COVERAGE_END = pd.Timestamp("2026-10-17")
HY_OAS_SERIES = "BAMLH0A0HYM2"
FALLBACK_SERIES = "BAA10Y"
OFF_NAME = "hedge_off"
STUDY = "spy_upro_trend"
FAMILY = "credit_hedge"

HYPOTHESIS = (
    "The live book keeps the 3x sleeve on until SPY closes through its 200-day band. "
    "That lag is the weakness in the refreshed subperiod table: the trend-core book trails "
    "SPY in 2020 and again in 2022. A widening credit spread is a faster stress signal. "
    "When the 20-session change in the spread exceeds a precommitted threshold, move only "
    "the leveraged sleeve to T-bills. The SPY core, the three-down-close dip, and the "
    "quarterly mix are unchanged. The sleeve goes back to UPRO when the trend is on and "
    "the 20-session widening is back at or below the threshold."
)


def series_coverage(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {"start": None, "end": None, "n_obs": 0, "covers_sample": False}
    start = pd.Timestamp(clean.index[0])
    end = pd.Timestamp(clean.index[-1])
    covers = bool(start <= COVERAGE_START and end >= COVERAGE_END)
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "n_obs": int(clean.shape[0]),
        "covers_sample": covers,
    }


def keep_candidate(candidate: dict, baseline: dict) -> bool:
    """Keep only when both the excess Sharpe and the deflated Sharpe improve."""
    values = (
        candidate.get("sharpe"),
        baseline.get("sharpe"),
        candidate.get("dsr"),
        baseline.get("dsr"),
    )
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return float(candidate["sharpe"]) > float(baseline["sharpe"]) and float(candidate["dsr"]) > float(baseline["dsr"])


def variant_name(series_id: str, threshold: float) -> str:
    bps = int(round(threshold * 100))
    return f"{series_id.lower()}_d20_gt_{bps}bp"


def _load_study():
    path = HERE / "run.py"
    spec = importlib.util.spec_from_file_location("spy_upro_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _headline(results: dict, name: str) -> dict | None:
    for label, metrics in results.get("headline", []):
        if label == name:
            return metrics
    return None


def _previous_results() -> dict | None:
    try:
        raw = subprocess.check_output(
            ["git", "show", "HEAD^:studies/spy_upro_trend/results.json"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return json.loads(raw)


def _refresh_block(metrics: dict | None, results: dict | None) -> dict:
    if not metrics or not results:
        return {}
    return {
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "dsr": metrics.get("dsr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "n_trials": results.get("n_trials"),
        "sharpe_dispersion_daily": results.get("sharpe_dispersion_daily"),
        "sample_end": (results.get("sample") or {}).get("end"),
    }


def _compact(metrics: dict) -> dict:
    keys = ("cagr", "vol", "sharpe", "dsr", "psr", "max_drawdown", "n_days", "start", "end", "ir_t")
    return {key: metrics.get(key) for key in keys}


def _choose_series() -> tuple[str | None, pd.Series | None, dict, dict]:
    hy_text = fetch_fred_csv(HY_OAS_SERIES)
    hy = parse_fred_csv(hy_text)
    hy_cov = series_coverage(hy)
    hy_info = {"series": HY_OAS_SERIES, "used": False, **hy_cov}
    if hy_cov["covers_sample"]:
        hy_info["used"] = True
        hy_info["reason"] = "The public series covers the walk-forward sample, so it is the spread."
        return HY_OAS_SERIES, hy, {"series": HY_OAS_SERIES, **hy_cov}, hy_info
    hy_info["reason"] = (
        f"The public CSV runs from {hy_cov['start']} to {hy_cov['end']} "
        f"({hy_cov['n_obs']} observations). A walk-forward from 2006 needs observations "
        "by 1999-01-01, so this file is not the spread in the test."
    )
    fallback_text = fetch_fred_csv(FALLBACK_SERIES)
    fallback = parse_fred_csv(fallback_text)
    cov = series_coverage(fallback)
    info = {"series": FALLBACK_SERIES, **cov}
    if cov["covers_sample"]:
        return FALLBACK_SERIES, fallback, info, hy_info
    return None, None, info, hy_info


def _simulate(study, bundle, trend, index, sso, upro):
    from quantforge.engine.backtest import run_trend_core
    from quantforge.strategies.dipbuy import dip_active

    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_trend_core(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def _prepare_bundle(study):
    from quantforge.data import YahooSource, load_bars
    from quantforge.engine.leverage import FINANCING_SPREAD, LEVERAGED_ETFS, synthesize_leveraged
    from quantforge.engine.schedule import rebalance_mask
    from quantforge.strategies.trend import trend_state

    frames = load_bars(YahooSource(), ["SPY", "SSO", "UPRO", "^IRX"], start="1993-01-01")
    spy = study._clean_prices(frames["SPY"])
    irx_pct, _unit = study._yield_series(frames["^IRX"]["close"])
    cal = spy.index
    rf = (irx_pct.reindex(cal).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    rf.index = cal
    bundle = {
        "cal": cal,
        "spy_open": spy["open"],
        "spy_close": spy["close"],
        "rf": rf,
        "trends": {(200, 0.02, "daily"): trend_state(spy["close"], 200, 0.02, "daily")},
        "masks": {"quarterly": rebalance_mask(cal, "quarterly")},
    }
    synth = {}
    for symbol, leverage in (("SSO", 2), ("UPRO", 3)):
        meta = LEVERAGED_ETFS[symbol]
        synth[leverage] = synthesize_leveraged(
            spy["open"], spy["close"], rf, leverage, meta["expense_ratio"], FINANCING_SPREAD,
        )
    sample_index = spy["close"].pct_change().loc[study.SAMPLE_START:].dropna().index
    return bundle, synth, sample_index


def _upsert(iterations: list, entry: dict) -> list:
    kept = [row for row in iterations if row.get("id") != entry["id"]]
    kept.append(entry)
    return kept


def _plain(value):
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    # NumPy scalars leak out of the metric helpers. Store plain JSON.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_outputs(entry: dict) -> None:
    existing = []
    if ITERATIONS_PATH.exists():
        existing = json.loads(ITERATIONS_PATH.read_text())
    payload = _upsert(existing, entry)
    ITERATIONS_PATH.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    results = json.loads(RESULTS_PATH.read_text())
    results["iterations"] = payload
    spec = importlib.util.spec_from_file_location("spy_upro_report", HERE / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (HERE / "RESULTS.md").write_text(module.render_report(results))


def _not_tested(fallback_info: dict, hy_info: dict, reason: str) -> dict:
    current = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    previous = _previous_results()
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Credit-spread sleeve hedge",
        "hypothesis": HYPOTHESIS,
        "status": "not_tested",
        "kept": False,
        "decision": "not tested",
        "decision_note": reason,
        "data": {**fallback_info, "hy_oas": hy_info},
        "variants": [],
        "refresh": {
            "before": _refresh_block(_headline(previous, "trend_core") if previous else None, previous),
            "after": _refresh_block(_headline(current, "trend_core"), current),
            "note": (
                "The lead strategy was not changed. No hedge trial was recorded because "
                "neither spread series covered the walk-forward sample."
            ),
        },
    }
    _write_outputs(entry)
    return entry


def main() -> None:
    series_id, spread, info, hy_info = _choose_series()
    if series_id is None or spread is None:
        reason = (
            f"{HY_OAS_SERIES} does not cover the sample ({hy_info.get('start')} to {hy_info.get('end')}). "
            f"{FALLBACK_SERIES} does not either ({info.get('start')} to {info.get('end')}). "
            "The hedge was not scored and the lead book is unchanged."
        )
        entry = _not_tested(info, hy_info, reason)
        print(json.dumps({"decision": entry["decision"], "data": entry["data"]}, indent=2))
        return

    study = _load_study()
    from quantforge.eval.ledger import TrialLedger
    from quantforge.eval.metrics import summarize
    from quantforge.eval.sharpe import deflated_sharpe

    bundle, synth, sample_index = _prepare_bundle(study)
    trend = bundle["trends"][(200, 0.02, "daily")]
    aligned = align_spread(spread, bundle["cal"])
    change = twenty_day_change(aligned, LOOKBACK)
    sso = synth[2]
    upro = synth[3]
    baseline, _base_turn = _simulate(study, bundle, trend, sample_index, sso, upro)
    current_results = json.loads(RESULTS_PATH.read_text())
    recorded = (_headline(current_results, "trend_core") or {}).get("cagr")
    from quantforge.eval.metrics import summarize as _summarize
    base_cagr = _summarize(baseline, rf=bundle["rf"].reindex(baseline.index))["cagr"]
    if recorded is None or abs(float(base_cagr) - float(recorded)) > 1e-5:
        raise RuntimeError(
            f"rebuilt trend-core CAGR {base_cagr} does not match the refreshed study {recorded}"
        )
    paths = {OFF_NAME: baseline}
    stress_by_name = {OFF_NAME: pd.Series(False, index=bundle["cal"])}
    for threshold in THRESHOLDS:
        name = variant_name(series_id, threshold)
        stress = stress_on(change, threshold).reindex(bundle["cal"]).fillna(False)
        stress_by_name[name] = stress
        gated = gate_trend(trend, stress)
        paths[name], _turn = _simulate(study, bundle, gated, sample_index, sso, upro)

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    rf = bundle["rf"]
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full_metrics = {}
    for threshold in THRESHOLDS:
        name = variant_name(series_id, threshold)
        full_metrics[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY,
            family=FAMILY,
            name=name,
            params={
                "base": "dip_3_down_closes",
                "series": series_id,
                "transform": f"{LOOKBACK}_session_change",
                "threshold": threshold,
                "units": "percentage_points",
                "action": "sleeve_to_tbill",
                "information_lag": "same_close_asof",
                "core_weight": 0.5,
                "sleeve_weight": 0.5,
                "dip_rule": "3_down_closes",
            },
            sample_start=study.SAMPLE_START,
            sample_end=sample_end,
            metrics=full_metrics[name],
            counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for threshold in THRESHOLDS:
        name = variant_name(series_id, threshold)
        scored = dict(full_metrics[name])
        scored["psr"] = study.probabilistic_sharpe(paths[name], rf=rf.reindex(paths[name].index))
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=rf.reindex(paths[name].index))
        full_metrics[name] = scored
        ledger.record(
            study=STUDY,
            family=FAMILY,
            name=name,
            params={
                "base": "dip_3_down_closes",
                "series": series_id,
                "transform": f"{LOOKBACK}_session_change",
                "threshold": threshold,
                "units": "percentage_points",
                "action": "sleeve_to_tbill",
                "information_lag": "same_close_asof",
                "core_weight": 0.5,
                "sleeve_weight": 0.5,
                "dip_rule": "3_down_closes",
            },
            sample_start=study.SAMPLE_START,
            sample_end=sample_end,
            metrics=scored,
            counts_for_deflation=True,
        )

    def _oos_metrics(series: pd.Series) -> dict:
        window_rf = rf.reindex(series.index)
        window_spy = spy_r.reindex(series.index)
        metrics = summarize(series, rf=window_rf, benchmark=window_spy)
        metrics["psr"] = study.probabilistic_sharpe(series, rf=window_rf)
        metrics["dsr"] = deflated_sharpe(series, n_trials, dispersion, rf=window_rf)
        return metrics

    current_metrics = _oos_metrics(current_oos)
    candidate_metrics = _oos_metrics(walk.returns)
    kept = keep_candidate(candidate_metrics, current_metrics)
    ledger.record(
        study=STUDY,
        family="audit",
        name="credit_hedge_walk_forward",
        params={
            "series": series_id,
            "thresholds": list(THRESHOLDS),
            "include_off": True,
            "score": "trailing_excess_sharpe",
            "first_year": study.WF_FIRST_YEAR,
            "min_obs": study.MIN_OBS,
            "kept": kept,
        },
        sample_start=str(walk.returns.index[0].date()),
        sample_end=str(walk.returns.index[-1].date()),
        metrics=candidate_metrics,
        counts_for_deflation=False,
    )
    ledger.save()

    year_counts = {name: 0 for name in paths}
    for pick in walk.picks.values():
        year_counts[pick] = year_counts.get(pick, 0) + 1
    variants = [{
        "name": OFF_NAME,
        "counts_for_deflation": False,
        "cagr": None,
        "sharpe": None,
        "max_drawdown": None,
        "years_selected": year_counts.get(OFF_NAME, 0),
        "note": "The current book. Already counted as dip_3_down_closes.",
    }]
    # Full-sample stats for the off book are the refreshed trend-core row, not a new trial.
    current_results = json.loads(RESULTS_PATH.read_text())
    trend_core = _headline(current_results, "trend_core") or {}
    variants[0]["cagr"] = trend_core.get("cagr")
    variants[0]["sharpe"] = trend_core.get("sharpe")
    variants[0]["max_drawdown"] = trend_core.get("max_drawdown")
    for threshold in THRESHOLDS:
        name = variant_name(series_id, threshold)
        row = full_metrics[name]
        variants.append({
            "name": name,
            "counts_for_deflation": True,
            "threshold": threshold,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
        })

    oos_index = walk.returns.index
    pieces = []
    labels = list(walk.picks)
    trend_oos = trend.reindex(oos_index).fillna(False)
    for i, label in enumerate(labels):
        start = pd.Timestamp(label)
        stop = pd.Timestamp(labels[i + 1]) if i + 1 < len(labels) else None
        flag = stress_by_name[walk.picks[label]].reindex(oos_index).fillna(False)
        if stop is None:
            piece = flag.loc[start:]
        else:
            piece = flag.loc[(flag.index >= start) & (flag.index < stop)]
        pieces.append(piece)
    held_stress = pd.concat(pieces)
    held_stress = held_stress[~held_stress.index.duplicated(keep="first")].reindex(oos_index).fillna(False)
    active = held_stress.astype(bool) & trend_oos.astype(bool)
    hedged_days = int(active.sum())

    previous = _previous_results()
    weakness = []
    sub = (current_results.get("subperiods") or {}).get("trend_core") or {}
    for label in ("2020", "2022"):
        row = sub.get(label) or {}
        weakness.append({"label": label, "excess": row.get("excess"), "cagr": row.get("cagr")})

    if kept:
        decision = "kept"
        note = (
            "The stitched hedge beat the current book on out-of-sample excess Sharpe and on "
            "deflated Sharpe. The live book becomes this walk-forward rule."
        )
    else:
        decision = "rejected"
        note = (
            "The stitched hedge did not beat the current book on both out-of-sample excess "
            "Sharpe and deflated Sharpe. The lead strategy is unchanged."
        )
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Credit-spread sleeve hedge",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "decision": decision,
        "decision_note": (
            f"{note} Deflation uses {n_trials} counting trials and a daily-Sharpe dispersion "
            f"of {dispersion:.6f}. On the out-of-sample window the selected rule put the sleeve "
            f"in T-bills on {hedged_days} sessions while the trend filter was on."
        ),
        "data": {**info, "hy_oas": hy_info, "lookback": LOOKBACK, "thresholds": list(THRESHOLDS)},
        "recorded_weakness": weakness,
        "refresh": {
            "before": _refresh_block(_headline(previous, "trend_core") if previous else None, previous),
            "after": _refresh_block(trend_core, current_results),
            "note": (
                "Before is the trend-core row in the previous results file. After is the same "
                "row from the study rerun on prices through the latest close. The path statistics "
                "are full-sample. The out-of-sample comparison of the hedge is the table below."
            ),
        },
        "variants": variants,
        "walk_forward_picks": [[date, name] for date, name in walk.picks.items()],
        "oos_window": {
            "start": str(walk.returns.index[0].date()),
            "end": str(walk.returns.index[-1].date()),
            "n_days": int(len(walk.returns)),
        },
        "oos": {
            "current": _compact(current_metrics),
            "candidate": _compact(candidate_metrics),
        },
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "hedged_trend_on_days": hedged_days,
    }
    _write_outputs(entry)
    print(json.dumps({
        "decision": decision,
        "kept": kept,
        "series": series_id,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos_current": entry["oos"]["current"],
        "oos_candidate": entry["oos"]["candidate"],
        "picks": entry["walk_forward_picks"],
        "hedged_trend_on_days": hedged_days,
        "hy_oas": hy_info,
    }, indent=2))


if __name__ == "__main__":
    main()
