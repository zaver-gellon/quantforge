#!/usr/bin/env python3
"""Re-score the kept Baa credit-spread hedge on the materiality bar.

The hedge grid is the one fixed in ``credit_hedge.py``. This script does not
add a threshold. It rebuilds the same walk-forward paths and accepts the hedge
only when ``evaluate_materiality`` passes. A failure clears ``kept`` on the
2026-10-25 entry so the live signal reverts to the pre-hedge trend core.

Run from the repo root:

    python studies/spy_upro_trend/reevaluate_credit.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.credit import parse_fred_csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-10-25-credit-spread-sleeve-hedge"
ITERATION_ID = "2026-10-26-credit-hedge-materiality-recheck"
ITERATION_DATE = "2026-10-26"
STUDY = "spy_upro_trend"
FAMILY = "credit_hedge"

HYPOTHESIS = (
    "The 2026-10-25 iteration kept a walk-forward Baa credit-spread hedge because "
    "out-of-sample excess Sharpe and deflated Sharpe were each higher. That keep is "
    "re-scored on the materiality bar, on paths rebuilt from the cached prices and the "
    "saved BAA10Y snapshot. The threshold grid is unchanged. The hedge stays in the "
    "live book only if the stitched path clears every clause. Otherwise the live book "
    "reverts to the pre-hedge trend core and the hedge code stays as a reference."
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params(series_id: str, lookback: int, threshold: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "series": series_id,
        "transform": f"{lookback}_session_change",
        "threshold": threshold,
        "units": "percentage_points",
        "action": "sleeve_to_tbill",
        "information_lag": "same_close_asof",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
    }


def _note(prior: dict, result: dict, kept: bool) -> str:
    fresh_c = result["oos"]["current"]
    fresh_p = result["oos"]["candidate"]
    delta = result["criterion_a"]
    old = prior.get("oos") or {}
    old_c = old.get("current") or {}
    old_p = old.get("candidate") or {}
    old_delta = None
    if old_c.get("sharpe") is not None and old_p.get("sharpe") is not None:
        old_delta = float(old_p["sharpe"]) - float(old_c["sharpe"])
    prior_bit = (
        f"The 2026-10-25 write-up had Sharpe {old_c.get('sharpe')} -> {old_p.get('sharpe')} "
        f"(delta {old_delta}), CAGR {old_c.get('cagr')} -> {old_p.get('cagr')}, "
        f"max drawdown {old_c.get('max_drawdown')} -> {old_p.get('max_drawdown')}."
        if old_delta is not None else
        "The 2026-10-25 write-up had no stored out-of-sample block."
    )
    fresh_bit = (
        f"Fresh run {fresh_c['start']} to {fresh_c['end']} ({fresh_c['n_days']} sessions): "
        f"Sharpe {fresh_c['sharpe']:.6f} -> {fresh_p['sharpe']:.6f} "
        f"(delta {delta['sharpe_delta']:.6f}), "
        f"CAGR {fresh_c['cagr']:.6f} -> {fresh_p['cagr']:.6f} "
        f"(delta {delta['cagr_delta']:.6f}), "
        f"max drawdown {fresh_c['max_drawdown']:.6f} -> {fresh_p['max_drawdown']:.6f} "
        f"(delta {delta['max_drawdown_delta']:.6f})."
    )
    if kept:
        ending = "The fresh run clears the materiality bar, so the hedge remains the live rule."
    else:
        ending = (
            "The fresh run does not clear the materiality bar. The live book reverts to the "
            "pre-hedge trend core. The hedge study and the BAA10Y snapshot stay for reference."
        )
    return f"{prior_bit} {fresh_bit} {ending}"


def main() -> None:
    hedge = _load("credit_hedge_study", HERE / "credit_hedge.py")
    study = hedge._load_study()
    prior_entries = json.loads((HERE / "iterations.json").read_text())
    prior = next(row for row in prior_entries if row.get("id") == PRIOR_ID)
    series_id = prior["data"]["series"]
    snapshot = HERE / f"{series_id}.csv"
    spread = parse_fred_csv(snapshot.read_text())
    info = hedge.series_coverage(spread)
    if not info["covers_sample"]:
        raise RuntimeError(f"{series_id} snapshot does not cover the walk-forward sample: {info}")

    bundle, synth, sample_index = hedge._prepare_bundle(study)
    trend = bundle["trends"][(200, 0.02, "daily")]
    aligned = hedge.align_spread(spread, bundle["cal"])
    change = hedge.twenty_day_change(aligned, hedge.LOOKBACK)
    sso = synth[2]
    upro = synth[3]
    baseline, _turn = hedge._simulate(study, bundle, trend, sample_index, sso, upro)
    paths = {hedge.OFF_NAME: baseline}
    ordered = []
    for threshold in hedge.THRESHOLDS:
        name = hedge.variant_name(series_id, threshold)
        ordered.append(name)
        stress = hedge.stress_on(change, threshold).reindex(bundle["cal"]).fillna(False)
        gated = hedge.gate_trend(trend, stress)
        paths[name], _ = hedge._simulate(study, bundle, gated, sample_index, sso, upro)

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
    full = {}
    for threshold in hedge.THRESHOLDS:
        name = hedge.variant_name(series_id, threshold)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name,
            params=_params(series_id, hedge.LOOKBACK, threshold),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for threshold in hedge.THRESHOLDS:
        name = hedge.variant_name(series_id, threshold)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name,
            params=_params(series_id, hedge.LOOKBACK, threshold),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    last_pick = list(walk.picks.values())[-1]
    if last_pick == hedge.OFF_NAME:
        cell_path = None
        neighbor_paths = {}
    else:
        cell_path = paths[last_pick]
        neighbor_paths = {
            cell[0]: paths[cell[0]]
            for cell in adjacent_cells((last_pick,), (tuple(ordered),))
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="credit_hedge_materiality_recheck",
        params={
            "bar": "materiality_v1",
            "supersedes": PRIOR_ID,
            "series": series_id,
            "thresholds": list(hedge.THRESHOLDS),
            "last_pick": last_pick,
        },
        sample_start=str(walk.returns.index[0].date()),
        sample_end=str(walk.returns.index[-1].date()),
        metrics={**result["oos"]["candidate"], "materiality_pass": kept},
        counts_for_deflation=False,
    )
    ledger.save()

    year_counts = {name: 0 for name in paths}
    for pick in walk.picks.values():
        year_counts[pick] = year_counts.get(pick, 0) + 1
    variants = []
    for threshold in hedge.THRESHOLDS:
        name = hedge.variant_name(series_id, threshold)
        row = full[name]
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

    note = _note(prior, result, kept)
    entries = json.loads((HERE / "iterations.json").read_text())
    for row in entries:
        if row.get("id") != PRIOR_ID:
            continue
        row["original_decision"] = row.get("original_decision") or row.get("decision")
        row["original_decision_note"] = row.get("original_decision_note") or row.get("decision_note")
        row["kept"] = kept
        row["decision"] = "kept" if kept else "rejected"
        row["decision_note"] = note
        row["overlay"] = "credit_hedge"
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Materiality re-check of the Baa credit-spread hedge",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": False,
        "overlay": "recheck",
        "decision": "kept" if kept else "rejected",
        "decision_note": note,
        "applies_to": PRIOR_ID,
        "data": {
            "series": series_id,
            **info,
            "lookback": hedge.LOOKBACK,
            "thresholds": list(hedge.THRESHOLDS),
            "price_sample_end": sample_end,
            "source": f"Saved snapshot {snapshot.name}. FRED was not contacted.",
        },
        "variants": variants,
        "walk_forward_picks": [[date, name] for date, name in walk.picks.items()],
        "oos_window": {
            "start": str(walk.returns.index[0].date()),
            "end": str(walk.returns.index[-1].date()),
            "n_days": int(len(walk.returns)),
        },
        "oos": result["oos"],
        "materiality": result,
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "last_pick": last_pick,
    }
    kept_rows = [row for row in entries if row.get("id") != entry["id"]]
    kept_rows.append(entry)
    (HERE / "iterations.json").write_text(json.dumps(kept_rows, indent=2) + "\n")
    hedge._write_outputs(entry)
    print(json.dumps({
        "decision": entry["decision"],
        "kept_live_hedge": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
        "criterion_a": result["criterion_a"],
        "criterion_b": {
            key: result["criterion_b"][key]
            for key in (
                "pass", "dsr_pass", "still_helps", "improvement_sharpe",
                "improvement_dsr", "best_year", "sr0_daily",
            )
        },
        "criterion_c": {
            "pass": result["criterion_c"]["pass"],
            "detail": result["criterion_c"]["detail"],
            "neighbors": result["criterion_c"]["neighbors"],
        },
        "price_sample_end": sample_end,
        "spread": info,
    }, indent=2))


if __name__ == "__main__":
    main()
