#!/usr/bin/env python3
"""One walk-forward test: scale the UPRO sleeve by trailing realized volatility.

The lookbacks, the target volatilities, and the keep rule are fixed in
``quantforge.strategies.volscale`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The 200-day band leaves the 3x sleeve on through the first part of a crash.
That lag is the -46.8% drawdown. Trailing realized volatility rises inside
that window. Scale only the UPRO sleeve by min(1, target / realized vol),
known at the close and applied at the next open. The SPY core, the
three-down-close dip, and the quarterly mix are unchanged. The unscaled
remainder of the sleeve earns T-bills. The scale never exceeds 1.

Run from the repo root:

    python studies/spy_upro_trend/vol_sleeve.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from quantforge.engine.backtest import run_vol_sleeve
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.volscale import (
    LOOKBACKS,
    OFF_NAME,
    TARGET_VOLS,
    sleeve_scale,
    variant_name,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
ITERATION_ID = "2026-10-26-vol-scaled-sleeve"
ITERATION_DATE = "2026-10-26"
STUDY = "spy_upro_trend"
FAMILY = "vol_sleeve"

HYPOTHESIS = (
    "The trend core's maximum drawdown is about -46.8% because the 200-day band "
    "keeps the 3x sleeve invested through the first leg of a crash. Trailing "
    "realized volatility of SPY rises in that window, before the slow average turns. "
    "Scale only the UPRO sleeve to min(1, target_vol / trailing sample volatility), "
    "using a close-to-close window known at the signal close and filled at the next "
    "open. The rest of the sleeve budget earns T-bills. The SPY core, the "
    "three-down-close dip, and the quarterly mix are unchanged. The scale is capped "
    "at 1. The grid is lookbacks 10, 21, and 63 sessions and target volatilities "
    "10%, 15%, 20%, and 25%, fixed before this sample is scored. The walk-forward "
    "may also select the unscaled book. The stitched path is kept only when it "
    "clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_vol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(lookback: int, target: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "lookback": int(lookback),
        "target_vol": float(target),
        "vol_annualization": 252,
        "vol_estimator": "sample_stdev_close_to_close",
        "cap": 1.0,
        "deadband": 0.0,
        "apply": "next_open",
        "action": "scale_upro_sleeve_rest_tbill",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
        "rebalance": "quarterly",
    }


def _scaled(study, bundle, trend, scale, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_vol_sleeve(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], scale,
        0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def _note(result: dict, kept: bool, last_pick: str) -> str:
    current = result["oos"]["current"]
    proposed = result["oos"]["candidate"]
    delta = result["criterion_a"]
    stats = (
        f"Out-of-sample {current['start']} to {current['end']} ({current['n_days']} sessions). "
        f"Sharpe {current['sharpe']:.6f} -> {proposed['sharpe']:.6f} "
        f"(delta {delta['sharpe_delta']:.6f}), "
        f"CAGR {current['cagr']:.6f} -> {proposed['cagr']:.6f} "
        f"(delta {delta['cagr_delta']:.6f}), "
        f"max drawdown {current['max_drawdown']:.6f} -> {proposed['max_drawdown']:.6f} "
        f"(delta {delta['max_drawdown_delta']:.6f}). "
        f"Last walk-forward pick: {last_pick}."
    )
    if kept:
        ending = "The stitched vol sleeve clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched vol sleeve does not clear the materiality bar. "
            "The live book stays the pre-hedge trend core."
        )
    return f"{stats} {ending}"


def main() -> None:
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    trend = bundle["trends"][(200, 0.02, "daily")]
    sso = synth[2]
    upro = synth[3]
    baseline, _base_turn = hedge._simulate(study, bundle, trend, sample_index, sso, upro)
    ones = pd.Series(1.0, index=bundle["cal"])
    identity, _id_turn = _scaled(study, bundle, trend, ones, sample_index, sso, upro)
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"a unit scale does not reproduce the trend core (max abs return gap {gap})")

    cells = [(lookback, target) for lookback in LOOKBACKS for target in TARGET_VOLS]
    paths = {OFF_NAME: baseline}
    for lookback, target in cells:
        scale = sleeve_scale(bundle["spy_close"], lookback, target).reindex(bundle["cal"]).fillna(1.0)
        paths[variant_name(lookback, target)], _turn = _scaled(
            study, bundle, trend, scale, sample_index, sso, upro,
        )

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
    for lookback, target in cells:
        name = variant_name(lookback, target)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(lookback, target),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for lookback, target in cells:
        name = variant_name(lookback, target)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(lookback, target),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(lookback, target): (lookback, target) for lookback, target in cells}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = {}
    else:
        cell_path = paths[last_pick]
        neighbor_paths = {
            variant_name(*cell): paths[variant_name(*cell)]
            for cell in adjacent_cells(by_name[last_pick], [LOOKBACKS, TARGET_VOLS])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="vol_sleeve_walk_forward",
        params={
            "bar": "materiality_v1",
            "lookbacks": list(LOOKBACKS),
            "target_vols": list(TARGET_VOLS),
            "include_off": True,
            "score": "trailing_excess_sharpe",
            "first_year": study.WF_FIRST_YEAR,
            "min_obs": study.MIN_OBS,
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
    variants = [{
        "name": OFF_NAME,
        "counts_for_deflation": False,
        "cagr": summarize(baseline, rf=rf.reindex(baseline.index))["cagr"],
        "sharpe": summarize(baseline, rf=rf.reindex(baseline.index))["sharpe"],
        "max_drawdown": summarize(baseline, rf=rf.reindex(baseline.index))["max_drawdown"],
        "years_selected": year_counts.get(OFF_NAME, 0),
        "note": "The current book. Already counted as dip_3_down_closes.",
    }]
    for lookback, target in cells:
        name = variant_name(lookback, target)
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": True,
            "lookback": lookback,
            "target_vol": target,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
        })

    previous_signal = json.loads((HERE / "signals.json").read_text())
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Volatility-scaled UPRO sleeve",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "vol_sleeve",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick),
        "data": {
            "lookbacks": list(LOOKBACKS),
            "target_vols": list(TARGET_VOLS),
            "cap": 1.0,
            "deadband": 0.0,
            "price_sample_end": sample_end,
            "identity_gap": gap,
        },
        "variants": variants,
        "walk_forward_picks": [[date, name] for date, name in walk.picks.items()],
        "oos_window": {
            "start": str(walk.returns.index[0].date()),
            "end": str(walk.returns.index[-1].date()),
            "n_days": int(len(walk.returns)),
        },
        "oos": result["oos"],
        "lead_oos": {
            "before": result["oos"]["current"],
            "after": result["oos"]["candidate"] if kept else result["oos"]["current"],
        },
        "materiality": result,
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "last_pick": last_pick,
        "previous_signal": {
            "date": previous_signal.get("date"),
            "weights": previous_signal.get("weights"),
            "trend_on": (previous_signal.get("diagnostics") or {}).get("trend_on"),
            "dip_on": (previous_signal.get("diagnostics") or {}).get("dip_on"),
            "hedge_on": (previous_signal.get("diagnostics") or {}).get("hedge_on"),
        },
    }
    hedge._write_outputs(entry)
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "identity_gap": gap,
        "oos": result["oos"],
        "criterion_a": result["criterion_a"],
        "criterion_b": {
            key: result["criterion_b"][key]
            for key in (
                "pass", "dsr_pass", "still_helps", "improvement_sharpe",
                "improvement_dsr", "best_year", "sr0_daily", "without_best_year",
            )
        },
        "criterion_c": {
            "pass": result["criterion_c"]["pass"],
            "detail": result["criterion_c"]["detail"],
            "neighbors": result["criterion_c"]["neighbors"],
        },
        "years": year_counts,
        "price_sample_end": sample_end,
    }, indent=2))


if __name__ == "__main__":
    main()
