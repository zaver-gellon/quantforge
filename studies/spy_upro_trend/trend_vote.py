#!/usr/bin/env python3
"""One walk-forward test: replace the UPRO gate with a multi-lookback vote.

The windows, the vote counts, the bands, and the keep rule are fixed in
``quantforge.strategies.vote`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The single 200-day band stays long through the first leg of a crash. The last
iteration scaled that sleeve by realized volatility and picked up 2.6 points of
drawdown, short of the 3-point bar, while the improvement series itself had a
negative Sharpe. Volatility scaling cuts the sleeve in high-vol rallies as well
as in breaks. A vote turns the sleeve off only when shorter trend states have
broken, and it can turn the sleeve on only when enough lookbacks agree.

Run from the repo root:

    python studies/spy_upro_trend/trend_vote.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import (
    BANDS,
    LONG_WINDOW,
    MIN_VOTES,
    OFF_NAME,
    SHORT_WINDOWS,
    cells,
    kept_vote_trend,
    variant_name,
    vote_state,
    windows_for,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-10-26-vol-scaled-sleeve"
ITERATION_ID = "2026-10-27-multi-lookback-vote"
ITERATION_DATE = "2026-10-27"
STUDY = "spy_upro_trend"
FAMILY = "trend_vote"

HYPOTHESIS = (
    "The live sleeve stays on until SPY closes through its 200-day band. That lag "
    "is why the trend-core drawdown is still about -46.8%, and why the book trailed "
    "SPY in 2020 and 2022. The 2026-10-26 volatility scale cut the sleeve whenever "
    "realized volatility was high, including in high-vol rallies, so the stitched "
    "path gained 2.6 drawdown points (short of 3) and the daily improvement had a "
    "negative Sharpe. A multi-lookback vote turns the sleeve off only when shorter "
    "trend states have broken, and it requires agreement before the sleeve is on. "
    "Each lookback uses the same hysteresis as the live filter. The grid is short "
    "windows 50 and 100 (middles 100 and 150, long window fixed at 200), minimum "
    "votes 2 and 3, and bands 0 and 2%, fixed before this sample is scored. The "
    "single-window grid already contains SMA 100, 150, and 200 on their own; it "
    "does not contain this vote. The SPY core, the three-down-close dip, and the "
    "quarterly mix are unchanged. The walk-forward may also select the current "
    "200-day band. The stitched path is kept only when it clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_vote", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(short: int, votes: int, band: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "windows": list(windows_for(short)),
        "short_window": int(short),
        "min_votes": int(votes),
        "band": float(band),
        "long_window": LONG_WINDOW,
        "freq": "daily",
        "apply": "next_open",
        "action": "replace_upro_gate_with_vote",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "rebalance": "quarterly",
    }


def _committed_signal() -> dict:
    """Signal on the parent commit. The working copy may already have been regenerated."""
    try:
        raw = subprocess.check_output(
            ["git", "show", "HEAD:studies/spy_upro_trend/signals.json"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return json.loads((HERE / "signals.json").read_text())
    return json.loads(raw)


def _prior_oos() -> dict | None:
    path = HERE / "iterations.json"
    if not path.exists():
        return None
    for row in reversed(json.loads(path.read_text())):
        if row.get("id") == PRIOR_ID:
            return (row.get("oos") or {}).get("current")
    return None


def _lead_refresh(refreshed: dict, baseline: pd.Series, rf: pd.Series) -> dict:
    prior = _prior_oos() or {}
    cagr_delta = None if prior.get("cagr") is None else float(refreshed["cagr"]) - float(prior["cagr"])
    sharpe_delta = None if prior.get("sharpe") is None else float(refreshed["sharpe"]) - float(prior["sharpe"])
    drawdown_delta = (
        None if prior.get("max_drawdown") is None
        else float(refreshed["max_drawdown"]) - float(prior["max_drawdown"])
    )
    same_display = False
    if prior:
        same_display = (
            round(float(refreshed["cagr"]) * 100, 2) == round(float(prior["cagr"]) * 100, 2)
            and round(float(refreshed["sharpe"]), 3) == round(float(prior["sharpe"]), 3)
            and round(float(refreshed["max_drawdown"]) * 100, 2) == round(float(prior["max_drawdown"]) * 100, 2)
        )
    overlap = None
    if prior.get("end"):
        cut = baseline.loc[: pd.Timestamp(prior["end"])]
        scored = summarize(cut, rf=rf.reindex(cut.index))
        overlap = {key: scored[key] for key in ("cagr", "sharpe", "max_drawdown", "n_days", "start", "end")}
    if not prior:
        note = "The prior vol-sleeve iteration row was not in iterations.json, so the refresh has no comparison row."
    elif same_display:
        note = (
            "On the write-up's rounding (CAGR and drawdown to 0.01 points, Sharpe to 0.001) "
            "the refreshed lead book matches the prior row."
        )
    else:
        note = (
            "On the write-up's rounding (CAGR and drawdown to 0.01 points, Sharpe to 0.001) "
            "the refreshed lead book differs from the prior row."
        )
    if overlap and prior:
        note += (
            f" On these prices, the lead book cut at the prior end {overlap['end']} "
            f"({overlap['n_days']} sessions) has CAGR {overlap['cagr']:.6f}, "
            f"Sharpe {overlap['sharpe']:.6f}, max drawdown {overlap['max_drawdown']:.6f}, "
            f"against the stored row CAGR {float(prior['cagr']):.6f}, "
            f"Sharpe {float(prior['sharpe']):.6f}, "
            f"max drawdown {float(prior['max_drawdown']):.6f}."
        )
    return {
        "prior_id": PRIOR_ID,
        "prior": prior,
        "refreshed": refreshed,
        "cagr_delta": cagr_delta,
        "sharpe_delta": sharpe_delta,
        "max_drawdown_delta": drawdown_delta,
        "display_unchanged": bool(same_display) if prior else None,
        "through_prior_end": overlap,
        "note": note,
    }


def _note(result: dict, kept: bool, last_pick: str, refresh: dict) -> str:
    current = result["oos"]["current"]
    proposed = result["oos"]["candidate"]
    delta = result["criterion_a"]
    fresh = refresh["refreshed"]
    stats = (
        f"Refreshed lead out-of-sample {fresh['start']} to {fresh['end']} "
        f"({fresh['n_days']} sessions): CAGR {fresh['cagr']:.6f}, "
        f"Sharpe {fresh['sharpe']:.6f}, max drawdown {fresh['max_drawdown']:.6f}. "
        f"{refresh['note']} "
        f"Candidate versus that refreshed book: Sharpe {current['sharpe']:.6f} -> {proposed['sharpe']:.6f} "
        f"(delta {delta['sharpe_delta']:.6f}), "
        f"CAGR {current['cagr']:.6f} -> {proposed['cagr']:.6f} "
        f"(delta {delta['cagr_delta']:.6f}), "
        f"max drawdown {current['max_drawdown']:.6f} -> {proposed['max_drawdown']:.6f} "
        f"(delta {delta['max_drawdown_delta']:.6f}). "
        f"Last walk-forward pick: {last_pick}."
    )
    if kept:
        ending = "The stitched vote clears the materiality bar. It becomes the live sleeve gate."
    else:
        ending = (
            "The stitched vote does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _write_live_signal(bundle) -> dict:
    as_of = bundle["spy_close"].index[-1]
    hedge_stress, hedge_rule, hedge_info = kept_hedge_stress(
        HERE / "iterations.json", bundle["spy_close"].index, as_of=as_of,
    )
    scale, vol_info = kept_sleeve_scale(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    vote_trend, vote_info = kept_vote_trend(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=vote_trend, vote_info=vote_info,
    )
    write_signal(payload, HERE / "signals.json")
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if vote_info:
        title = "Trend-core with the kept multi-lookback vote on the UPRO sleeve"
    write_signals_note(HERE / "signals.md", [("signals.json", title, payload)])
    (HERE / "signals.md").write_text(
        (HERE / "signals.md").read_text()
        + "\n`signals_flagship.json` is written by `python studies/flagship/run.py` and is not this fixed book.\n"
    )
    return payload


def main() -> None:
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    baseline_trend = bundle["trends"][(200, 0.02, "daily")]
    sso = synth[2]
    upro = synth[3]
    baseline, _base_turn = hedge._simulate(study, bundle, baseline_trend, sample_index, sso, upro)

    # A unanimous vote includes the 200-day leg, so it cannot be on while that leg is off.
    for short, votes, band in cells():
        if votes != 3:
            continue
        vote = vote_state(bundle["spy_close"], windows_for(short), band, votes)
        slow = trend_state(bundle["spy_close"], LONG_WINDOW, band, "daily")
        leaked = vote & ~slow.reindex(vote.index).fillna(False)
        if bool(leaked.any()):
            raise RuntimeError(f"{variant_name(short, votes, band)} was on while its 200-day leg was off")

    paths = {OFF_NAME: baseline}
    differ = {}
    for short, votes, band in cells():
        vote = vote_state(bundle["spy_close"], windows_for(short), band, votes).reindex(bundle["cal"]).fillna(False)
        name = variant_name(short, votes, band)
        paths[name], _turn = hedge._simulate(study, bundle, vote, sample_index, sso, upro)
        slow_band = baseline_trend.reindex(sample_index).astype(bool)
        voted = vote.reindex(sample_index).fillna(False).astype(bool)
        differ[name] = float((voted != slow_band).mean())

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
    for short, votes, band in cells():
        name = variant_name(short, votes, band)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(short, votes, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for short, votes, band in cells():
        name = variant_name(short, votes, band)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(short, votes, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(short, votes, band): (short, votes, band) for short, votes, band in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = {}
    else:
        cell_path = paths[last_pick]
        neighbor_paths = {
            variant_name(*cell): paths[variant_name(*cell)]
            for cell in adjacent_cells(by_name[last_pick], [SHORT_WINDOWS, MIN_VOTES, BANDS])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="trend_vote_walk_forward",
        params={
            "bar": "materiality_v1",
            "short_windows": list(SHORT_WINDOWS),
            "min_votes": list(MIN_VOTES),
            "bands": list(BANDS),
            "long_window": LONG_WINDOW,
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
    base_summary = summarize(baseline, rf=rf.reindex(baseline.index))
    variants = [{
        "name": OFF_NAME,
        "counts_for_deflation": False,
        "cagr": base_summary["cagr"],
        "sharpe": base_summary["sharpe"],
        "max_drawdown": base_summary["max_drawdown"],
        "years_selected": year_counts.get(OFF_NAME, 0),
        "note": "The current book. Already counted as dip_3_down_closes.",
    }]
    for short, votes, band in cells():
        name = variant_name(short, votes, band)
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": True,
            "short_window": short,
            "min_votes": votes,
            "band": band,
            "windows": list(windows_for(short)),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "sleeve_state_diff": differ[name],
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Multi-lookback vote on the UPRO sleeve",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "trend_vote",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh),
        "data": {
            "short_windows": list(SHORT_WINDOWS),
            "min_votes": list(MIN_VOTES),
            "bands": list(BANDS),
            "long_window": LONG_WINDOW,
            "price_sample_end": sample_end,
            "sleeve_state_diff": differ,
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
        "lead_refresh": refresh,
        "materiality": result,
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "last_pick": last_pick,
        "previous_signal": {
            "date": previous_signal.get("date"),
            "weights": previous_signal.get("weights"),
            "trend_on": (previous_signal.get("diagnostics") or {}).get("trend_on"),
            "dip_on": (previous_signal.get("diagnostics") or {}).get("dip_on"),
            "signal_date": previous_signal.get("signal_date"),
        },
    }
    hedge._write_outputs(entry)
    signal = _write_live_signal(bundle)
    again, again_info = kept_vote_trend(HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1])
    if kept and last_pick != OFF_NAME:
        if again is None or again_info is None:
            raise RuntimeError("a kept vote did not reload from the iteration log")
    elif again is not None:
        raise RuntimeError("a rejected vote reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
        "lead_refresh": {
            "note": refresh["note"],
            "cagr_delta": refresh["cagr_delta"],
            "sharpe_delta": refresh["sharpe_delta"],
            "max_drawdown_delta": refresh["max_drawdown_delta"],
            "display_unchanged": refresh["display_unchanged"],
            "refreshed": refresh["refreshed"],
        },
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
        "sleeve_state_diff": differ,
        "price_sample_end": sample_end,
        "signal": {
            "date": signal.get("date"),
            "signal_date": signal.get("signal_date"),
            "weights": signal.get("weights"),
            "trend_on": (signal.get("diagnostics") or {}).get("trend_on"),
            "dip_on": (signal.get("diagnostics") or {}).get("dip_on"),
        },
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
