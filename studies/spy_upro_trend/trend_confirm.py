#!/usr/bin/env python3
"""One walk-forward test: wait N closes through the 200-day band.

The bands, the confirmation counts, and the keep rule are fixed in
``quantforge.strategies.confirm`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The live filter already has a 2% hysteresis band. This test adds a
consecutive-close confirmation. The 2% band with one day is the current book
and is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/trend_confirm.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.data import YahooSource, load_bars
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.strategies.confirm import (
    BANDS,
    CONFIRMS,
    WINDOW,
    cells,
    confirmed_trend_state,
    is_live_cell,
    kept_confirm_trend,
    variant_name,
)
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.idle import idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-10-30-idle-diversifier"
ITERATION_ID = "2026-10-31-trend-confirmation"
ITERATION_DATE = "2026-10-31"
STUDY = "spy_upro_trend"
FAMILY = "trend_confirm"

HYPOTHESIS = (
    "The live sleeve already uses a 2% hysteresis band: it turns on only when "
    "SPY closes above 1.02 times its 200-day average and turns off only when the "
    "close is below 0.98 times that average. That rule, and a 0% band, are already "
    "in the original grid. A single close through the band still flips the sleeve, "
    "and the new state then sticks until the other side of the band. Those "
    "one-session pierces are the whipsaw: the book pays the 5 bp UPRO cost and the "
    "leveraged overnight gap, then often sits in the new state through a reversal. "
    "This test does not rescore the 2% band by itself. It requires N consecutive "
    "closes beyond the band before the state changes. A close that is not beyond "
    "the band in the flip direction resets the count, and the previous state is "
    "kept until the count is met. Bands are 0.5%, 1%, 2%, and 3%. Confirmation "
    "counts are 1, 2, 3, 4, and 5 sessions. The cell with a 2% band and 1 day is "
    "the live filter, already counted, and is the walk-forward's option to keep "
    "the current book. Every other cell is a new trial. The SPY core, the "
    "three-down-close dip, and the quarterly mix are unchanged. Fills stay "
    "next-open with 5 bp one-way costs. The stitched path is kept only when it "
    "clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_confirm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(band: float, confirm: int) -> dict:
    return {
        "base": "dip_3_down_closes",
        "window": WINDOW,
        "band": float(band),
        "confirm_days": int(confirm),
        "rule": "consecutive_closes_beyond_band",
        "apply": "next_open",
        "action": "replace_trend_state",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "rebalance": "quarterly",
        "cost_one_way": 0.0005,
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
        note = "The prior idle-diversifier iteration row was not in iterations.json, so the refresh has no comparison row."
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


def _stitch(by_name: dict[str, pd.Series], picks: dict[str, str]) -> pd.Series:
    pieces = []
    items = list(picks.items())
    for i, (label, name) in enumerate(items):
        decision = pd.Timestamp(label)
        nxt = pd.Timestamp(items[i + 1][0]) if i + 1 < len(items) else None
        chosen = by_name[name].dropna()
        if nxt is None:
            piece = chosen[chosen.index >= decision]
        else:
            piece = chosen[(chosen.index >= decision) & (chosen.index < nxt)]
        pieces.append(piece)
    combined = pd.concat(pieces)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def _flips(state: pd.Series) -> int:
    values = state.astype(bool).to_numpy()
    if len(values) < 2:
        return 0
    return int((values[1:] != values[:-1]).sum())


def _note(result: dict, kept: bool, last_pick: str, refresh: dict, turnover_note: str) -> str:
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
        f"Last walk-forward pick: {last_pick}. {turnover_note}"
    )
    if kept:
        ending = "The stitched confirmation rule clears the materiality bar. It becomes the live trend filter."
    else:
        ending = (
            "The stitched confirmation rule does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _fmt_weights(weights: dict) -> str:
    order = ["SPY", "UPRO", "SSO", "BIL"]
    keys = [key for key in order if key in weights] + [key for key in weights if key not in order]
    if not keys:
        return "(empty)"
    return ", ".join(f"{key} {float(weights[key]):.4f}" for key in keys)


def _signal_sentence(previous: dict, current: dict) -> str:
    prev_w = previous.get("weights") or {}
    new_w = current.get("weights") or {}
    prev_diag = previous.get("diagnostics") or {}
    new_diag = current.get("diagnostics") or {}
    weight_same = prev_w == new_w
    trend_same = prev_diag.get("trend_on") == new_diag.get("trend_on")
    dip_same = prev_diag.get("dip_on") == new_diag.get("dip_on")
    return (
        f"Previous signal {previous.get('date')} (close {previous.get('signal_date')}): "
        f"{_fmt_weights(prev_w)}; trend on {prev_diag.get('trend_on')}; dip on {prev_diag.get('dip_on')}. "
        f"This signal {current.get('date')} (close {current.get('signal_date')}): "
        f"{_fmt_weights(new_w)}; trend on {new_diag.get('trend_on')}; dip on {new_diag.get('dip_on')}. "
        f"Weights changed: {not weight_same}. Trend state changed: {not trend_same}. "
        f"Dip state changed: {not dip_same}."
    )


def _write_live_signal(bundle) -> dict:
    as_of = bundle["spy_close"].index[-1]
    hedge_stress, hedge_rule, hedge_info = kept_hedge_stress(
        HERE / "iterations.json", bundle["spy_close"].index, as_of=as_of,
    )
    scale, vol_info = kept_sleeve_scale(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    dd_scale, dd_info = kept_drawdown_scale(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if dd_scale is not None:
        scale, vol_info = dd_scale, None
    vote_trend, vote_info = kept_vote_trend(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    confirm_trend, confirm_info = kept_confirm_trend(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if confirm_trend is not None and vote_trend is not None:
        raise RuntimeError("the iteration log kept both a vote and a confirmation rule")
    trend_override = confirm_trend if confirm_trend is not None else vote_trend
    if confirm_trend is not None:
        vote_info = None
    idle_choice = kept_idle_choice(HERE / "iterations.json", as_of)
    idle_kwargs = {}
    if idle_choice is not None:
        frames = load_bars(YahooSource(), [idle_choice["symbol"]], start="1993-01-01")
        idle_kwargs = idle_signal_kwargs(
            HERE / "iterations.json", bundle["spy_close"], as_of,
            {idle_choice["symbol"]: frames[idle_choice["symbol"]]["close"]},
        )
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=trend_override, vote_info=vote_info,
        confirm_info=confirm_info,
        dd_info=dd_info,
        **idle_kwargs,
    )
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_kwargs.get("idle_symbol"):
        allowed.add(idle_kwargs["idle_symbol"])
    write_signal(payload, HERE / "signals.json", allowed_tickers=frozenset(allowed))
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if confirm_info:
        title = "Trend-core with the kept N-day confirmation of the trend band"
    elif idle_kwargs.get("idle_info"):
        title = "Trend-core that leaves equities for the kept bond or gold trend"
    elif dd_info:
        title = "Trend-core with the kept trailing-peak stop on the UPRO sleeve"
    elif vote_info:
        title = "Trend-core with the kept multi-lookback vote on the UPRO sleeve"
    write_signals_note(HERE / "signals.md", [("signals.json", title, payload)])
    (HERE / "signals.md").write_text(
        (HERE / "signals.md").read_text()
        + "\n`signals_flagship.json` is written by `python studies/flagship/run.py` and is not this fixed book.\n"
    )
    return payload


def main() -> None:
    refresh_frames = load_bars(
        YahooSource(),
        ["SPY", "SSO", "UPRO", "^IRX"],
        start="1993-01-01",
        refresh=True,
    )
    spy_listed = refresh_frames["SPY"]["close"].dropna()
    price_note = (
        f"After a forced Yahoo refresh, SPY closes run from {spy_listed.index[0].date()} "
        f"to {spy_listed.index[-1].date()} ({len(spy_listed)} sessions). "
        f"SSO ends {refresh_frames['SSO']['close'].dropna().index[-1].date()}, "
        f"UPRO ends {refresh_frames['UPRO']['close'].dropna().index[-1].date()}, "
        f"^IRX ends {refresh_frames['^IRX']['close'].dropna().index[-1].date()}."
    )
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    live = trend_state(bundle["spy_close"], WINDOW, 0.02, "daily")
    if not confirmed_trend_state(bundle["spy_close"], WINDOW, 0.02, 1).equals(live):
        raise RuntimeError("one-day confirmation on the 2% band does not match the live filter")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    states = {}
    turns = {}
    paths = {}
    for band, confirm in cells():
        state = confirmed_trend_state(bundle["spy_close"], WINDOW, band, confirm)
        name = variant_name(band, confirm)
        states[name] = state
        if is_live_cell(band, confirm):
            paths[name] = baseline
            turns[name] = base_turn
            continue
        paths[name], turns[name] = hedge._simulate(study, bundle, state, sample_index, sso, upro)
    changed = [
        name for name in paths
        if name != variant_name(0.02, 1) and float((paths[name] - baseline).abs().max()) > 1e-12
    ]
    if not changed:
        raise RuntimeError("no confirmation cell changed a return versus the live filter")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")

    stitched_turn = _stitch(turns, walk.picks).reindex(walk.returns.index).fillna(0.0)
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    stitched_state = _stitch(
        {name: state.reindex(sample_index).astype(float) for name, state in states.items()},
        walk.picks,
    ).reindex(walk.returns.index)
    base_oos_state = live.reindex(walk.returns.index).astype(bool)
    rf = bundle["rf"]
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    turnover_note = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Trend-state flips on that window {_flips(base_oos_state)} -> {_flips(stitched_state.fillna(0).astype(bool))}."
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    counting = [(band, confirm) for band, confirm in cells() if not is_live_cell(band, confirm)]
    for band, confirm in counting:
        name = variant_name(band, confirm)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(band, confirm),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for band, confirm in counting:
        name = variant_name(band, confirm)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(band, confirm),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(band, confirm): (band, confirm) for band, confirm in cells()}
    last_pick = list(walk.picks.values())[-1]
    cell_path = paths[last_pick]
    neighbor_paths = {
        variant_name(*cell): paths[variant_name(*cell)]
        for cell in adjacent_cells(by_name[last_pick], [list(BANDS), list(CONFIRMS)])
    }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="trend_confirm_walk_forward",
        params={
            "bar": "materiality_v1",
            "bands": list(BANDS),
            "confirms": list(CONFIRMS),
            "window": WINDOW,
            "live_cell": variant_name(0.02, 1),
            "live_cell_counts": False,
            "score": "trailing_excess_sharpe",
            "first_year": study.WF_FIRST_YEAR,
            "min_obs": study.MIN_OBS,
            "last_pick": last_pick,
        },
        sample_start=str(walk.returns.index[0].date()),
        sample_end=str(walk.returns.index[-1].date()),
        metrics={
            **result["oos"]["candidate"],
            "turnover_per_year": cand_oos_summary["turnover_per_year"],
            "materiality_pass": kept,
        },
        counts_for_deflation=False,
    )
    ledger.save()

    year_counts = {name: 0 for name in paths}
    for pick in walk.picks.values():
        year_counts[pick] = year_counts.get(pick, 0) + 1
    base_summary = summarize(
        baseline, rf=rf.reindex(baseline.index), turnover=base_turn,
    )
    variants = []
    for band, confirm in cells():
        name = variant_name(band, confirm)
        if is_live_cell(band, confirm):
            row = base_summary
            counts = False
        else:
            row = full[name]
            counts = True
        state = states[name].reindex(sample_index).astype(bool)
        variants.append({
            "name": name,
            "counts_for_deflation": counts,
            "band": float(band),
            "confirm": int(confirm),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "trend_flips": _flips(state),
            "note": "The current book. Already counted." if not counts else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "N-day confirmation of the 200-day band",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "trend_confirm",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, turnover_note),
        "data": {
            "bands": list(BANDS),
            "confirms": list(CONFIRMS),
            "window": WINDOW,
            "price_sample_end": sample_end,
            "spy_close_end": str(spy_listed.index[-1].date()),
            "coverage": [{
                "symbol": "SPY",
                "start": str(spy_listed.index[0].date()),
                "end": str(spy_listed.index[-1].date()),
                "n_obs": int(len(spy_listed)),
            }],
            "oos_turnover_per_year": {
                "current": base_oos_summary["turnover_per_year"],
                "candidate": cand_oos_summary["turnover_per_year"],
            },
            "oos_trend_flips": {
                "current": _flips(base_oos_state),
                "candidate": _flips(stitched_state.fillna(0).astype(bool)),
            },
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
    entry["decision_note"] = entry["decision_note"] + " " + _signal_sentence(previous_signal, signal)
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": (signal.get("diagnostics") or {}).get("trend_on"),
        "dip_on": (signal.get("diagnostics") or {}).get("dip_on"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again, again_info = kept_confirm_trend(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    live_name = variant_name(0.02, 1)
    if kept and last_pick != live_name:
        if again is None or again_info is None:
            raise RuntimeError("a kept confirmation rule did not reload from the iteration log")
    elif again is not None:
        raise RuntimeError("a rejected confirmation rule reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
        "turnover_note": turnover_note,
        "price_note": price_note,
        "lead_refresh": {
            "note": refresh["note"],
            "cagr_delta": refresh["cagr_delta"],
            "sharpe_delta": refresh["sharpe_delta"],
            "max_drawdown_delta": refresh["max_drawdown_delta"],
            "display_unchanged": refresh["display_unchanged"],
            "refreshed": refresh["refreshed"],
            "through_prior_end": refresh["through_prior_end"],
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
        "price_sample_end": sample_end,
        "checks": result["checks"],
        "signal_sentence": _signal_sentence(previous_signal, signal),
        "signal": entry["signal"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
