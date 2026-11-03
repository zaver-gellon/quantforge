#!/usr/bin/env python3
"""One walk-forward test: size the dip, and take it only while the trend is on.

The gates, the SSO fractions, and the keep rule are fixed in
``quantforge.strategies.dipsize`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The ungated full-swap cell is the current book and is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/dip_size.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_dip_size
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.strategies.confirm import kept_confirm_trend
from quantforge.strategies.corehold import kept_core_hold
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.curve import kept_curve_clock
from quantforge.strategies.dipsize import (
    BOOSTS,
    GATES,
    LIVE_BOOST,
    LIVE_GATE,
    cells,
    dip_boost,
    is_live_cell,
    kept_dip_size,
    variant_name,
)
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.idle import idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-02-yield-curve-clock"
ITERATION_ID = "2026-11-03-dip-size-gate"
ITERATION_DATE = "2026-11-03"
STUDY = "spy_upro_trend"
FAMILY = "dip_size"

HYPOTHESIS = (
    "The live overlay swaps the entire SPY half into 2x SSO after three strictly "
    "down closes, then leaves at the first up close or five sessions. It does that "
    "whether or not the 200-day band is on. The study's recorded full-sample table "
    "already shows the cost of that choice: the 50/50 book without the overlay has "
    "Sharpe 0.53 and max drawdown -46.1% with turnover 0.8 a year, and the same book "
    "with the overlay has Sharpe 0.52 and max drawdown -46.8% with turnover 13.6. "
    "The later iterations tried to repair the -46.8% trough on 2009-03-09 by removing "
    "equity: a faster UPRO exit, cash in place of the core whenever the band is off, "
    "or cash for the whole book after a curve inversion. Those cuts reached the trough "
    "by selling the SPY half, and the rebound they sat out made the improvement's "
    "Sharpe fail the deflated-Sharpe cutoff and pushed the CAGR give-up past half a "
    "point. This test does not sell the core and does not change the sleeve's entry. "
    "It changes the overlay that adds leverage on down closes. A new dip starts only "
    "when the 200-day 2% band is on at that close. If the band turns off while a dip "
    "is held, the dip flattens at that close and does not re-enter until a fresh "
    "three-down signal with the band on. The first-up exit and the five-session cap "
    "stay. The original protocol already scored five entry rules at a full SSO swap; "
    "this grid does not rescore those entries. While a dip is on, only a fixed "
    "fraction of the core pile is SSO and the rest stays SPY. Fractions are 0.25, "
    "0.50, 0.75, and 1. Gates are ungated and trend-on. The ungated fraction 1 is "
    "the live book, already counted, and is the walk-forward's option to keep the "
    "current book. The other seven cells are new trials. Fills stay next-open with "
    "5 bp one-way costs. Cash pays no spread. The stitched path is kept only when "
    "it clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_dip_size", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(gate: str, boost: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "gate": gate,
        "boost": float(boost),
        "max_hold": 5,
        "exit": "first_up_close_or_max_hold",
        "rule": "3_down_closes",
        "apply": "next_open",
        "action": "size_sso_share_of_core",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
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
        note = "The prior yield-curve iteration row was not in iterations.json, so the refresh has no comparison row."
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


def _simulate(study, bundle, trend, boost, index, sso, upro):
    aligned = boost.reindex(bundle["cal"]).astype(float)
    if aligned.isna().any():
        raise RuntimeError("dip boost does not cover the price calendar")
    result = run_dip_size(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, aligned, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def _note(result: dict, kept: bool, last_pick: str, refresh: dict, extra: str) -> str:
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
        f"Last walk-forward pick: {last_pick}. {extra}"
    )
    if kept:
        ending = "The stitched sized dip clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched sized dip does not clear the materiality bar. "
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
    core_hold, core_hold_info = kept_core_hold(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    curve_on, curve_info = kept_curve_clock(
        HERE / "iterations.json", bundle["curve_spread"], as_of=as_of,
    )
    equity_trend = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    sized, dip_info = kept_dip_size(
        HERE / "iterations.json", bundle["spy_close"], equity_trend, as_of=as_of,
    )
    dip_kwargs = {}
    if sized is not None:
        dip_kwargs = {"dip_boost": sized, "dip_size_info": dip_info}
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=trend_override, vote_info=vote_info,
        confirm_info=confirm_info,
        dd_info=dd_info,
        core_hold=core_hold, core_hold_info=core_hold_info,
        curve_on=curve_on, curve_info=curve_info,
        **dip_kwargs,
        **idle_kwargs,
    )
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_kwargs.get("idle_symbol"):
        allowed.add(idle_kwargs["idle_symbol"])
    write_signal(payload, HERE / "signals.json", allowed_tickers=frozenset(allowed))
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if dip_info:
        title = "Trend-core with the kept sized dip overlay"
    elif curve_info:
        title = "Trend-core that sits in T-bills on the kept yield-curve recession clock"
    elif core_hold_info:
        title = "Trend-core with the kept off-trend core hold"
    elif confirm_info:
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
        start="1990-01-01",
        refresh=True,
    )
    spy_listed = refresh_frames["SPY"]["close"].dropna()
    irx_listed = refresh_frames["^IRX"]["close"].dropna()
    sso_listed = refresh_frames["SSO"]["close"].dropna()
    upro_listed = refresh_frames["UPRO"]["close"].dropna()
    price_note = (
        f"After a forced Yahoo refresh, SPY closes run from {spy_listed.index[0].date()} "
        f"to {spy_listed.index[-1].date()} ({len(spy_listed)} sessions). "
        f"SSO ends {sso_listed.index[-1].date()}, "
        f"UPRO ends {upro_listed.index[-1].date()}, "
        f"^IRX ends {irx_listed.index[-1].date()}."
    )
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    # The curve series is loaded only so a previously kept clock can still build
    # a signal. This iteration does not score it. A missing file is an empty
    # spread; kept_curve_clock returns nothing unless an earlier entry was kept.
    bundle["curve_spread"] = pd.Series(dtype=float)
    live = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    paths = {}
    turns = {}
    boosts = {}
    for gate, boost in cells():
        name = variant_name(gate, boost)
        fraction = dip_boost(bundle["spy_close"], live, gate, boost)
        boosts[name] = fraction
        if is_live_cell(gate, boost):
            check, _check_turn = _simulate(study, bundle, live, fraction, sample_index, sso, upro)
            if float((check - baseline).abs().max()) > 1e-10:
                raise RuntimeError("the ungated full boost does not match the live trend-core path")
            paths[name] = baseline
            turns[name] = base_turn
        else:
            paths[name], turns[name] = _simulate(study, bundle, live, fraction, sample_index, sso, upro)
    changed = [
        name for name in paths
        if name != variant_name(LIVE_GATE, LIVE_BOOST)
        and float((paths[name] - baseline).abs().max()) > 1e-12
    ]
    if len(changed) != len(cells()) - 1:
        missing = [name for name in paths if name not in changed and name != variant_name(LIVE_GATE, LIVE_BOOST)]
        raise RuntimeError(f"a sized-dip cell matched the live book, so it is not a test: {missing}")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")
    gate_left = boosts[variant_name("any", 1.0)].reindex(walk.returns.index)
    gate_right = boosts[variant_name("trend_on", 1.0)].reindex(walk.returns.index)
    if gate_left.isna().any() or gate_right.isna().any():
        raise RuntimeError("the trend-gate comparison is missing out-of-sample sessions")
    n_gate = int((gate_left - gate_right).abs().gt(1e-12).sum())
    if n_gate == 0:
        raise RuntimeError("the trend gate never changes the full-boost overlay in the out-of-sample window")

    pieces = []
    items = list(walk.picks.items())
    for i, (label, name) in enumerate(items):
        decision = pd.Timestamp(label)
        nxt = pd.Timestamp(items[i + 1][0]) if i + 1 < len(items) else None
        chosen = turns[name].dropna()
        if nxt is None:
            piece = chosen[chosen.index >= decision]
        else:
            piece = chosen[(chosen.index >= decision) & (chosen.index < nxt)]
        pieces.append(piece)
    stitched_turn = pd.concat(pieces)
    stitched_turn = stitched_turn[~stitched_turn.index.duplicated(keep="first")].sort_index()
    stitched_turn = stitched_turn.reindex(walk.returns.index).fillna(0.0)
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    rf = bundle["rf"]
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"At a full SSO boost, the trend gate changes the overlay on {n_gate} of "
        f"{len(walk.returns)} out-of-sample sessions."
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    for gate, boost in cells():
        name = variant_name(gate, boost)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if is_live_cell(gate, boost):
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(gate, boost),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for gate, boost in cells():
        if is_live_cell(gate, boost):
            continue
        name = variant_name(gate, boost)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(gate, boost),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(gate, boost): (gate, boost) for gate, boost in cells()}
    last_pick = list(walk.picks.values())[-1]
    cell_path = paths[last_pick]
    neighbor_paths = {
        variant_name(*cell): paths[variant_name(*cell)]
        for cell in adjacent_cells(by_name[last_pick], [list(GATES), list(BOOSTS)])
    }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="dip_size_walk_forward",
        params={
            "bar": "materiality_v1",
            "gates": list(GATES),
            "boosts": list(BOOSTS),
            "live_cell": variant_name(LIVE_GATE, LIVE_BOOST),
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
    variants = []
    for gate, boost in cells():
        name = variant_name(gate, boost)
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": not is_live_cell(gate, boost),
            "gate": gate,
            "boost": float(boost),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. Already counted." if is_live_cell(gate, boost) else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Sized trend-gated dip",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "dip_size",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "gates": list(GATES),
            "boosts": list(BOOSTS),
            "live_cell": variant_name(LIVE_GATE, LIVE_BOOST),
            "max_hold": 5,
            "rule": "3_down_closes",
            "price_sample_end": sample_end,
            "spy_close_end": str(spy_listed.index[-1].date()),
            "coverage": [
                {
                    "symbol": symbol,
                    "start": str(frame.index[0].date()),
                    "end": str(frame.index[-1].date()),
                    "n_obs": int(len(frame)),
                }
                for symbol, frame in (
                    ("SPY", spy_listed),
                    ("SSO", sso_listed),
                    ("UPRO", upro_listed),
                    ("^IRX", irx_listed),
                )
            ],
            "oos_sessions_trend_gate_changes_full_boost": n_gate,
            "oos_turnover_per_year": {
                "current": base_oos_summary["turnover_per_year"],
                "candidate": cand_oos_summary["turnover_per_year"],
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
        "dip_boost": (signal.get("diagnostics") or {}).get("dip_boost"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again, again_info = kept_dip_size(
        HERE / "iterations.json", bundle["spy_close"], live, as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if again is None or again_info is None:
            raise RuntimeError("a kept sized dip did not reload from the iteration log")
    elif again is not None:
        raise RuntimeError("a rejected sized dip reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
        "extra": extra,
        "price_note": price_note,
        "n_gate": n_gate,
        "cells_that_changed_the_path": changed,
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
