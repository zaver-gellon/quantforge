#!/usr/bin/env python3
"""One walk-forward test: re-enter the SPY core on a shorter trend.

The windows, the bands, and the keep rule are fixed in
``quantforge.strategies.reentry`` and ``quantforge.eval.materiality``. Do not
add a cell after seeing this sample.

The current book is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/fast_reentry.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_fast_reentry
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
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.dipsize import kept_dip_size
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.drift import kept_drift_mix
from quantforge.strategies.idle import idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.reentry import (
    BANDS,
    OFF_NAME,
    WINDOWS,
    cells,
    kept_fast_reentry,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-04-drift-band"
ITERATION_ID = "2026-11-05-fast-reentry"
ITERATION_DATE = "2026-11-05"
STUDY = "spy_upro_trend"
FAMILY = "fast_reentry"

HYPOTHESIS = (
    "The live book's maximum drawdown is still about -46.8%, with the trough "
    "on 2009-03-09. Faster exits of the UPRO sleeve left that trough in place. "
    "Parking the SPY half in T-bills for the entire time the 200-day band is "
    "off did cut the drawdown by more than 3 points, and it was rejected "
    "because the cash stake also sat out the rebound and the CAGR give-up was "
    "larger than half a point. Replacing that cash with bonds or gold, and "
    "parking the whole book after a curve inversion, failed for the same "
    "reason: the rebound that was missed cost more CAGR than the bar allows. "
    "The one-sided drift band and the sized dip did not touch that trough. "
    "This test does not change the sleeve's 200-day rule and does not hold a "
    "constant cash fraction for the whole off-trend episode. While the "
    "200-day 2% band is on, the book is the unchanged trend core. While that "
    "band is off, the core is in SPY, or in SSO when the three-down-close dip "
    "is on, only when a shorter trend filter is on, and in T-bills when that "
    "shorter filter is off. The sleeve stays in T-bills until the 200-day band "
    "turns back on. The shorter filter uses the same hysteresis as the live "
    "band: it turns on only above the average times one plus the band and off "
    "only below the average times one minus the band. Windows are 20, 50, and "
    "100 sessions. Bands are 0 and 2%. The six cells are new trials. The "
    "current book, which never moves the core to T-bills, is the "
    "walk-forward's option to keep the present rule and is not a new trial. "
    "A constant off-trend cash fraction is not in this grid. Fills stay "
    "next-open with 5 bp one-way costs. Cash pays no spread. The stitched "
    "path is kept only when it clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_reentry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(window: int, band: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "reentry_window": int(window),
        "reentry_band": float(band),
        "slow_trend": "sma200_band2_daily",
        "applies_when": "slow_trend_off",
        "core_when_fast_on": 1.0,
        "core_when_fast_off": 0.0,
        "sleeve": "unchanged_until_slow_trend",
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
        note = "The prior drift-band iteration row was not in iterations.json, so the refresh has no comparison row."
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


def _simulate(study, bundle, slow, fast, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_fast_reentry(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        slow, fast, dip, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def _stitch(series_by_name: dict[str, pd.Series], picks: dict, index: pd.Index) -> pd.Series:
    pieces = []
    items = list(picks.items())
    for i, (label, name) in enumerate(items):
        decision = pd.Timestamp(label)
        nxt = pd.Timestamp(items[i + 1][0]) if i + 1 < len(items) else None
        chosen = series_by_name[name].dropna()
        if nxt is None:
            piece = chosen[chosen.index >= decision]
        else:
            piece = chosen[(chosen.index >= decision) & (chosen.index < nxt)]
        pieces.append(piece)
    stitched = pd.concat(pieces)
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()
    return stitched.reindex(index)


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
        ending = "The stitched fast re-entry clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched fast re-entry does not clear the materiality bar. "
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


def _write_live_signal(bundle, sso, upro) -> dict:
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
    drift_mix, drift_info = kept_drift_mix(
        HERE / "iterations.json",
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        bundle["rf"], as_of=as_of,
    )
    reentry_hold, reentry_info = kept_fast_reentry(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    if reentry_hold is not None and core_hold is not None:
        raise RuntimeError("the iteration log kept both an off-trend core hold and a fast re-entry")
    if reentry_hold is not None:
        core_hold, core_hold_info = reentry_hold, reentry_info
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=trend_override, vote_info=vote_info,
        confirm_info=confirm_info,
        dd_info=dd_info,
        core_hold=core_hold, core_hold_info=core_hold_info,
        curve_on=curve_on, curve_info=curve_info,
        drift_mix=drift_mix, drift_info=drift_info,
        **dip_kwargs,
        **idle_kwargs,
    )
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_kwargs.get("idle_symbol"):
        allowed.add(idle_kwargs["idle_symbol"])
    write_signal(payload, HERE / "signals.json", allowed_tickers=frozenset(allowed))
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if drift_info and not drift_info.get("drift_calendar", True):
        title = "Trend-core that resets the mix on the kept UPRO drift band"
    elif dip_info:
        title = "Trend-core with the kept sized dip overlay"
    elif curve_info:
        title = "Trend-core that sits in T-bills on the kept yield-curve recession clock"
    elif reentry_info and not reentry_info.get("reentry_live", False):
        title = "Trend-core that re-enters the SPY half on a shorter trend while the 200-day band is off"
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
    bundle["curve_spread"] = pd.Series(dtype=float)
    live = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    always = pd.Series(True, index=bundle["cal"])
    always_path, _always_turn = _simulate(study, bundle, live, always, sample_index, sso, upro)
    if float((always_path - baseline).abs().max()) > 1e-12:
        raise RuntimeError("a fast filter that is always on did not match the trend core")
    never = pd.Series(False, index=bundle["cal"])
    never_path, _never_turn = _simulate(study, bundle, live, never, sample_index, sso, upro)
    if float((never_path - baseline).abs().max()) <= 1e-12:
        raise RuntimeError("parking the core for every off-trend session matched the live book")

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    fast_states = {OFF_NAME: always}
    for window, band in cells():
        name = variant_name(window, band)
        fast = trend_state(bundle["spy_close"], window, band, "daily").reindex(bundle["cal"]).fillna(False)
        returns, traded = _simulate(study, bundle, live, fast.astype(bool), sample_index, sso, upro)
        if float((returns - baseline).abs().max()) <= 1e-12:
            raise RuntimeError(f"{name} matched the current book, so it is not a test")
        paths[name] = returns
        turns[name] = traded
        fast_states[name] = fast.astype(bool)

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")

    stitched_turn = _stitch(turns, walk.picks, walk.returns.index).fillna(0.0)
    stitched_fast = _stitch(
        {name: series.astype(float) for name, series in fast_states.items()},
        walk.picks,
        walk.returns.index,
    ).fillna(0.0) > 0.5
    rule_flag = {
        name: pd.Series(0.0 if name == OFF_NAME else 1.0, index=sample_index)
        for name in paths
    }
    stitched_rule = _stitch(rule_flag, walk.picks, walk.returns.index).fillna(0.0) > 0.5
    slow_off = ~live.reindex(walk.returns.index).fillna(False).astype(bool)
    parked = int((slow_off & stitched_rule & ~stitched_fast).sum())
    reentered = int((slow_off & stitched_rule & stitched_fast).sum())
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    rf = bundle["rf"]
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Of {int(slow_off.sum())} out-of-sample sessions with the 200-day band off, "
        f"the stitched rule held the core in T-bills on {parked} and had re-entered it on {reentered}."
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [
        (window, band, variant_name(window, band)) for window, band in cells()
    ]
    for window, band, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(window, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for window, band, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(window, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(window, band): (window, band) for window, band in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        window, band = by_name[last_pick]
        neighbor_paths = {
            variant_name(*neighbor): paths[variant_name(*neighbor)]
            for neighbor in adjacent_cells((window, band), [list(WINDOWS), list(BANDS)])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="fast_reentry_walk_forward",
        params={
            "bar": "materiality_v1",
            "windows": list(WINDOWS),
            "bands": list(BANDS),
            "live_cell": OFF_NAME,
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
    for window, band, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "window": None if window is None else int(window),
            "band": None if band is None else float(band),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Fast re-entry of the SPY core",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "fast_reentry",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "windows": list(WINDOWS),
            "bands": list(BANDS),
            "live_cell": OFF_NAME,
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
            "oos_off_trend_sessions": {
                "slow_off": int(slow_off.sum()),
                "core_in_tbills": parked,
                "core_reentered": reentered,
            },
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
    signal = _write_live_signal(bundle, sso, upro)
    entry["decision_note"] = entry["decision_note"] + " " + _signal_sentence(previous_signal, signal)
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": (signal.get("diagnostics") or {}).get("trend_on"),
        "dip_on": (signal.get("diagnostics") or {}).get("dip_on"),
        "reentry_rule": (signal.get("diagnostics") or {}).get("reentry_rule"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again, again_info = kept_fast_reentry(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if again is None or again_info is None:
            raise RuntimeError("a kept fast re-entry did not reload from the iteration log")
    elif again is not None or again_info is not None:
        raise RuntimeError("a rejected fast re-entry reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
        "extra": extra,
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
