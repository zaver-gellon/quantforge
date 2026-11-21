#!/usr/bin/env python3
"""One walk-forward test: an off-trend dip hold earns T-bills, not SSO.

The scopes, the cuts, and the keep rule are fixed in
``quantforge.strategies.dipcut`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The live dip, including an off-trend SSO swap, is the current book and is
not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/dip_cut.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.dipcut import (
    CUTS,
    OFF_NAME,
    SCOPES,
    cells,
    kept_dip_cut,
    offtrend_cut,
    variant_name,
)
from quantforge.strategies.trend import trend_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-20-abs-momentum"
STUDY = "spy_upro_trend"
FAMILY = "dip_cut"
CALENDAR_DAY = "2026-11-20"

HYPOTHESIS = (
    "UPRO-sleeve volatility targeting is already in the ledger. The "
    "2026-10-26 vol sleeve scaled only that sleeve by min(1, target_vol / "
    "trailing volatility) at lookbacks 10, 21, and 63 and targets 10%, 15%, "
    "20%, and 25%, with the rest of the sleeve in T-bills. The 2026-11-09 "
    "vol gear scaled the same sleeve's leverage as clip(target_vol / "
    "realized_vol, 1, 3). Neither is rerun. The 2026-11-20 close put the live "
    "book off its 200-day band with the three-down dip still on, so the next "
    "open is half SSO and half BIL. The 2026-11-03 sized dip already tested "
    "refusing that SSO swap while the band is off, which leaves the core in "
    "SPY and ends the hold. Those trend-gated cells were never selected. This "
    "test does not rescore that block, a core hold for the whole off-trend "
    "episode, or the dip's exit clock. The clock stays the live one: three "
    "strictly down closes, leave at the first up close or five sessions, and "
    "the exit wins over a fresh entry on the same close. While the band is "
    "on, the hold is still a full SSO swap. While the band is off, a fixed "
    "fraction of the core pile earns T-bills and the rest stays in SPY, but "
    "only for a hold the scope includes. Scope carry includes every "
    "off-trend session of a hold, including a hold that started while the "
    "band was on. Scope born includes only a hold whose entry close was "
    "already off-trend; a hold that started on-trend stays in SSO after the "
    "band turns off, which is the live book. Cuts are one half and the whole "
    "core. The four cells were fixed before this sample is scored. The "
    "walk-forward may also keep the current book. That cell is already "
    "counted, and its name sorts first, so a tie keeps it. The sleeve, the "
    "200-day band, and the quarterly mix are unchanged. Fills stay next-open "
    "with 5 bp one-way costs. Cash pays no spread. The stitched path is kept "
    "only when it clears the materiality bar."
)


def _load(name: str):
    import importlib.util

    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_dip_cut"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(scope: str, cut: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "scope": scope,
        "cut": float(cut),
        "max_hold": 5,
        "rule": "3_down_closes",
        "signal": "offtrend_dip_hold_earns_tbills",
        "apply": "next_open",
        "action": "replace_sso_with_tbills_while_trend_off",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "rebalance": "quarterly",
        "cost_one_way": 0.0005,
    }


def _prior_oos() -> dict | None:
    path = HERE / "iterations.json"
    if not path.exists():
        return None
    for row in reversed(json.loads(path.read_text())):
        if row.get("id") == PRIOR_ID:
            return (row.get("oos") or {}).get("current")
    return None


def _rounded_match(left: dict, right: dict) -> bool:
    return (
        round(float(left["cagr"]) * 100, 2) == round(float(right["cagr"]) * 100, 2)
        and round(float(left["sharpe"]), 3) == round(float(right["sharpe"]), 3)
        and round(float(left["max_drawdown"]) * 100, 2) == round(float(right["max_drawdown"]) * 100, 2)
    )


def _lead_refresh(refreshed: dict, baseline: pd.Series, rf: pd.Series) -> dict:
    prior = _prior_oos() or {}
    cagr_delta = None if prior.get("cagr") is None else float(refreshed["cagr"]) - float(prior["cagr"])
    sharpe_delta = None if prior.get("sharpe") is None else float(refreshed["sharpe"]) - float(prior["sharpe"])
    drawdown_delta = (
        None if prior.get("max_drawdown") is None
        else float(refreshed["max_drawdown"]) - float(prior["max_drawdown"])
    )
    same_display = bool(prior) and _rounded_match(refreshed, prior)
    overlap = None
    prior_cut_matches = None
    if prior.get("end"):
        cut = baseline.loc[: pd.Timestamp(prior["end"])]
        scored = summarize(cut, rf=rf.reindex(cut.index))
        overlap = {key: scored[key] for key in ("cagr", "sharpe", "max_drawdown", "n_days", "start", "end")}
        prior_cut_matches = _rounded_match(overlap, prior)
    if not prior:
        note = "The prior trend-core iteration row was not in iterations.json, so the refresh has no comparison row."
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
        if prior_cut_matches:
            note += " That prior-cut path matches the stored row at write-up rounding."
        else:
            note += " That prior-cut path does not match the stored row at write-up rounding."
    note += (
        " The comparison row is the prior cut stored by "
        f"`{PRIOR_ID}`. That row is the live trend-core book. "
        "The absolute-momentum ensemble tested that day was rejected."
    )
    return {
        "prior_id": PRIOR_ID,
        "prior": prior,
        "refreshed": refreshed,
        "cagr_delta": cagr_delta,
        "sharpe_delta": sharpe_delta,
        "max_drawdown_delta": drawdown_delta,
        "display_unchanged": bool(same_display) if prior else None,
        "prior_cut_matches": prior_cut_matches,
        "through_prior_end": overlap,
        "note": note,
    }


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
        ending = (
            "The stitched off-trend dip cut clears the materiality bar. "
            "It becomes the live dip vehicle while the band is off."
        )
    else:
        ending = (
            "The stitched off-trend dip cut does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _calendar_note(end: pd.Timestamp) -> str:
    nxt = next_nyse_session(end)
    return (
        f" That is the {end.day_name()} {end.date()} close. "
        f"The next NYSE session is {nxt.date()}."
    )


def _committed_signal() -> dict:
    try:
        raw = subprocess.check_output(
            ["git", "show", "HEAD:studies/spy_upro_trend/signals.json"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return json.loads((HERE / "signals.json").read_text())
    return json.loads(raw)


def _price_place(close_px: float, sma: float) -> str:
    if close_px > sma * 1.02:
        return "above the upper band"
    if close_px < sma * 0.98:
        return "below the lower band"
    return "inside the 2% band"


def _simulate_cut(study, bundle, trend, invested, sso, index, sso_px, upro):
    from quantforge.engine.backtest import run_dip_cut

    result = run_dip_cut(
        bundle["spy_open"], bundle["spy_close"],
        sso_px["open"], sso_px["close"], upro["open"], upro["close"],
        trend, invested, sso, bundle["masks"]["quarterly"],
        0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def main() -> None:
    if cells() != [("born", 0.50), ("born", 1.0), ("carry", 0.50), ("carry", 1.0)]:
        raise RuntimeError("the off-trend dip grid changed after it was fixed")
    symbols = ["SPY", "SSO", "UPRO", "^IRX", "^VIX", "^VIX3M", "GLD", "IEF", "TLT"]
    refresh_frames = load_bars(YahooSource(), symbols, start="1990-01-01", refresh=True)
    listed = {symbol: refresh_frames[symbol]["close"].dropna() for symbol in symbols}
    last_close = listed["SPY"].index[-1]
    iteration_date = str(last_close.date())
    iteration_id = f"{iteration_date}-dip-cut"
    if last_close.date().isoformat() >= CALENDAR_DAY:
        published = f" The {last_close.date()} close is published."
    else:
        published = (
            f" The {CALENDAR_DAY} close is not in the refreshed SPY series. "
            f"The latest available close is {last_close.date()}."
        )
    coverage = [
        {
            "symbol": symbol,
            "start": str(frame.index[0].date()),
            "end": str(frame.index[-1].date()),
            "n_obs": int(len(frame)),
        }
        for symbol, frame in listed.items()
    ]
    ends = ", ".join(
        f"{symbol} ends {frame.index[-1].date()}" for symbol, frame in listed.items() if symbol != "SPY"
    )
    price_note = (
        f"After a forced Yahoo refresh, SPY closes run from {listed['SPY'].index[0].date()} "
        f"to {listed['SPY'].index[-1].date()} ({len(listed['SPY'])} sessions). "
        f"{ends}."
        + _calendar_note(last_close)
        + published
    )
    hedge = _load("credit_hedge.py")
    hold = _load("dip_hold.py")
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    bundle["curve_spread"] = pd.Series(dtype=float)
    if bundle["spy_close"].index[-1] != listed["SPY"].index[-1]:
        raise RuntimeError("the prepared SPY sample does not end on the refreshed close")
    live = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")
    clock = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False).astype(bool)

    sso = synth[2]
    upro = synth[3]
    ones = pd.Series(1.0, index=bundle["cal"])
    identity, _identity_turn = _simulate_cut(study, bundle, live, ones, clock, sample_index, sso, upro)
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    if float((identity - baseline).abs().max()) > 1e-10:
        raise RuntimeError("an invested core with the live dip flag does not match the trend-core book")

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    invested_by = {}
    sso_by = {}
    for scope, cut in cells():
        name = variant_name(scope, cut)
        invested, sso_flag, cell_clock = offtrend_cut(bundle["spy_close"], live, scope, cut)
        invested = invested.reindex(bundle["cal"])
        sso_flag = sso_flag.reindex(bundle["cal"]).astype(bool)
        cell_clock = cell_clock.reindex(bundle["cal"]).fillna(False).astype(bool)
        if not cell_clock.equals(clock):
            raise RuntimeError(f"{name} changed the live dip clock")
        returns, traded = _simulate_cut(study, bundle, live, invested, sso_flag, sample_index, sso, upro)
        paths[name] = returns
        turns[name] = traded
        invested_by[name] = invested
        sso_by[name] = sso_flag
        if float((returns - baseline).abs().max()) <= 1e-12:
            raise RuntimeError(f"{name} reproduced the current book")
    born = variant_name("born", 1.0)
    carry = variant_name("carry", 1.0)
    if float((paths[born] - paths[carry]).abs().max()) <= 1e-12:
        raise RuntimeError("the born and carry scopes produced the same book")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    oos_index = sample_index[sample_index >= decisions[0]]
    pre_lead = summarize(baseline.reindex(oos_index), rf=bundle["rf"].reindex(oos_index))
    print(json.dumps({
        "price_note": price_note,
        "lead_oos_before_variants": {
            "cagr": pre_lead["cagr"],
            "sharpe": pre_lead["sharpe"],
            "max_drawdown": pre_lead["max_drawdown"],
            "n_days": pre_lead["n_days"],
            "start": pre_lead["start"],
            "end": pre_lead["end"],
        },
    }, default=float), flush=True)

    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")
    full_oos = baseline.loc[walk.returns.index[0]:]
    if len(walk.returns) != len(full_oos):
        raise RuntimeError(
            f"the stitched path has {len(walk.returns)} sessions and the lead book has {len(full_oos)} "
            "from the same start"
        )

    def _stitch(series_by_name: dict[str, pd.Series]) -> pd.Series:
        pieces = []
        items = list(walk.picks.items())
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
        return stitched.reindex(walk.returns.index)

    stitched_turn = _stitch(turns).fillna(0.0)
    rf = bundle["rf"]
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    lead_oos = summarize(current_oos, rf=rf.reindex(current_oos.index))
    if abs(float(lead_oos["cagr"]) - float(pre_lead["cagr"])) > 1e-12:
        raise RuntimeError("the walk-forward window does not match the pre-judgment lead book")

    differ = {}
    for scope, cut in cells():
        name = variant_name(scope, cut)
        invested = invested_by[name].reindex(walk.returns.index)
        sso_flag = sso_by[name].reindex(walk.returns.index).astype(bool)
        live_sso = clock.reindex(walk.returns.index).fillna(False).astype(bool)
        differ[name] = int((sso_flag.ne(live_sso) | invested.lt(1.0 - 1e-12)).sum())
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Out-of-sample sessions where the core vehicle leaves the live dip: {differ}."
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [(scope, cut, variant_name(scope, cut)) for scope, cut in cells()]
    for scope, cut, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(scope, cut),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for scope, cut, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(scope, cut),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(scope, cut): (scope, cut) for scope, cut in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        scope, cut = by_name[last_pick]
        neighbor_paths = {}
        for neighbor in adjacent_cells((scope, cut), [list(SCOPES), list(CUTS)]):
            neighbor_name = variant_name(neighbor[0], neighbor[1])
            neighbor_paths[neighbor_name] = paths[neighbor_name]
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="dip_cut_walk_forward",
        params={
            "bar": "materiality_v1",
            "scopes": list(SCOPES),
            "cuts": list(CUTS),
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
    for scope, cut, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "scope": scope,
            "cut": None if cut is None else float(cut),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "sessions_off_live_vehicle": None if name == OFF_NAME else differ[name],
            "note": "The current book. Already counted as dip_3_down_closes." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Off-trend dip hold earns T-bills instead of SSO",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "dip_cut",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "scopes": list(SCOPES),
            "cuts": list(CUTS),
            "live_cell": OFF_NAME,
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "sessions_off_live_vehicle": differ,
            "lead_oos_before_judgment": {
                "cagr": lead_oos["cagr"],
                "sharpe": lead_oos["sharpe"],
                "max_drawdown": lead_oos["max_drawdown"],
                "n_days": lead_oos["n_days"],
                "start": lead_oos["start"],
                "end": lead_oos["end"],
            },
            "oos_turnover_per_year": {
                "current": base_oos_summary["turnover_per_year"],
                "candidate": cand_oos_summary["turnover_per_year"],
            },
        },
        "variants": variants,
        "walk_forward_picks": [[str(date), name] for date, name in walk.picks.items()],
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
    irx_pct, _unit = study._yield_series(refresh_frames["^IRX"]["close"])
    from quantforge.strategies.billgear import align_bill_yield

    bill_yield = align_bill_yield(irx_pct, bundle["cal"])
    signal = hold._write_live_signal(
        bundle, sso, upro, bill_yield, listed["^VIX"], listed["^VIX3M"],
        {asset: listed[asset] for asset in ("GLD", "IEF", "TLT")},
    )
    diagnostics = signal.get("diagnostics") or {}
    close_px = float(diagnostics["reference_close"])
    sma = float(diagnostics["sma"])
    level_sentence = (
        f" SPY close {close_px:.4f} versus the 200-day average {sma:.4f} "
        f"({close_px / sma - 1.0:+.2%} versus the average), {_price_place(close_px, sma)}."
    )
    entry["decision_note"] = (
        entry["decision_note"] + level_sentence + " " + hold._signal_sentence(previous_signal, signal)
    )
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": diagnostics.get("trend_on"),
        "dip_on": diagnostics.get("dip_on"),
        "dip_cut_rule": diagnostics.get("dip_cut_rule"),
        "dip_cut_on": diagnostics.get("dip_cut_on"),
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "rebalance_mix": signal.get("rebalance_mix"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again_hold, again_boost, again_info = kept_dip_cut(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_hold is not None or again_info is not None:
                raise RuntimeError("a kept cut whose current pick is the live book reloaded an override")
        elif again_info is None or again_hold is None or again_boost is None:
            raise RuntimeError("a kept off-trend dip cut did not reload")
        elif again_info["dip_cut_rule"] != last_pick:
            raise RuntimeError("a kept off-trend dip cut reloaded a different cell")
        elif bool(diagnostics.get("dip_cut_on")) != bool(again_info["dip_cut_on"]):
            raise RuntimeError("the published dip-cut flag does not match the reloaded rule")
    elif again_hold is not None or again_boost is not None or again_info is not None:
        raise RuntimeError("a rejected off-trend dip cut reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "iteration_id": iteration_id,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "lead_oos_before_judgment": entry["data"]["lead_oos_before_judgment"],
        "prior_cut_matches": refresh.get("prior_cut_matches"),
        "through_prior_end": refresh.get("through_prior_end"),
        "oos": result["oos"],
        "checks": result["checks"],
        "neighbors": result["criterion_c"]["neighbors"],
        "picks": entry["walk_forward_picks"],
        "variants": [
            {
                "name": row["name"],
                "cagr": row["cagr"],
                "sharpe": row["sharpe"],
                "max_drawdown": row["max_drawdown"],
                "years_selected": row["years_selected"],
            }
            for row in variants
        ],
        "signal": entry["signal"],
        "extra": extra,
        "price_note": price_note,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
