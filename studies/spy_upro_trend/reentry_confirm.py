#!/usr/bin/env python3
"""One walk-forward test: delay the UPRO sleeve's re-entry, not its exit.

The counts and the keep rule are fixed in ``quantforge.strategies.reconfirm``
and ``quantforge.eval.materiality``. Do not add a cell after seeing this sample.

The immediate re-entry is the current book and is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/reentry_confirm.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.reconfirm import (
    AXIS,
    BAND,
    CONFIRMS,
    LIVE_CONFIRM,
    OFF_NAME,
    WINDOW,
    cells,
    kept_reentry_confirm,
    reentry_trend_state,
    variant_name,
)
from quantforge.strategies.trend import trend_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-16-sym-drift"
STUDY = "spy_upro_trend"
FAMILY = "reentry_confirm"
CALENDAR_DAY = "2026-11-17"

HYPOTHESIS = (
    "The live sleeve turns on at the first close above 1.02 times the 200-day "
    "average and turns off at the first close below 0.98 times that average. "
    "The original grid already scored that 2% band. The 2026-10-31 test then "
    "required N consecutive closes beyond the band before either transition, "
    "so the exit waited as well as the re-entry and the sleeve stayed at 3x "
    "through the first closes of a break. The walk-forward preferred a wider "
    "band with an immediate flip and missed the materiality bar. Static mix "
    "ratios of 30/70 through 70/30 are already trials in that original grid, "
    "including the live 200-day 2% filter with a quarterly reset, so this test "
    "does not rescore them. The sized dip, the dip-exit grid, and the "
    "shorter-filter re-entry of the SPY core are already scored as well. This "
    "test does not change the exit. The sleeve still turns off on the first "
    "close below 0.98 times the average. After it is off, it turns back on "
    "only after N consecutive closes above 1.02 times the average. A close "
    "that is not above that upper band resets the count, and the sleeve stays "
    "in T-bills until the count is met. N of 1 is the live filter, already "
    "counted, and is the walk-forward's option to keep the current book. Its "
    "name sorts first, so a tie in the trailing excess Sharpe keeps it. The "
    "new trials are N of 2, 3, 5, 8, and 10, fixed before this sample is "
    "scored. The SPY core, the three-down-close dip, and the quarterly mix "
    "are unchanged. Fills stay next-open with 5 bp one-way costs. Cash pays "
    "no spread. The study counts, on the live filter and before any new cell "
    "is scored, how often the sleeve re-enters and how many of those "
    "re-entries would still complete under each longer count. The stitched "
    "path is kept only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_reentry"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(confirms: int) -> dict:
    return {
        "base": "dip_3_down_closes",
        "confirms": int(confirms),
        "exit": "immediate_below_lower_band",
        "entry": "consecutive_closes_above_upper_band",
        "window": WINDOW,
        "band": BAND,
        "apply": "next_open",
        "action": "delay_sleeve_reentry",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
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
    note += (
        " The comparison row is the last trend-core lead, "
        f"`{PRIOR_ID}`, not the later faster-book rows."
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
        ending = "The stitched re-entry delay clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched re-entry delay does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


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


def _changes(series: pd.Series) -> int:
    flags = series.astype(bool)
    if len(flags) < 2:
        return 0
    return int(flags.ne(flags.shift(1)).iloc[1:].sum())


def _measure_reentries(close: pd.Series, trend: pd.Series, oos_index: pd.Index) -> dict:
    """How the live sleeve re-enters. No new cell is scored."""
    series = close.astype(float).sort_index()
    average = series.rolling(WINDOW, min_periods=WINDOW).mean()
    upper = (series > average * (1.0 + BAND)).fillna(False)
    state = trend.reindex(series.index).fillna(False).astype(bool)
    oos = pd.DatetimeIndex(oos_index)
    turned_on = state & state.shift(1).eq(False)
    entry_dates = turned_on.loc[turned_on].index.intersection(oos)
    streaks = []
    for ts in entry_dates:
        loc = series.index.get_loc(ts)
        if not isinstance(loc, (int, np.integer)):
            raise RuntimeError("duplicate SPY session in the re-entry count")
        length = 0
        j = int(loc)
        while j < len(series) and bool(upper.iloc[j]):
            length += 1
            j += 1
        streaks.append(length)
    shares = {}
    for count in CONFIRMS:
        if streaks:
            shares[variant_name(count)] = float(sum(item >= count for item in streaks) / len(streaks))
        else:
            shares[variant_name(count)] = float("nan")
    completed = []
    open_length = 0
    open_in_window = False
    on = False
    length = 0
    started = None
    for i, flag in enumerate(state.to_numpy(dtype=bool)):
        if flag and not on:
            on = True
            length = 1
            started = series.index[i]
        elif flag and on:
            length += 1
        elif on and started is not None:
            if started in oos:
                completed.append(length)
            on = False
            length = 0
            started = None
    if on and started is not None and started in oos:
        open_length = length
        open_in_window = True
    median = float(pd.Series(streaks, dtype=float).median()) if streaks else float("nan")
    return {
        "oos_sessions": int(len(oos)),
        "oos_reentries": int(len(streaks)),
        "median_upper_streak": median,
        "share_reaching": shares,
        "completed_on_spells": int(len(completed)),
        "on_spells_ending_within_5": int(sum(item <= 5 for item in completed)),
        "on_spells_ending_within_10": int(sum(item <= 10 for item in completed)),
        "on_spells_ending_within_21": int(sum(item <= 21 for item in completed)),
        "on_spells_still_open": int(open_in_window),
        "open_spell_sessions": int(open_length),
    }


def _measured_paragraph(stats: dict) -> str:
    shares = ", ".join(
        f"{name} {stats['share_reaching'][name]:.1%}" for name in sorted(stats["share_reaching"])
    )
    return (
        "Measured on the live 2% filter before any new cell was scored. "
        f"Of {stats['oos_sessions']} out-of-sample sessions, the sleeve re-entered "
        f"{stats['oos_reentries']} times. The median run of closes above the upper band, "
        f"starting at that re-entry, was {stats['median_upper_streak']:.1f} sessions. "
        f"The share of those re-entries whose run still reaches each count was {shares}. "
        f"Completed on-spells that started in the window: {stats['completed_on_spells']}, "
        f"of which {stats['on_spells_ending_within_5']} ended within 5 sessions, "
        f"{stats['on_spells_ending_within_10']} within 10, and "
        f"{stats['on_spells_ending_within_21']} within 21. "
        f"{stats['on_spells_still_open']} spell was still open at the sample end "
        f"({stats['open_spell_sessions']} sessions)."
    )


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


def main() -> None:
    symbols = ["SPY", "SSO", "UPRO", "^IRX", "^VIX", "^VIX3M", "GLD", "IEF", "TLT"]
    refresh_frames = load_bars(YahooSource(), symbols, start="1990-01-01", refresh=True)
    listed = {symbol: refresh_frames[symbol]["close"].dropna() for symbol in symbols}
    last_close = listed["SPY"].index[-1]
    iteration_date = str(last_close.date())
    iteration_id = f"{iteration_date}-reentry-confirm"
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
    live = trend_state(bundle["spy_close"], WINDOW, BAND, "daily")
    cached = bundle["trends"][(WINDOW, BAND, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")
    immediate = reentry_trend_state(bundle["spy_close"], WINDOW, BAND, LIVE_CONFIRM)
    if not immediate.equals(live):
        raise RuntimeError("a one-close re-entry does not match the live filter")
    if bundle["spy_close"].index[-1] != listed["SPY"].index[-1]:
        raise RuntimeError("the prepared SPY sample does not end on the refreshed close")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    oos_index = sample_index[sample_index >= decisions[0]]
    measured = _measure_reentries(bundle["spy_close"], live, oos_index)
    measured_text = _measured_paragraph(measured)
    pre_lead = summarize(baseline.reindex(oos_index), rf=bundle["rf"].reindex(oos_index))
    print(json.dumps({
        "lead_oos_before_variants": {
            "cagr": pre_lead["cagr"],
            "sharpe": pre_lead["sharpe"],
            "max_drawdown": pre_lead["max_drawdown"],
            "n_days": pre_lead["n_days"],
            "start": pre_lead["start"],
            "end": pre_lead["end"],
        },
        "measured_before_variants_scored": measured,
        "coverage": coverage,
    }, default=float), flush=True)

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    states = {OFF_NAME: live.astype(float)}
    widest = 0.0
    for count in cells():
        name = variant_name(count)
        state = reentry_trend_state(bundle["spy_close"], WINDOW, BAND, count)
        returns, traded = hedge._simulate(study, bundle, state, sample_index, sso, upro)
        widest = max(widest, float((returns - baseline).abs().max()))
        paths[name] = returns
        turns[name] = traded
        states[name] = state.astype(float)
    if widest <= 1e-12:
        raise RuntimeError("every re-entry delay reproduced the current book")

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

    stitched_turn = _stitch(turns, walk.picks, walk.returns.index).fillna(0.0)
    stitched_state = _stitch(states, walk.picks, walk.returns.index).fillna(0.0)
    rf = bundle["rf"]
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    lead_oos = summarize(current_oos, rf=rf.reindex(current_oos.index))
    lead_before = {
        "cagr": lead_oos["cagr"],
        "sharpe": lead_oos["sharpe"],
        "max_drawdown": lead_oos["max_drawdown"],
        "n_days": lead_oos["n_days"],
        "start": lead_oos["start"],
        "end": lead_oos["end"],
    }
    if abs(float(lead_before["cagr"]) - float(pre_lead["cagr"])) > 1e-12:
        raise RuntimeError("the walk-forward window does not match the pre-judgment lead book")

    differ = {
        variant_name(count): int(
            states[variant_name(count)].reindex(walk.returns.index).fillna(0.0).ne(
                states[OFF_NAME].reindex(walk.returns.index).fillna(0.0)
            ).sum()
        )
        for count in cells()
    }
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Trend-state changes on that window {_changes(states[OFF_NAME].reindex(walk.returns.index))} -> "
        f"{_changes(stitched_state)}. "
        f"Sessions where a frozen count differs from the live filter: {differ}. "
        f"{measured_text}"
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(LIVE_CONFIRM, OFF_NAME)] + [(count, variant_name(count)) for count in cells()]
    for count, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(count),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for count, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(count),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(count): count for count in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        count = by_name[last_pick]
        neighbor_paths = {}
        for neighbor in adjacent_cells((count,), [list(AXIS)]):
            neighbor_count = neighbor[0]
            if neighbor_count == LIVE_CONFIRM:
                neighbor_paths[OFF_NAME] = paths[OFF_NAME]
            else:
                neighbor_paths[variant_name(neighbor_count)] = paths[variant_name(neighbor_count)]
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="reentry_confirm_walk_forward",
        params={
            "bar": "materiality_v1",
            "confirms": list(CONFIRMS),
            "live_cell": OFF_NAME,
            "live_cell_counts": False,
            "exit": "immediate_below_lower_band",
            "entry": "consecutive_closes_above_upper_band",
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
    for count, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "confirms": int(count),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. Immediate re-entry. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Re-entry delay on the UPRO sleeve",
        "hypothesis": HYPOTHESIS + "\n\n" + measured_text,
        "status": "tested",
        "kept": kept,
        "overlay": "reentry_confirm",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "confirms": list(CONFIRMS),
            "live_cell": OFF_NAME,
            "exit": "immediate_below_lower_band",
            "entry": "consecutive_closes_above_upper_band",
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "live_reentries": measured,
            "sessions_differing_from_live": differ,
            "lead_oos_before_judgment": lead_before,
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
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "rebalance_mix": signal.get("rebalance_mix"),
        "trade_reason": signal.get("trade_reason"),
        "reentry_rule": diagnostics.get("reentry_rule"),
        "reentry_confirms": diagnostics.get("reentry_confirms"),
        "reentry_on": diagnostics.get("reentry_on"),
    }
    hedge._write_outputs(entry)
    again_state, again_info = kept_reentry_confirm(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_state is not None or again_info is not None:
                raise RuntimeError("a kept delay whose current pick is the live filter reloaded an override")
        elif again_state is None or again_info is None:
            raise RuntimeError("a kept re-entry delay did not reload")
        elif int(again_info["reentry_confirms"]) != int(by_name[last_pick]):
            raise RuntimeError("a kept re-entry delay reloaded a different count")
    elif again_state is not None or again_info is not None:
        raise RuntimeError("a rejected re-entry delay reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "iteration_id": iteration_id,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "lead_oos_before_judgment": lead_before,
        "measured": measured,
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
        "coverage": coverage,
        "through_prior_end": refresh.get("through_prior_end"),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
