#!/usr/bin/env python3
"""One walk-forward test: reset the mix when UPRO leaves a band on either side.

The bands and the keep rule are fixed in ``quantforge.strategies.symdrift``
and ``quantforge.eval.materiality``. Do not add a cell after seeing this sample.

The quarterly calendar is the current book and is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/sym_drift.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_symmetric_drift, run_trend_core
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.symdrift import BANDS, OFF_NAME, cells, kept_sym_drift, variant_name
from quantforge.strategies.trend import trend_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-13-dip-hold"
STUDY = "spy_upro_trend"
FAMILY = "sym_drift"
CALENDAR_DAY = "2026-11-16"

HYPOTHESIS = (
    "The live book resets both piles to 50/50 on a quarterly calendar. Between "
    "those dates the sleeve is 3x while the trend is on, so a rally lifts its "
    "weight above one half and a drop pushes it below. The calendar also trades "
    "when the gap is small. The original grid compared monthly and quarterly "
    "calendars. The 2026-11-13 exposure run then scored daily, monthly, and "
    "quarterly on this same 50/50 book; the walk-forward pick was the daily "
    "reset and it missed the materiality bar. The 2026-10-31 test required N "
    "closes through the 200-day band before the sleeve flipped, and the 2% "
    "band itself is already the live filter, so this test does not rescore "
    "hysteresis. The 2026-11-04 test replaced the calendar with a one-sided "
    "ceiling: it sold UPRO only after the weight was above half, and it left "
    "a shrunken sleeve alone. The walk-forward kept the calendar in every "
    "year. That rule dropped the side the calendar still does, which is buying "
    "the sleeve back after UPRO has fallen and the book is under its 2x budget "
    "while the trend is on. This test does not add a regime signal and does "
    "not sell the core. It replaces the quarterly mix date with a two-sided "
    "band. After the close, if the sleeve is in UPRO and its weight is "
    "strictly more than a fixed band away from one half, in either direction, "
    "the next open resets both piles to 50/50. If the gap is inside the band, "
    "or the sleeve is in T-bills, the piles are left alone. A trend flip still "
    "swaps the sleeve instrument and does not by itself resize the piles. The "
    "three-down-close dip is unchanged. Bands are 2, 4, 6, 8, and 10 percentage "
    "points, fixed before this sample is scored. The quarterly calendar is the "
    "current book, already counted, and is the walk-forward's option to keep "
    "it. Its name sorts first, so a tie in the trailing excess Sharpe keeps "
    "it. The five bands are new trials. Fills stay next-open with 5 bp "
    "one-way costs. Cash pays no spread. The study counts, on the live book "
    "and before any new cell is scored, how far the UPRO weight sits from one "
    "half. The stitched path is kept only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_sym_drift"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(band: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "band": float(band),
        "trigger": "abs_upro_weight_minus_sleeve_weight",
        "buys_weakness": True,
        "sells_strength": True,
        "calendar": None,
        "apply": "next_open",
        "action": "reset_both_piles",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
        "rebalance": "symmetric_drift",
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
        ending = "The stitched two-sided drift band clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched two-sided drift band does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _simulate_band(study, bundle, trend, band, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result, mix = run_symmetric_drift(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, float(band), 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    flagged = mix.reindex(index).fillna(False)
    return returns, turnover, flagged


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


def _measure_live(bundle, trend, sso, upro, oos_index) -> dict:
    """How far the live book's UPRO weight sits from one half. No new cell is scored."""
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    detailed = run_trend_core(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
        record_sleeve=True,
    )
    weight = detailed.sleeve_weight.reindex(oos_index)
    held = weight.dropna()
    gap = (held - 0.5).abs()
    quarter = bundle["masks"]["quarterly"].reindex(oos_index).fillna(False)
    on_quarter = weight.where(quarter).dropna()
    quarter_gap = (on_quarter - 0.5).abs()
    beyond = {variant_name(band): float((gap > band).mean()) if len(gap) else float("nan") for band in BANDS}
    return {
        "oos_sessions": int(len(oos_index)),
        "oos_upro_sessions": int(len(held)),
        "mean_abs_gap": float(gap.mean()) if len(gap) else float("nan"),
        "share_beyond": beyond,
        "quarter_resets_in_upro": int(len(on_quarter)),
        "quarter_resets_inside_2pp": int((quarter_gap <= 0.02).sum()) if len(quarter_gap) else 0,
        "quarter_resets_outside_2pp": int((quarter_gap > 0.02).sum()) if len(quarter_gap) else 0,
    }


def _measured_paragraph(stats: dict) -> str:
    shares = ", ".join(
        f"{name} {stats['share_beyond'][name]:.1%}" for name in sorted(stats["share_beyond"])
    )
    return (
        "Measured on the live quarterly book before any new cell was scored. "
        f"Of {stats['oos_sessions']} out-of-sample sessions, the sleeve was in UPRO on "
        f"{stats['oos_upro_sessions']}. The mean absolute gap from one half on those "
        f"sessions was {stats['mean_abs_gap']:.4f}. The share beyond each band was {shares}. "
        f"Quarter-end decisions with the sleeve in UPRO: {stats['quarter_resets_in_upro']}, "
        f"of which {stats['quarter_resets_inside_2pp']} had a gap of 2 percentage points or "
        f"less and {stats['quarter_resets_outside_2pp']} were wider."
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
    iteration_id = f"{iteration_date}-sym-drift"
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
    ends = ", ".join(f"{symbol} ends {frame.index[-1].date()}" for symbol, frame in listed.items() if symbol != "SPY")
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
    live = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")
    if bundle["spy_close"].index[-1] != listed["SPY"].index[-1]:
        raise RuntimeError("the prepared SPY sample does not end on the refreshed close")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    oos_index = sample_index[sample_index >= decisions[0]]
    measured = _measure_live(bundle, live, sso, upro, oos_index)
    measured_text = _measured_paragraph(measured)

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    flags = {OFF_NAME: pd.Series(False, index=sample_index)}
    widest = 0.0
    for band in cells():
        name = variant_name(band)
        returns, traded, flagged = _simulate_band(study, bundle, live, band, sample_index, sso, upro)
        widest = max(widest, float((returns - baseline).abs().max()))
        paths[name] = returns
        turns[name] = traded
        flags[name] = flagged.astype(float)
    if widest <= 1e-12:
        raise RuntimeError("every two-sided band reproduced the current book")

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
    stitched_flag = _stitch(flags, walk.picks, walk.returns.index).fillna(0.0) > 0.5
    quarter = bundle["masks"]["quarterly"].reindex(walk.returns.index).fillna(False)
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
    print(json.dumps({
        "lead_oos_before_judgment": lead_before,
        "measured_before_variants_scored": measured,
        "coverage": coverage,
    }, default=float), flush=True)

    band_calls = {
        variant_name(band): int(flags[variant_name(band)].reindex(walk.returns.index).fillna(0).sum())
        for band in cells()
    }
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"The quarterly calendar calls a mix on {int(quarter.sum())} closes in the out-of-sample window. "
        f"The stitched rule calls a mix on {int(stitched_flag.sum())} closes. "
        f"Band calls on that window: {band_calls}. "
        f"{measured_text}"
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, OFF_NAME)] + [(band, variant_name(band)) for band in cells()]
    for band, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for band, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(band): band for band in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        band = by_name[last_pick]
        neighbor_paths = {
            variant_name(neighbor[0]): paths[variant_name(neighbor[0])]
            for neighbor in adjacent_cells((band,), [list(BANDS)])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="sym_drift_walk_forward",
        params={
            "bar": "materiality_v1",
            "bands": list(BANDS),
            "live_cell": OFF_NAME,
            "live_cell_counts": False,
            "trigger": "abs_upro_weight_minus_sleeve_weight",
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
    for band, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "band": None if band is None else float(band),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. Quarterly mix. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Two-sided UPRO drift band",
        "hypothesis": HYPOTHESIS + "\n\n" + measured_text,
        "status": "tested",
        "kept": kept,
        "overlay": "sym_drift",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "bands": list(BANDS),
            "live_cell": OFF_NAME,
            "trigger": "abs_upro_weight_minus_sleeve_weight",
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "live_gap": measured,
            "lead_oos_before_judgment": lead_before,
            "oos_mix": {
                "sessions": int(len(walk.returns)),
                "quarter_calls": int(quarter.sum()),
                "stitched_calls": int(stitched_flag.sum()),
                "band_calls": band_calls,
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
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "rebalance_mix": signal.get("rebalance_mix"),
        "trade_reason": signal.get("trade_reason"),
        "sym_rule": diagnostics.get("sym_rule"),
        "sym_band": diagnostics.get("sym_band"),
        "sym_calendar": diagnostics.get("sym_calendar"),
        "sym_mix": diagnostics.get("sym_mix"),
    }
    hedge._write_outputs(entry)
    again_flag, again_info = kept_sym_drift(
        HERE / "iterations.json",
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        bundle["rf"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_info is None or not again_info.get("sym_calendar", False):
                raise RuntimeError("a kept two-sided band whose current pick is the calendar did not reload it")
        elif again_flag is None or again_info is None or again_info.get("sym_calendar", True):
            raise RuntimeError("a kept two-sided band did not reload from the iteration log")
        elif abs(float(again_info["sym_band"]) - float(by_name[last_pick])) > 1e-12:
            raise RuntimeError("a kept two-sided band reloaded a different width")
    elif again_flag is not None or again_info is not None:
        raise RuntimeError("a rejected two-sided band reloaded as if it were kept")
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
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
