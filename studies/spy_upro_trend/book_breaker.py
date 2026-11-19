#!/usr/bin/env python3
"""One walk-forward test: a book-level drawdown breaker on the UPRO sleeve.

The cuts, the re-entry modes, and the keep rule are fixed in
``quantforge.strategies.bookbreaker`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The unbraked sleeve is the current book and is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/book_breaker.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_book_breaker, run_trend_core
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.strategies.bookbreaker import (
    CUTS,
    MODES,
    OFF_NAME,
    ONE_WAY_COST,
    REENTRY_FRACTION,
    cells,
    kept_book_breaker,
    variant_name,
)
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.trend import trend_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-18-credit-regime"
STUDY = "spy_upro_trend"
FAMILY = "book_breaker"
CALENDAR_DAY = "2026-11-19"
PROBE_CUT = 0.99

HYPOTHESIS = (
    "The live book's out-of-sample max drawdown is still about -46.8%. "
    "The 2026-10-29 trailing-peak stop cut the UPRO sleeve when SPY's price "
    "was 8%, 12%, or 16% below its own high over 63, 126, or 252 sessions, "
    "and it put the sleeve back when that price drawdown had healed to half "
    "the cut. That is a price rule. It is already in the ledger as "
    "drawdown_stop, and this test does not rescore those windows or those "
    "cuts. A book-level breaker watches this book's equity, not SPY's price. "
    "The equity is the trend-core account itself: half SPY (SSO while the "
    "three-down-close dip is on) and half UPRO or T-bills, with the quarterly "
    "mix and 5 bp one-way costs. The running peak starts at the inception "
    "cash value of 1 and is updated at each close with that close's equity. "
    "The drawdown is equity divided by that peak, minus one. It uses the "
    "breaker's own path, so a sleeve that has already moved to T-bills slows "
    "the later decline. When the drawdown is at or below a cut, the UPRO "
    "sleeve earns T-bills on the next open even if the 200-day trend is still "
    "on. The same-day close is not a same-day fill. The SPY core, the dip, "
    "and the quarterly mix are unchanged. The cuts are 15%, 20%, and 25%, "
    "fixed before this sample is scored. They sit inside the recorded "
    "drawdown and above a one-week wobble of a book that is about 2x while "
    "the trend is on. Re-entry is the second axis, also fixed beforehand. "
    "recover clears the breaker only when the drawdown has healed to half "
    "the cut and the 200-day trend is on. The half-cut heal is not searched. "
    "confirm does not clear on the same on-spell: after the trip, the trend "
    "must be off at a close and then on at a later close, and the sleeve "
    "comes back on the following open. A new high alone does not clear it. "
    "The walk-forward may also keep the current book. That cell is already "
    "counted, and its name sorts first, so a tie in the trailing excess "
    "Sharpe keeps it. The six cells are the new trials. Fills stay next-open "
    "with 5 bp one-way costs. Cash pays no spread. The stitched path is kept "
    "only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_book_breaker"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(cut: float, mode: str) -> dict:
    return {
        "base": "dip_3_down_closes",
        "peak": "book_equity_since_inception",
        "cut": float(cut),
        "reentry": str(mode),
        "reentry_fraction": REENTRY_FRACTION if mode == "recover" else None,
        "action": "sleeve_to_tbill",
        "apply": "next_open",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
        "rebalance": "quarterly",
        "cost_one_way": ONE_WAY_COST,
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
        "The credit-regime gate tested that day was rejected."
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
        ending = "The stitched book-level drawdown breaker clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched book-level drawdown breaker does not clear the materiality bar. "
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


def _run_cell(study, bundle, sso, upro, index, cut: float, mode: str):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    trend = bundle["trends"][(200, 0.02, "daily")]
    result, breaker, sleeve, drawdown = run_book_breaker(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"],
        float(cut), mode, 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
        reentry_fraction=REENTRY_FRACTION,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    flags = {
        "breaker": breaker.reindex(index).astype(bool),
        "sleeve": sleeve.reindex(index).astype(bool),
        "drawdown": drawdown.reindex(index).astype(float),
    }
    if flags["breaker"].isna().any() or flags["sleeve"].isna().any() or flags["drawdown"].isna().any():
        raise RuntimeError("the book breaker is missing sample sessions")
    return returns, turnover, flags


def _baseline(study, bundle, trend, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_trend_core(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
        record_sleeve=True,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    sleeve = result.sleeve_weight.reindex(index).notna()
    return returns, turnover, sleeve


def _live_drawdown(drawdown: pd.Series, sleeve: pd.Series) -> dict:
    """How deep the unbraked account already was. Not a trial."""
    dd = drawdown.astype(float)
    held = sleeve.astype(bool)
    trough_at = dd.idxmin()
    cuts = {}
    for cut in CUTS:
        deep = dd <= -float(cut)
        first = None
        first_sleeve = None
        if bool(deep.any()):
            first_at = dd.index[deep.to_numpy()][0]
            first = str(first_at.date())
            first_sleeve = bool(held.loc[first_at])
        cuts[variant_name(cut, "recover").split("_")[1]] = {
            "cut": float(cut),
            "sessions": int(deep.sum()),
            "sessions_sleeve_in_upro": int((deep & held).sum()),
            "first": first,
            "first_sleeve_in_upro": first_sleeve,
        }
    return {
        "sessions": int(len(dd)),
        "trough": float(dd.loc[trough_at]),
        "trough_date": str(trough_at.date()),
        "trough_sleeve_in_upro": bool(held.loc[trough_at]),
        "cuts": cuts,
    }


def _measured_text(stats: dict) -> str:
    parts = []
    for key, row in stats["cuts"].items():
        points = int(round(float(row["cut"]) * 100))
        if row["first"] is None:
            first = "never"
        else:
            first = f"{row['first']} (sleeve in UPRO {row['first_sleeve_in_upro']})"
        parts.append(
            f"{points}% on {row['sessions']} sessions "
            f"({row['sessions_sleeve_in_upro']} still in UPRO), first {first}"
        )
    return (
        "Measured on the live book's own equity before any new cell was scored. "
        "The running peak starts at inception cash of 1, not at the 2006 window, "
        "and the sleeve is whatever the unbraked book held that close. "
        f"Of {stats['sessions']} out-of-sample sessions, the deepest mark is "
        f"{stats['trough']:.4%} on {stats['trough_date']} "
        f"(sleeve in UPRO that close: {stats['trough_sleeve_in_upro']}). "
        "Sessions at or through each cut: " + "; ".join(parts) + "."
    )


def _publish_lead(signal: dict, kept: bool, result: dict) -> dict:
    """Write the annotated lead signal. Does not read or write positions or risk."""
    from quantforge.publish import annotate_lead, load_spec
    from quantforge.signals.emit import write_signal
    from quantforge.signals.schema import ALLOWED_TREND_TICKERS

    spec = load_spec(ROOT / "studies" / "faster" / "live.json")
    annotated = annotate_lead(signal, spec)
    if kept:
        candidate = result["oos"]["candidate"]
        annotated["oos"] = {
            "cagr": candidate["cagr"],
            "sharpe": candidate["sharpe"],
            "max_drawdown": candidate["max_drawdown"],
            "start": candidate["start"],
            "end": candidate["end"],
            "n_days": candidate["n_days"],
        }
        annotated["validated_reason"] = "book_breaker_cleared_materiality_bar"
    write_signal(annotated, HERE / "signals.json", allowed_tickers=ALLOWED_TREND_TICKERS)
    write_signal(annotated, ROOT / "signals" / "trend-core.json", allowed_tickers=ALLOWED_TREND_TICKERS)
    bundle_path = ROOT / "signals" / "signals.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text())
        strategies = bundle.get("strategies") or {}
        strategies["trend-core"] = annotated
        bundle["strategies"] = strategies
        bundle["date"] = annotated.get("date")
        bundle["as_of"] = annotated.get("as_of")
        bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    return annotated


def main() -> None:
    symbols = ["SPY", "SSO", "UPRO", "^IRX", "^VIX", "^VIX3M", "GLD", "IEF", "TLT"]
    refresh_frames = load_bars(YahooSource(), symbols, start="1990-01-01", refresh=True)
    listed = {symbol: refresh_frames[symbol]["close"].dropna() for symbol in symbols}
    last_close = listed["SPY"].index[-1]
    iteration_date = str(last_close.date())
    iteration_id = f"{iteration_date}-book-breaker"
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
    if float(study.BASE_COST) != float(ONE_WAY_COST):
        raise RuntimeError("the book breaker cost does not match the live book's one-way cost")
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    bundle["curve_spread"] = pd.Series(dtype=float)
    if bundle["spy_close"].index[-1] != listed["SPY"].index[-1]:
        raise RuntimeError("the prepared SPY sample does not end on the refreshed close")
    live = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    cached = bundle["trends"][(200, 0.02, "daily")]
    if not live.equals(cached):
        raise RuntimeError("the prepared 200-day 2% state does not match trend_state")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn, base_sleeve = _baseline(study, bundle, live, sample_index, sso, upro)
    probe_r, probe_turn, probe = _run_cell(study, bundle, sso, upro, sample_index, PROBE_CUT, "recover")
    if bool(probe["breaker"].any()):
        raise RuntimeError("the non-binding probe tripped, so it is not the live book")
    gap = float((probe_r - baseline).abs().max())
    if gap > 1e-10 or float((probe_turn - base_turn).abs().max()) > 1e-10:
        raise RuntimeError(f"a non-binding breaker diverged from the live book by {gap}")
    if not probe["sleeve"].equals(base_sleeve):
        raise RuntimeError("the unbraked sleeve state does not match the live book")
    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    oos_index = sample_index[sample_index >= decisions[0]]
    pre_lead = summarize(baseline.reindex(oos_index), rf=bundle["rf"].reindex(oos_index))
    live_stats = _live_drawdown(probe["drawdown"].reindex(oos_index), probe["sleeve"].reindex(oos_index))
    measured = _measured_text(live_stats)
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
        "live_book_drawdown_before_scoring": live_stats,
    }, default=float), flush=True)

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    breakers = {OFF_NAME: pd.Series(False, index=sample_index)}
    sleeves = {OFF_NAME: base_sleeve.astype(float)}
    widest = 0.0
    for cut, mode in cells():
        name = variant_name(cut, mode)
        returns, traded, flags = _run_cell(study, bundle, sso, upro, sample_index, cut, mode)
        widest = max(widest, float((returns - baseline).abs().max()))
        paths[name] = returns
        turns[name] = traded
        breakers[name] = flags["breaker"]
        sleeves[name] = flags["sleeve"].astype(float)
    if widest <= 1e-12:
        raise RuntimeError("every book-breaker cell reproduced the current book")
    for cut, mode in cells():
        if cut != CUTS[0]:
            continue
        name = variant_name(cut, mode)
        if float((paths[name] - baseline).abs().max()) <= 1e-12:
            raise RuntimeError(f"{name} never left the live book")

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
    stitched_breaker = _stitch(
        {name: flags.astype(float) for name, flags in breakers.items()},
        walk.picks, walk.returns.index,
    ).fillna(0.0)
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

    trend_oos = live.reindex(walk.returns.index).fillna(False).astype(bool)
    forced = {}
    differ = {}
    trips = {}
    for cut, mode in cells():
        name = variant_name(cut, mode)
        on = breakers[name].reindex(walk.returns.index).fillna(False).astype(bool)
        forced[name] = int((on & trend_oos).sum())
        differ[name] = int(sleeves[name].reindex(walk.returns.index).fillna(0.0).ne(
            sleeves[OFF_NAME].reindex(walk.returns.index).fillna(0.0)
        ).sum())
        trips[name] = _changes(on)
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Breaker trips on the stitched path: {_changes(stitched_breaker)}. "
        f"Out-of-sample sessions where a frozen breaker is on while the trend is on: {forced}. "
        f"Sessions where a frozen sleeve differs from the live book: {differ}. "
        f"Trips of each frozen breaker on that window: {trips}. "
        f"{measured}"
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [(cut, mode, variant_name(cut, mode)) for cut, mode in cells()]
    for cut, mode, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(cut, mode),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for cut, mode, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(cut, mode),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(cut, mode): (cut, mode) for cut, mode in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        cut, mode = by_name[last_pick]
        neighbor_paths = {}
        for neighbor in adjacent_cells((cut, mode), [list(CUTS), list(MODES)]):
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
        study=STUDY, family="audit", name="book_breaker_walk_forward",
        params={
            "bar": "materiality_v1",
            "cuts": list(CUTS),
            "modes": list(MODES),
            "reentry_fraction": REENTRY_FRACTION,
            "live_cell": OFF_NAME,
            "live_cell_counts": False,
            "probe_cut_not_a_trial": PROBE_CUT,
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
    for cut, mode, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "cut": None if cut is None else float(cut),
            "reentry": mode,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "sessions_forced_while_trend_on": None if name == OFF_NAME else forced[name],
            "note": "The current book. No book breaker. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Book-level drawdown breaker on the UPRO sleeve",
        "hypothesis": HYPOTHESIS + "\n\n" + measured,
        "status": "tested",
        "kept": kept,
        "overlay": "book_breaker",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "cuts": list(CUTS),
            "modes": list(MODES),
            "reentry_fraction": REENTRY_FRACTION,
            "live_cell": OFF_NAME,
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "live_book_drawdown": live_stats,
            "sessions_forced_while_trend_on": forced,
            "sessions_differing_from_live": differ,
            "trips": trips,
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
        "book_breaker_rule": diagnostics.get("book_breaker_rule"),
        "book_breaker_on": diagnostics.get("book_breaker_on"),
        "book_drawdown": diagnostics.get("book_drawdown"),
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "rebalance_mix": signal.get("rebalance_mix"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    annotated = _publish_lead(signal, kept, result)
    entry["signal"]["published_date"] = annotated.get("date")
    entry["signal"]["published_as_of"] = annotated.get("as_of")
    hedge._write_outputs(entry)
    again_scale, again_info = kept_book_breaker(
        HERE / "iterations.json",
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        bundle["rf"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_scale is not None or again_info is not None:
                raise RuntimeError("a kept breaker whose current pick is the live book reloaded an override")
        elif again_scale is None or again_info is None:
            raise RuntimeError("a kept book breaker did not reload")
        elif again_info["book_breaker_rule"] != last_pick:
            raise RuntimeError("a kept book breaker reloaded a different cell")
        elif bool(again_info["book_breaker_on"]) != bool(again_scale.iloc[-1] == 0.0):
            raise RuntimeError("a kept book breaker reloaded a scale that does not match its flag")
    elif again_scale is not None or again_info is not None:
        raise RuntimeError("a rejected book breaker reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "iteration_id": iteration_id,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "lead_oos_before_judgment": lead_before,
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
        "live_book_drawdown": live_stats,
        "forced": forced,
        "differ": differ,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
