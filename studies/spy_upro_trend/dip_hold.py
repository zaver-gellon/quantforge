#!/usr/bin/env python3
"""One walk-forward test: the exit of the three-down-close dip.

The exits, the caps, and the keep rule are fixed in
``quantforge.strategies.diphold`` and ``quantforge.eval.materiality``. Do not
add a cell after seeing this sample.

The first-up exit with a five-session cap is the current book and is not a
new trial.

Run from the repo root:

    python studies/spy_upro_trend/dip_hold.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_trend_core
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.strategies.billgear import kept_bill_gear
from quantforge.strategies.confirm import kept_confirm_trend
from quantforge.strategies.corehold import kept_core_hold
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.cushion import kept_cushion
from quantforge.strategies.curve import kept_curve_clock
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.diphold import (
    EXITS,
    HOLDS,
    OFF_NAME,
    cells,
    cell_name,
    dip_hold_flag,
    episode_lengths,
    kept_dip_hold,
)
from quantforge.strategies.dipsize import kept_dip_size
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.drift import kept_drift_mix
from quantforge.strategies.symdrift import kept_sym_drift
from quantforge.strategies.ema import kept_ema_trend
from quantforge.strategies.idle import idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.onslice import kept_ontrend_slice
from quantforge.strategies.pause import kept_streak_pause
from quantforge.strategies.reentry import kept_fast_reentry
from quantforge.strategies.sleevediv import ASSETS as SLEEVE_ASSETS
from quantforge.strategies.sleevediv import sleeve_cash_signal_kwargs
from quantforge.strategies.sleevegear import kept_sleeve_gear
from quantforge.strategies.termgate import kept_term_gate
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volgear import kept_vol_gear
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-13-ontrend-slice"
STUDY = "spy_upro_trend"
FAMILY = "dip_hold"
CALENDAR_DAY = "2026-11-15"

HYPOTHESIS = (
    "The live book's maximum drawdown is still about -46.8%. The 2026-11-13 "
    "on-trend gold or Treasury slice, and the sleeve rules since 2026-11-06, "
    "either cut the bull market or barely left the current book. None of them "
    "changed the dip. The 2026-11-03 sized dip changed only how much of the "
    "SPY half becomes SSO, and whether the 200-day band had to be on. It left "
    "the three-down entry, the first-up exit, and the five-session cap "
    "untouched. That exit sells the rebound and keeps the 2x stake while SPY "
    "is still falling: an up close ends the hold, and a down close extends it "
    "until the cap. The stored full-sample table already has this overlay at "
    "turnover 13.6 a year against 0.8 for the same book without it. This test "
    "does not add a sleeve gate, a second average, or a new entry rule. It "
    "keeps the three-down entry on the SPY half only. The exit is the live "
    "first-up rule, the inverted rule that leaves on the first down close of "
    "the hold, or a fixed hold that ignores the sign of the close. Caps are "
    "3, 5, and 10 sessions. The first-up cap of 5 is the current book. It is "
    "the walk-forward's option to keep the present rule and is not a new "
    "trial. The other eight cells are new trials. Variant names are prefixed "
    "so the current book sorts first. A tie in the trailing excess Sharpe "
    "therefore keeps it. An exit close does not re-enter on that same close. "
    "The UPRO sleeve, the 200-day 2% band, and the quarterly mix are "
    "unchanged. Fills stay next-open with 5 bp one-way costs. Cash pays no "
    "spread. The study counts the live rule's holds before any new cell is "
    "scored. The stitched path is kept only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_dip_hold"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(exit_mode: str, hold: int) -> dict:
    return {
        "base": "dip_3_down_closes",
        "exit": exit_mode,
        "max_hold": int(hold),
        "rule": "3_down_closes",
        "same_close_reentry": False,
        "applies_to": "spy_half_only",
        "sleeve_unchanged": True,
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
        note = "The prior on-trend-slice iteration row was not in iterations.json, so the refresh has no comparison row."
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
        ending = "The stitched dip exit clears the materiality bar. It becomes the live book."
    else:
        ending = (
            "The stitched dip exit does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _simulate(study, bundle, trend, flag, index, sso, upro):
    aligned = flag.reindex(bundle["cal"])
    if aligned.isna().any():
        raise RuntimeError("dip hold flag does not cover the price calendar")
    result = run_trend_core(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, aligned.astype(bool), bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
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


def _measure_live(close: pd.Series, flag: pd.Series, oos_index: pd.Index) -> dict:
    """Hold facts for the current exit. This does not score a new cell."""
    lengths = episode_lengths(flag.reindex(close.index).fillna(False))
    if not lengths:
        raise RuntimeError("the live dip never held")
    held = flag.shift(1).reindex(oos_index).fillna(False).astype(bool)
    session = close.pct_change().reindex(oos_index)
    down = session[held] < 0
    return {
        "n_episodes": int(len(lengths)),
        "share_one": float(sum(item == 1 for item in lengths) / len(lengths)),
        "share_cap": float(sum(item >= 5 for item in lengths) / len(lengths)),
        "mean_length": float(sum(lengths) / len(lengths)),
        "oos_held": int(held.sum()),
        "oos_sessions": int(len(oos_index)),
        "oos_down_share": float(down.mean()) if int(held.sum()) else float("nan"),
    }


def _measured_paragraph(stats: dict) -> str:
    return (
        "Measured on the live exit before any new cell was scored. "
        f"The live rule produced {stats['n_episodes']} episodes. "
        f"{stats['share_one']:.1%} lasted one session, "
        f"{stats['share_cap']:.1%} reached the five-session cap, "
        f"and the mean length was {stats['mean_length']:.2f} sessions. "
        f"Out of sample, the overlay was held on {stats['oos_held']} of "
        f"{stats['oos_sessions']} sessions, and {stats['oos_down_share']:.1%} of "
        "those held sessions closed below the prior SPY close."
    )


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


def _write_live_signal(bundle, sso, upro, bill_yield, vix_close, vix3m_close, asset_closes) -> dict:
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
    ema_trend, ema_info = kept_ema_trend(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if ema_trend is not None and trend_override is not None:
        raise RuntimeError("the iteration log kept both an exponential filter and another trend override")
    if ema_trend is not None:
        trend_override = ema_trend
        vote_info = None
    idle_choice = kept_idle_choice(HERE / "iterations.json", as_of)
    idle_kwargs = {}
    if idle_choice is not None:
        idle_frames = load_bars(YahooSource(), [idle_choice["symbol"]], start="1993-01-01")
        idle_kwargs = idle_signal_kwargs(
            HERE / "iterations.json", bundle["spy_close"], as_of,
            {idle_choice["symbol"]: idle_frames[idle_choice["symbol"]]["close"]},
        )
    sleeve_frames = load_bars(YahooSource(), list(SLEEVE_ASSETS), start="1993-01-01")
    sleeve_kwargs = sleeve_cash_signal_kwargs(
        HERE / "iterations.json", bundle["spy_close"], as_of,
        {symbol: sleeve_frames[symbol]["close"] for symbol in SLEEVE_ASSETS},
    )
    if idle_kwargs.get("idle_symbol") and sleeve_kwargs.get("sleeve_cash_symbol"):
        raise RuntimeError("the iteration log kept both a whole-book diversifier and a sleeve diversifier")
    core_hold, core_hold_info = kept_core_hold(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    curve_on, curve_info = kept_curve_clock(
        HERE / "iterations.json", bundle["curve_spread"], as_of=as_of,
    )
    equity_trend = trend_state(bundle["spy_close"], 200, 0.02, "daily")
    sized, dip_info = kept_dip_size(
        HERE / "iterations.json", bundle["spy_close"], equity_trend, as_of=as_of,
    )
    hold_flag, hold_info = kept_dip_hold(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if sized is not None and hold_flag is not None:
        raise RuntimeError("the iteration log kept both a sized dip and a dip-hold exit")
    dip_kwargs = {}
    if sized is not None:
        dip_kwargs = {"dip_boost": sized, "dip_size_info": dip_info}
    if hold_flag is not None:
        dip_kwargs = {"dip_hold_flag": hold_flag, "dip_hold_info": hold_info}
    drift_mix, drift_info = kept_drift_mix(
        HERE / "iterations.json",
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        bundle["rf"], as_of=as_of,
    )
    sym_mix, sym_info = kept_sym_drift(
        HERE / "iterations.json",
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        bundle["rf"], as_of=as_of,
    )
    if (
        sym_mix is not None
        and drift_info is not None
        and not drift_info.get("drift_calendar", True)
    ):
        raise RuntimeError("the iteration log kept both a one-sided drift band and a two-sided drift band")
    sym_kwargs = {}
    if sym_info is not None:
        sym_kwargs["sym_info"] = sym_info
    if sym_mix is not None:
        sym_kwargs["sym_mix"] = sym_mix
    reentry_hold, reentry_info = kept_fast_reentry(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    if reentry_hold is not None and core_hold is not None:
        raise RuntimeError("the iteration log kept both an off-trend core hold and a fast re-entry")
    if reentry_hold is not None:
        core_hold, core_hold_info = reentry_hold, reentry_info
    sleeve_fast, gear_info = kept_sleeve_gear(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    sleeve_upro, gear_vol_info = kept_vol_gear(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    if sleeve_upro is not None and sleeve_fast is not None:
        raise RuntimeError("the iteration log kept both a vol gear and a sleeve gear")
    if sleeve_upro is not None and scale is not None:
        raise RuntimeError("the iteration log kept both a vol gear and a sleeve scale")
    cushion, cushion_info = kept_cushion(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if cushion is not None and (sleeve_upro is not None or sleeve_fast is not None or scale is not None):
        raise RuntimeError("the iteration log kept both a cushion scale and another sleeve rule")
    bill, bill_info = kept_bill_gear(HERE / "iterations.json", bill_yield, as_of=as_of)
    if bill is not None and (cushion is not None or sleeve_upro is not None or sleeve_fast is not None or scale is not None):
        raise RuntimeError("the iteration log kept both a bill-yield gear and another sleeve rule")
    pause, pause_info = kept_streak_pause(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if pause is not None and (
        bill is not None or cushion is not None or sleeve_upro is not None
        or sleeve_fast is not None or scale is not None
    ):
        raise RuntimeError("the iteration log kept both a down-streak pause and another sleeve rule")
    term, term_info = kept_term_gate(
        HERE / "iterations.json", bundle["spy_close"], vix_close, vix3m_close, as_of=as_of,
    )
    if term is not None and (
        pause is not None or bill is not None or cushion is not None
        or sleeve_upro is not None or sleeve_fast is not None or scale is not None
    ):
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and another sleeve rule")
    slice_symbol, slice_weight, slice_ready, slice_info = kept_ontrend_slice(
        HERE / "iterations.json", asset_closes, as_of=as_of, calendar=bundle["spy_close"].index,
    )
    if slice_symbol is not None and (
        term is not None or pause is not None or bill is not None or cushion is not None
        or sleeve_upro is not None or sleeve_fast is not None or scale is not None
    ):
        raise RuntimeError("the iteration log kept both an on-trend slice and another sleeve rule")
    slice_kwargs = {}
    if slice_symbol is not None:
        slice_kwargs = {
            "slice_symbol": slice_symbol,
            "slice_weight": slice_weight,
            "slice_ready": slice_ready,
            "slice_info": slice_info,
        }
    elif slice_info is not None:
        slice_kwargs = {"slice_info": slice_info}
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
        **sym_kwargs,
        sleeve_fast=sleeve_fast, sleeve_gear_info=gear_info,
        sleeve_upro_share=sleeve_upro, vol_gear_info=gear_vol_info,
        cushion_share=cushion, cushion_info=cushion_info,
        bill_share=bill, bill_info=bill_info,
        pause_share=pause, pause_info=pause_info,
        term_share=term, term_info=term_info,
        ema_info=ema_info,
        **slice_kwargs,
        **dip_kwargs,
        **idle_kwargs,
        **sleeve_kwargs,
    )
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_kwargs.get("idle_symbol"):
        allowed.add(idle_kwargs["idle_symbol"])
    if sleeve_kwargs.get("sleeve_cash_symbol"):
        allowed.add(sleeve_kwargs["sleeve_cash_symbol"])
    if slice_symbol is not None:
        allowed.add(slice_symbol)
    write_signal(payload, HERE / "signals.json", allowed_tickers=frozenset(allowed))
    if hold_info and not hold_info.get("dip_hold_live", True):
        title = "Trend-core whose dip keeps the bounce and drops the continuation"
    elif slice_info and not slice_info.get("slice_live", True):
        title = "Trend-core that holds a fixed gold or Treasury slice in the on-trend sleeve"
    elif dip_info:
        title = "Trend-core with the kept sized dip overlay"
    else:
        title = "Trend-core (2% band, dip overlay on the SPY half)"
    write_signals_note(HERE / "signals.md", [("signals.json", title, payload)])
    (HERE / "signals.md").write_text(
        (HERE / "signals.md").read_text()
        + "\n`signals_flagship.json` is written by `python studies/flagship/run.py` and is not this fixed book.\n"
    )
    return payload


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
    iteration_id = f"{iteration_date}-dip-hold"
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
    live_flag = dip_hold_flag(bundle["spy_close"], "up", 5).reindex(bundle["cal"]).astype(bool)
    active = dip_active(bundle["spy_close"], "3_down_closes", 5).reindex(bundle["cal"]).astype(bool)
    if not live_flag.equals(active):
        raise RuntimeError("the first-up five-session exit does not match the live dip")
    identity, _identity_turn = _simulate(study, bundle, live, live_flag, sample_index, sso, upro)
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"the live exit does not reproduce the trend core (max abs return gap {gap})")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    oos_index = sample_index[sample_index >= decisions[0]]
    measured = _measure_live(bundle["spy_close"], live_flag, oos_index)
    measured_text = _measured_paragraph(measured)

    flags = {OFF_NAME: live_flag}
    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    widest = 0.0
    for exit_mode, hold in cells():
        name = cell_name(exit_mode, hold)
        flag = dip_hold_flag(bundle["spy_close"], exit_mode, hold).reindex(bundle["cal"]).astype(bool)
        returns, traded = _simulate(study, bundle, live, flag, sample_index, sso, upro)
        widest = max(widest, float((returns - baseline).abs().max()))
        flags[name] = flag
        paths[name] = returns
        turns[name] = traded
    if widest <= 1e-12:
        raise RuntimeError("every dip exit reproduced the current book")

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
    stitched_flag = _stitch(
        {name: series.astype(float) for name, series in flags.items()},
        walk.picks,
        walk.returns.index,
    ).fillna(0.0) > 0.5
    live_oos_flag = live_flag.reindex(walk.returns.index).astype(bool)
    changed_sessions = int((stitched_flag != live_oos_flag).sum())
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

    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Of {int(len(walk.returns))} out-of-sample decision closes, the stitched exit "
        f"differed from the live dip on {changed_sessions}. "
        f"The live exit matches the trend core (max abs return gap {gap:.3e}). "
        f"{measured_text}"
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [
        (exit_mode, hold, cell_name(exit_mode, hold)) for exit_mode, hold in cells()
    ]
    for exit_mode, hold, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(exit_mode, hold),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for exit_mode, hold, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(exit_mode, hold),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {cell_name(exit_mode, hold): (exit_mode, hold) for exit_mode, hold in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        exit_mode, hold = by_name[last_pick]
        neighbor_paths = {
            cell_name(*neighbor): paths[cell_name(*neighbor)]
            for neighbor in adjacent_cells((exit_mode, hold), [list(EXITS), list(HOLDS)])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="dip_hold_walk_forward",
        params={
            "bar": "materiality_v1",
            "exits": list(EXITS),
            "holds": list(HOLDS),
            "live_cell": OFF_NAME,
            "live_cell_counts": False,
            "rule": "3_down_closes",
            "applies_to": "spy_half_only",
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
    for exit_mode, hold, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "exit": "up" if exit_mode is None else exit_mode,
            "max_hold": 5 if hold is None else int(hold),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. First up close, or five sessions. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Dip exit that keeps the bounce and drops the continuation",
        "hypothesis": HYPOTHESIS + "\n\n" + measured_text,
        "status": "tested",
        "kept": kept,
        "overlay": "dip_hold",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "exits": list(EXITS),
            "holds": list(HOLDS),
            "live_cell": OFF_NAME,
            "applies_to": "spy_half_only",
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "live_hold": measured,
            "identity_max_abs_return_gap": gap,
            "lead_oos_before_judgment": lead_before,
            "oos_exit": {
                "sessions": int(len(walk.returns)),
                "decision_closes_different_from_live": changed_sessions,
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
    signal = _write_live_signal(
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
        entry["decision_note"] + level_sentence + " " + _signal_sentence(previous_signal, signal)
    )
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": diagnostics.get("trend_on"),
        "dip_on": diagnostics.get("dip_on"),
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "dip_hold_rule": diagnostics.get("dip_hold_rule"),
        "dip_hold_exit": diagnostics.get("dip_hold_exit"),
        "dip_hold_max": diagnostics.get("dip_hold_max"),
        "dip_hold_live": diagnostics.get("dip_hold_live"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again_flag, again_info = kept_dip_hold(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_info is None or not again_info.get("dip_hold_live", False):
                raise RuntimeError("a kept dip exit whose current pick is the live book did not reload it")
        elif again_flag is None or again_info is None or again_info.get("dip_hold_live", True):
            raise RuntimeError("a kept dip exit did not reload from the iteration log")
        elif again_info["dip_hold_exit"] != by_name[last_pick][0]:
            raise RuntimeError("a kept dip exit reloaded a different exit")
        elif int(again_info["dip_hold_max"]) != int(by_name[last_pick][1]):
            raise RuntimeError("a kept dip exit reloaded a different cap")
    elif again_flag is not None or again_info is not None:
        raise RuntimeError("a rejected dip exit reloaded as if it were kept")
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
