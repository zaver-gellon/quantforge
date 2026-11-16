#!/usr/bin/env python3
"""One walk-forward test: a fixed gold or Treasury slice inside the on-trend sleeve.

The assets, the weights, and the keep rule are fixed in
``quantforge.strategies.onslice`` and ``quantforge.eval.materiality``. Do not
add a cell after seeing this sample.

The current book, which holds no diversifier, is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/ontrend_slice.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_sleeve_slice
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
from quantforge.strategies.dipsize import kept_dip_size
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.drift import kept_drift_mix
from quantforge.strategies.ema import kept_ema_trend
from quantforge.strategies.idle import align_ohlc, idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.onslice import (
    ASSETS,
    OFF_NAME,
    SLICES,
    cells,
    kept_ontrend_slice,
    variant_name,
)
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
PRIOR_ID = "2026-11-13-term-gate"
STUDY = "spy_upro_trend"
FAMILY = "ontrend_slice"
CALENDAR_DAY = "2026-11-14"

HYPOTHESIS = (
    "The live book's maximum drawdown is still about -46.8%. The VIX "
    "term-structure gate, the last sleeve rule, was on for 152 of 4141 "
    "band-on sessions. Its out-of-sample Sharpe rose by 0.021, short of "
    "0.05, the drawdown did not move, and removing 2020 flipped the Sharpe "
    "change negative. The sleeve rules since 2026-11-06, an exponential "
    "average, a shorter trend that swaps UPRO for SPY, realized volatility, "
    "the cushion over the 200-day average, the T-bill yield, a down-streak "
    "pause, and that term gate, all read a state of the same equity book. "
    "They either cut bull markets or almost never left the current book. "
    "Rules that sold the SPY half for the whole off-trend spell did move "
    "the 2009 trough and were rejected, because the cash stake also sat out "
    "the rebound. Bond and gold were tested only as a replacement for the "
    "T-bill sleeve while the equity trend was off. That stitched book added "
    "about 0.016 Sharpe and less than a point of drawdown, and the gain was "
    "2008. They were never held against the 3x sleeve during the on-trend "
    "window, which is where the late exits of 2020 and 2022 happen, while "
    "the band is still on. This test does not add a gate, a second average, "
    "or an off-trend replacement of cash. While the 200-day 2% band is on "
    "and the diversifier has a usable close, a fixed slice of the book is "
    "taken from the UPRO sleeve and held in that asset with no filter of "
    "its own. The rest of the sleeve stays UPRO. The SPY core, the "
    "three-down-close dip, and the quarterly mix are unchanged. While the "
    "band is off, or before the diversifier's first usable close, the slice "
    "is not held and the book is the current trend core. Assets are GLD, "
    "IEF, and TLT, alphabetical. Slices are 10, 20, and 30 percent of the "
    "book, which is 20, 40, and 60 percent of the 50 percent sleeve. The "
    "nine cells are new trials. The current book, which holds no slice, is "
    "the walk-forward's option to keep the present rule and is not a new "
    "trial. Variant names are prefixed so slice_off sorts first. A tie in "
    "the trailing excess Sharpe therefore keeps the current book. Sessions "
    "before the asset can be held are the trend core and do not count "
    "toward the 756-session minimum. Fills stay next-open with 5 bp one-way "
    "costs. Cash pays no spread. The diversifier pays the same 5 bp. The "
    "stitched path is kept only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_ontrend_slice"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(asset: str, weight: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "asset": asset,
        "slice_weight": float(weight),
        "sleeve_weight": 0.5,
        "filter": "none",
        "applies_to": "sleeve_only_when_trend_on",
        "missing": "slice_off",
        "fill": "gap_up_to_five_sessions",
        "core_unchanged": True,
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
        note = "The prior term-structure iteration row was not in iterations.json, so the refresh has no comparison row."
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
        ending = (
            "The stitched on-trend diversifier slice clears the materiality bar. "
            "It becomes the live book."
        )
    else:
        ending = (
            "The stitched on-trend diversifier slice does not clear the materiality bar. "
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


def _simulate(study, bundle, trend, div_open, div_close, ready, weight, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_sleeve_slice(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        div_open, div_close, trend, ready, dip, bundle["masks"]["quarterly"],
        0.5, 0.5, float(weight), bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
        div_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


def _walk_history(returns: pd.Series, ready: pd.Series) -> tuple[pd.Series, str]:
    """Drop sessions before the slice can be held. Those days are the trend core."""
    prior = ready.reindex(returns.index).fillna(False).astype(bool).shift(1).fillna(False)
    if not bool(prior.any()):
        raise RuntimeError("diversifier never became holdable")
    first = prior[prior].index[0]
    scored = returns.copy()
    scored.loc[scored.index < first] = float("nan")
    return scored, str(pd.Timestamp(first).date())


def _held_flag(trend: pd.Series, ready: pd.Series, index: pd.Index) -> pd.Series:
    """True on sessions whose previous close held the slice."""
    flag = trend.reindex(index).fillna(False).astype(bool) & ready.reindex(index).fillna(False).astype(bool)
    return flag.shift(1).fillna(False)


def _write_live_signal(
    bundle, sso, upro, bill_yield: pd.Series, vix_close: pd.Series, vix3m_close: pd.Series,
    asset_closes: dict[str, pd.Series],
) -> dict:
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
    cushion, cushion_info = kept_cushion(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    if cushion is not None and sleeve_upro is not None:
        raise RuntimeError("the iteration log kept both a cushion scale and a vol gear")
    if cushion is not None and sleeve_fast is not None:
        raise RuntimeError("the iteration log kept both a cushion scale and a sleeve gear")
    if cushion is not None and scale is not None:
        raise RuntimeError("the iteration log kept both a cushion scale and a sleeve scale")
    bill, bill_info = kept_bill_gear(
        HERE / "iterations.json", bill_yield, as_of=as_of,
    )
    if bill is not None and cushion is not None:
        raise RuntimeError("the iteration log kept both a bill-yield gear and a cushion scale")
    if bill is not None and sleeve_upro is not None:
        raise RuntimeError("the iteration log kept both a bill-yield gear and a vol gear")
    if bill is not None and sleeve_fast is not None:
        raise RuntimeError("the iteration log kept both a bill-yield gear and a sleeve gear")
    if bill is not None and scale is not None:
        raise RuntimeError("the iteration log kept both a bill-yield gear and a sleeve scale")
    pause, pause_info = kept_streak_pause(
        HERE / "iterations.json", bundle["spy_close"], as_of=as_of,
    )
    if pause is not None and bill is not None:
        raise RuntimeError("the iteration log kept both a down-streak pause and a bill-yield gear")
    if pause is not None and cushion is not None:
        raise RuntimeError("the iteration log kept both a down-streak pause and a cushion scale")
    if pause is not None and sleeve_upro is not None:
        raise RuntimeError("the iteration log kept both a down-streak pause and a vol gear")
    if pause is not None and sleeve_fast is not None:
        raise RuntimeError("the iteration log kept both a down-streak pause and a sleeve gear")
    if pause is not None and scale is not None:
        raise RuntimeError("the iteration log kept both a down-streak pause and a sleeve scale")
    term, term_info = kept_term_gate(
        HERE / "iterations.json", bundle["spy_close"], vix_close, vix3m_close, as_of=as_of,
    )
    if term is not None and pause is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a down-streak pause")
    if term is not None and bill is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a bill-yield gear")
    if term is not None and cushion is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a cushion scale")
    if term is not None and sleeve_upro is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a vol gear")
    if term is not None and sleeve_fast is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a sleeve gear")
    if term is not None and scale is not None:
        raise RuntimeError("the iteration log kept both a VIX term-structure gate and a sleeve scale")
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
    if slice_info and not slice_info.get("slice_live", True):
        title = "Trend-core that holds a fixed gold or Treasury slice in the on-trend sleeve"
    elif term_info and not term_info.get("term_live", True):
        title = "Trend-core that cuts the UPRO sleeve when the VIX term structure is inverted"
    elif pause_info and not pause_info.get("pause_live", True):
        title = "Trend-core that pauses the UPRO sleeve after a down streak"
    elif bill_info and not bill_info.get("bill_live", True):
        title = "Trend-core that holds the sleeve in SPY when the T-bill yield is above the kept threshold"
    elif cushion_info and not cushion_info.get("cushion_live", True):
        title = "Trend-core that scales the UPRO sleeve by its cushion over the 200-day average"
    elif gear_vol_info and not gear_vol_info.get("vol_gear_live", True):
        title = "Trend-core that scales the UPRO sleeve's leverage by realized volatility"
    elif sleeve_kwargs.get("sleeve_cash_info"):
        title = "Trend-core that invests the off-trend sleeve in the kept bond or gold trend"
    elif ema_info:
        title = "Trend-core whose UPRO sleeve follows the kept exponential average"
    elif drift_info and not drift_info.get("drift_calendar", True):
        title = "Trend-core that resets the mix on the kept UPRO drift band"
    elif dip_info:
        title = "Trend-core with the kept sized dip overlay"
    elif curve_info:
        title = "Trend-core that sits in T-bills on the kept yield-curve recession clock"
    elif reentry_info and not reentry_info.get("reentry_live", False):
        title = "Trend-core that re-enters the SPY half on a shorter trend while the 200-day band is off"
    elif gear_info and not gear_info.get("sleeve_gear_live", True):
        title = "Trend-core that holds the sleeve in SPY when a shorter trend is off"
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


def _fmt_num(value: float, digits: int = 4) -> str:
    number = float(value)
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def _price_place(close_px: float, sma: float) -> str:
    if close_px > sma * 1.02:
        return "above the upper band"
    if close_px < sma * 0.98:
        return "below the lower band"
    return "inside the 2% band"


def main() -> None:
    symbols = ["SPY", "SSO", "UPRO", "^IRX", "^VIX", "^VIX3M", *ASSETS]
    refresh_frames = load_bars(YahooSource(), symbols, start="1990-01-01", refresh=True)
    listed = {symbol: refresh_frames[symbol]["close"].dropna() for symbol in symbols}
    last_close = listed["SPY"].index[-1]
    iteration_date = str(last_close.date())
    iteration_id = f"{iteration_date}-ontrend-slice"
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
    gear = _load("bill_gear.py")
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
    prepared = {}
    ready_by_asset = {}
    for asset in ASSETS:
        div_open, div_close = align_ohlc(
            refresh_frames[asset]["open"], refresh_frames[asset]["close"], bundle["cal"],
        )
        ready = div_close.notna() & (div_close > 0)
        prepared[asset] = (div_open, div_close, ready.rename("slice_ready"))
        ready_by_asset[asset] = ready
    blank_open = pd.Series(float("nan"), index=bundle["cal"])
    blank_close = pd.Series(float("nan"), index=bundle["cal"])
    never = pd.Series(False, index=bundle["cal"])
    identity, _identity_turn = _simulate(
        study, bundle, live, blank_open, blank_close, never, 0.20, sample_index, sso, upro,
    )
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"an unlisted slice does not reproduce the trend core (max abs return gap {gap})")

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    scored_paths = {OFF_NAME: baseline}
    held = {OFF_NAME: pd.Series(False, index=sample_index)}
    history_start = {OFF_NAME: str(sample_index[0].date())}
    widest = 0.0
    for asset, weight in cells():
        div_open, div_close, ready = prepared[asset]
        name = variant_name(asset, weight)
        returns, traded = _simulate(
            study, bundle, live, div_open, div_close, ready, weight, sample_index, sso, upro,
        )
        widest = max(widest, float((returns - baseline).abs().max()))
        paths[name] = returns
        turns[name] = traded
        scored_paths[name], history_start[name] = _walk_history(returns, ready)
        held[name] = _held_flag(live, ready, sample_index)
    if widest <= 1e-12:
        raise RuntimeError("every diversifier slice reproduced the current book")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        scored_paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
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
    stitched_held = _stitch(
        {name: series.astype(float) for name, series in held.items()},
        walk.picks,
        walk.returns.index,
    ).fillna(0.0) > 0.5
    trend_decision = live.shift(1).reindex(walk.returns.index).fillna(False).astype(bool)
    if bool((stitched_held & ~trend_decision).any()):
        raise RuntimeError("the slice was held on a session whose previous close had the band off")
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
    print(json.dumps({"lead_oos_before_judgment": lead_before, "coverage": coverage}, default=float), flush=True)

    held_days = int(stitched_held.sum())
    band_on_days = int(trend_decision.sum())
    weights_held = []
    for label, name in walk.picks.items():
        if name == OFF_NAME:
            continue
        for asset, weight in cells():
            if variant_name(asset, weight) == name:
                weights_held.append(weight)
    mean_slice = float(sum(weights_held) / len(weights_held)) if weights_held else 0.0
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Of {int(len(walk.returns))} out-of-sample sessions, {band_on_days} followed a "
        f"close with the 200-day band on and the stitched rule held the slice on {held_days}. "
        f"Mean slice weight across the years a cell was selected was {mean_slice:.2f}. "
        f"An unlisted slice matches the trend core (max abs return gap {gap:.3e}). "
        + " ".join(
            f"{asset} is first usable {history_start[variant_name(asset, SLICES[0])]}."
            for asset in ASSETS
        )
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [
        (asset, weight, variant_name(asset, weight)) for asset, weight in cells()
    ]
    for asset, weight, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(asset, weight),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for asset, weight, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(asset, weight),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(asset, weight): (asset, weight) for asset, weight in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        asset, weight = by_name[last_pick]
        neighbor_paths = {
            variant_name(*neighbor): paths[variant_name(*neighbor)]
            for neighbor in adjacent_cells((asset, weight), [list(ASSETS), list(SLICES)])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="ontrend_slice_walk_forward",
        params={
            "bar": "materiality_v1",
            "assets": list(ASSETS),
            "slices": list(SLICES),
            "filter": "none",
            "live_cell": OFF_NAME,
            "live_cell_counts": False,
            "applies_to": "sleeve_only_when_trend_on",
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
    for asset, weight, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "asset": asset,
            "slice_weight": None if weight is None else float(weight),
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "history_start": history_start.get(name),
            "note": "The current book. The on-trend sleeve stays at 3x. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Constant gold or Treasury slice in the on-trend sleeve",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "ontrend_slice",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "assets": list(ASSETS),
            "slices": list(SLICES),
            "filter": "none",
            "live_cell": OFF_NAME,
            "applies_to": "sleeve_only_when_trend_on",
            "price_sample_end": sample_end,
            "spy_close_end": str(listed["SPY"].index[-1].date()),
            "coverage": coverage,
            "history_start": history_start,
            "identity_max_abs_return_gap": gap,
            "lead_oos_before_judgment": lead_before,
            "oos_slice": {
                "sessions": int(len(walk.returns)),
                "band_on_previous_close": band_on_days,
                "slice_held": held_days,
                "mean_selected_slice_weight": mean_slice,
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
    irx_pct, _unit = study._yield_series(refresh_frames["^IRX"]["close"])
    from quantforge.strategies.billgear import align_bill_yield
    bill_yield = align_bill_yield(irx_pct, bundle["cal"])
    signal = _write_live_signal(
        bundle, sso, upro, bill_yield, listed["^VIX"], listed["^VIX3M"],
        {asset: listed[asset] for asset in ASSETS},
    )
    diagnostics = signal.get("diagnostics") or {}
    close_px = float(diagnostics["reference_close"])
    sma = float(diagnostics["sma"])
    level_sentence = (
        f" SPY close {close_px:.4f} versus the 200-day average {sma:.4f} "
        f"({close_px / sma - 1.0:+.2%} versus the average), {_price_place(close_px, sma)}."
    )
    entry["decision_note"] = (
        entry["decision_note"] + level_sentence + " " + gear._signal_sentence(previous_signal, signal)
    )
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": diagnostics.get("trend_on"),
        "dip_on": diagnostics.get("dip_on"),
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "slice_rule": diagnostics.get("slice_rule"),
        "slice_symbol": diagnostics.get("slice_symbol"),
        "slice_weight": diagnostics.get("slice_weight"),
        "slice_applied": diagnostics.get("slice_applied"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again_symbol, again_weight, _again_ready, again_info = kept_ontrend_slice(
        HERE / "iterations.json",
        {asset: listed[asset] for asset in ASSETS},
        as_of=bundle["spy_close"].index[-1],
        calendar=bundle["spy_close"].index,
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_info is None or not again_info.get("slice_live", False):
                raise RuntimeError("a kept slice whose current pick is the plain book reloaded a cell")
        elif again_symbol is None or again_info is None or again_info.get("slice_live", True):
            raise RuntimeError("a kept on-trend slice did not reload from the iteration log")
        elif abs(float(again_weight) - float(by_name[last_pick][1])) > 1e-12:
            raise RuntimeError("a kept on-trend slice reloaded a different weight")
        elif again_symbol != by_name[last_pick][0]:
            raise RuntimeError("a kept on-trend slice reloaded a different asset")
    elif again_symbol is not None or again_info is not None:
        raise RuntimeError("a rejected on-trend slice reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "iteration_id": iteration_id,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "lead_oos_before_judgment": lead_before,
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
