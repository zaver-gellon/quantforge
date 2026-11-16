#!/usr/bin/env python3
"""One walk-forward test: pause the on-trend UPRO sleeve after a down streak.

The streaks, the floors, the five-session cap, and the keep rule are fixed in
``quantforge.strategies.pause`` and ``quantforge.eval.materiality``. Do not
add a cell after seeing this sample.

The current book, which keeps the sleeve at 3x while the 200-day band is on,
is not a new trial.

Run from the repo root:

    python studies/spy_upro_trend/streak_pause.py
"""

from __future__ import annotations

import importlib.util
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
from quantforge.strategies.idle import idle_signal_kwargs, kept_idle_choice
from quantforge.strategies.pause import (
    FLOORS,
    MAX_HOLD,
    OFF_NAME,
    STREAKS,
    cells,
    kept_streak_pause,
    pause_share,
    variant_name,
)
from quantforge.strategies.reentry import kept_fast_reentry
from quantforge.strategies.sleevediv import ASSETS, sleeve_cash_signal_kwargs
from quantforge.strategies.sleevegear import kept_sleeve_gear
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.volgear import kept_vol_gear
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-11-11-bill-gear"
STUDY = "spy_upro_trend"
FAMILY = "streak_pause"

HYPOTHESIS = (
    "The live book's maximum drawdown is still about -46.8%, and the book "
    "trails SPY in the fast 2020 and 2022 breaks, because the 3x sleeve stays "
    "fully invested until SPY closes through the lower side of its 200-day "
    "band. The last three sleeve rules were rejected for cutting that sleeve "
    "on a large share of the sessions when the band was on. The 2026-11-09 "
    "realized-vol gear was on for 489 of 4137 band-on sessions and the daily "
    "improvement had a negative Sharpe. The 2026-11-10 cushion scale was "
    "never selected. The 2026-11-11 bill-yield gear was on for 1249 of 4139 "
    "band-on sessions, and the stitched book gave up about 2.7 points of "
    "CAGR because the cut landed on high-bill bull markets. All three read a "
    "slow state, volatility or the price gap or the bill, and hold the cut "
    "until that state heals. The 2026-10-29 trailing-peak stop is the same "
    "shape at a longer horizon: it parks the sleeve in T-bills after an 8% "
    "to 16% drop from a 63- to 252-session high, and it was rejected. The "
    "2026-11-03 sized dip changed only how much of the SPY half becomes SSO. "
    "It left the UPRO half at 3x, the stitched Sharpe rose by 0.006, and the "
    "drawdown did not move. This test does not add a second average, does "
    "not read volatility, the bill, or the cushion, and does not sell the "
    "sleeve into T-bills while the 200-day band is on. It uses a short clock "
    "on the sleeve only. After K strictly down SPY closes, and only when the "
    "200-day 2% band is on at that close, the sleeve's UPRO share drops to a "
    "floor. The rest of the sleeve is SPY, so the book stays in equities. "
    "The pause ends at the next open after the first up close, or after five "
    "sessions, whichever comes first. Five is the live dip's cap and is not "
    "a searched axis. A same-close exit is not re-armed. If the band turns "
    "off while a pause is held, the pause flattens at that close and does "
    "not carry into the next time the band turns on. The core's "
    "three-down-close dip, which can still swap the SPY half into SSO, is a "
    "separate clock and is unchanged. The quarterly mix is unchanged. K is "
    "2, 3, and 4. Floors are 0 and 1/2, so the sleeve goes to SPY or to 2x. "
    "The six cells are new trials. The current book, which keeps the share "
    "at 1 whenever the band is on, is the walk-forward's option to keep the "
    "present rule and is not a new trial. Ties in the trailing excess Sharpe "
    "break alphabetically, so the current book wins a tie. Fills stay "
    "next-open with 5 bp one-way costs. Cash pays no spread. The blend pays "
    "5 bp on the SPY dollars and the UPRO dollars that trade. The stitched "
    "path is kept only when it clears the materiality bar."
)


def _load(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_for_streak_pause"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(streak: int, floor: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "streak": int(streak),
        "floor": float(floor),
        "max_hold": MAX_HOLD,
        "exit": "first_up_close_or_five_sessions_or_trend_off",
        "arm": "streak_strictly_down_and_trend_on",
        "residual": "SPY",
        "applies_to": "sleeve_only_when_trend_on",
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
        note = "The prior bill-yield iteration row was not in iterations.json, so the refresh has no comparison row."
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
            "The stitched down-streak pause clears the materiality bar. "
            "It becomes the live book."
        )
    else:
        ending = (
            "The stitched down-streak pause does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _write_live_signal(bundle, sso, upro, bill_yield: pd.Series) -> dict:
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
    sleeve_frames = load_bars(YahooSource(), list(ASSETS), start="1993-01-01")
    sleeve_kwargs = sleeve_cash_signal_kwargs(
        HERE / "iterations.json", bundle["spy_close"], as_of,
        {symbol: sleeve_frames[symbol]["close"] for symbol in ASSETS},
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
        ema_info=ema_info,
        **dip_kwargs,
        **idle_kwargs,
        **sleeve_kwargs,
    )
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_kwargs.get("idle_symbol"):
        allowed.add(idle_kwargs["idle_symbol"])
    if sleeve_kwargs.get("sleeve_cash_symbol"):
        allowed.add(sleeve_kwargs["sleeve_cash_symbol"])
    write_signal(payload, HERE / "signals.json", allowed_tickers=frozenset(allowed))
    if pause_info and not pause_info.get("pause_live", True):
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
    last_close = spy_listed.index[-1]
    iteration_date = str(last_close.date())
    iteration_id = f"{iteration_date}-streak-pause"
    if last_close.date().isoformat() == "2026-11-12":
        published = " The 2026-11-12 close is published."
    else:
        published = (
            f" The 2026-11-12 close is not in the refreshed SPY series. "
            f"The latest available close is {last_close.date()}."
        )
    coverage = [
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
    ]
    price_note = (
        f"After a forced Yahoo refresh, SPY closes run from {spy_listed.index[0].date()} "
        f"to {spy_listed.index[-1].date()} ({len(spy_listed)} sessions). "
        f"SSO ends {sso_listed.index[-1].date()}, "
        f"UPRO ends {upro_listed.index[-1].date()}, "
        f"^IRX ends {irx_listed.index[-1].date()}."
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
    if bundle["spy_close"].index[-1] != spy_listed.index[-1]:
        raise RuntimeError("the prepared SPY sample does not end on the refreshed close")

    sso = synth[2]
    upro = synth[3]
    baseline, base_turn = hedge._simulate(study, bundle, live, sample_index, sso, upro)
    ones = pd.Series(1.0, index=bundle["cal"])
    identity, _identity_turn = gear._simulate(study, bundle, live, ones, sample_index, sso, upro)
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"a full 3x sleeve does not reproduce the trend core (max abs return gap {gap})")

    paths = {OFF_NAME: baseline}
    turns = {OFF_NAME: base_turn}
    shares = {OFF_NAME: pd.Series(1.0, index=sample_index)}
    widest = 0.0
    for streak, floor in cells():
        share = pause_share(bundle["spy_close"], live, streak, floor).reindex(bundle["cal"])
        if share.isna().any():
            raise RuntimeError(f"share is missing for streak {streak} floor {floor}")
        name = variant_name(streak, floor)
        returns, traded = gear._simulate(study, bundle, live, share, sample_index, sso, upro)
        widest = max(widest, float((returns - baseline).abs().max()))
        paths[name] = returns
        turns[name] = traded
        shares[name] = share.reindex(sample_index).astype(float)
    if widest <= 1e-12:
        raise RuntimeError("every down-streak cell reproduced the current book")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
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

    stitched_turn = gear._stitch(turns, walk.picks, walk.returns.index).fillna(0.0)
    stitched_share = gear._stitch(shares, walk.picks, walk.returns.index).astype(float)
    if stitched_share.isna().any():
        raise RuntimeError("the stitched UPRO share is missing sessions")
    trend_oos = live.reindex(walk.returns.index).fillna(False).astype(bool)
    reduced = trend_oos & (stitched_share < 1.0 - 1e-8)
    on_share = stitched_share[trend_oos]
    mean_leverage = float((1.0 + 2.0 * on_share).mean()) if len(on_share) else float("nan")
    dip_oos = dip_active(bundle["spy_close"], "3_down_closes").reindex(walk.returns.index).fillna(False).astype(bool)
    overlap_dip = int((reduced & dip_oos).sum())
    rf = bundle["rf"]
    base_oos_turn = base_turn.reindex(walk.returns.index).fillna(0.0)
    base_oos_summary = summarize(current_oos, rf=rf.reindex(current_oos.index), turnover=base_oos_turn)
    cand_oos_summary = summarize(walk.returns, rf=rf.reindex(walk.returns.index), turnover=stitched_turn)
    # The lead book's out-of-sample numbers are fixed here, before the candidate is judged.
    lead_oos = summarize(current_oos, rf=rf.reindex(current_oos.index))
    extra = (
        f"Out-of-sample turnover per year {base_oos_summary['turnover_per_year']:.2f} -> "
        f"{cand_oos_summary['turnover_per_year']:.2f}. "
        f"Of {int(trend_oos.sum())} out-of-sample sessions with the 200-day band on, "
        f"the stitched rule set the sleeve's UPRO share below 1 on {int(reduced.sum())}. "
        f"Mean close-known leverage on band-on sessions was {mean_leverage:.3f}. "
        f"Of those paused sessions, {overlap_dip} also had the core's three-down-close dip on. "
        f"A share of 1 matches the trend core (max abs return gap {gap:.3e})."
    )

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    ordered = [(None, None, OFF_NAME)] + [
        (streak, floor, variant_name(streak, floor)) for streak, floor in cells()
    ]
    for streak, floor, name in ordered:
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
            turnover=turns[name],
        )
        if name == OFF_NAME:
            continue
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(streak, floor),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for streak, floor, name in ordered:
        if name == OFF_NAME:
            continue
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(streak, floor),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(streak, floor): (streak, floor) for streak, floor in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = None
    else:
        cell_path = paths[last_pick]
        streak, floor = by_name[last_pick]
        neighbor_paths = {
            variant_name(*neighbor): paths[variant_name(*neighbor)]
            for neighbor in adjacent_cells((streak, floor), [list(STREAKS), list(FLOORS)])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    if abs(float(result["oos"]["current"]["cagr"]) - float(lead_oos["cagr"])) > 1e-12:
        raise RuntimeError("the scored lead book does not match the pre-judgment lead numbers")
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="streak_pause_walk_forward",
        params={
            "bar": "materiality_v1",
            "streaks": list(STREAKS),
            "floors": list(FLOORS),
            "max_hold": MAX_HOLD,
            "residual": "SPY",
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
    for streak, floor, name in ordered:
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": name != OFF_NAME,
            "streak": None if streak is None else int(streak),
            "floor": None if floor is None else float(floor),
            "max_hold": MAX_HOLD,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "turnover_per_year": row.get("turnover_per_year"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "note": "The current book. The on-trend sleeve stays at 3x. Already counted." if name == OFF_NAME else None,
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    refresh["note"] = price_note + " " + refresh["note"]
    entry = {
        "id": iteration_id,
        "date": iteration_date,
        "title": "Down-streak pause on the UPRO sleeve",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "streak_pause",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh, extra),
        "data": {
            "streaks": list(STREAKS),
            "floors": list(FLOORS),
            "max_hold": MAX_HOLD,
            "residual": "SPY",
            "live_cell": OFF_NAME,
            "applies_to": "sleeve_only_when_trend_on",
            "price_sample_end": sample_end,
            "spy_close_end": str(spy_listed.index[-1].date()),
            "coverage": coverage,
            "identity_max_abs_return_gap": gap,
            "lead_oos_before_judgment": {
                "cagr": lead_oos["cagr"],
                "sharpe": lead_oos["sharpe"],
                "max_drawdown": lead_oos["max_drawdown"],
                "n_days": lead_oos["n_days"],
                "start": lead_oos["start"],
                "end": lead_oos["end"],
            },
            "oos_band_on": {
                "sessions": int(len(walk.returns)),
                "band_on": int(trend_oos.sum()),
                "upro_share_below_one": int(reduced.sum()),
                "mean_leverage_when_band_on": mean_leverage,
                "paused_and_core_dip_on": overlap_dip,
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
    signal = _write_live_signal(bundle, sso, upro, bill_yield)
    entry["decision_note"] = entry["decision_note"] + " " + gear._signal_sentence(previous_signal, signal)
    diagnostics = signal.get("diagnostics") or {}
    entry["signal"] = {
        "date": signal.get("date"),
        "signal_date": signal.get("signal_date"),
        "weights": signal.get("weights"),
        "trend_on": diagnostics.get("trend_on"),
        "dip_on": diagnostics.get("dip_on"),
        "reference_close": diagnostics.get("reference_close"),
        "sma": diagnostics.get("sma"),
        "pause_rule": diagnostics.get("pause_rule"),
        "pause_share": diagnostics.get("pause_share"),
        "pause_on": diagnostics.get("pause_on"),
        "sleeve_leverage_applied": diagnostics.get("sleeve_leverage_applied"),
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again_share, again_info = kept_streak_pause(
        HERE / "iterations.json", bundle["spy_close"], bundle["spy_close"].index[-1],
    )
    if kept:
        if last_pick == OFF_NAME:
            if again_info and not again_info.get("pause_live", False):
                raise RuntimeError("a kept down-streak pause whose current pick is the plain book reloaded a cell")
        elif again_share is None or again_info is None or again_info.get("pause_live", True):
            raise RuntimeError("a kept down-streak pause did not reload from the iteration log")
    elif again_share is not None or again_info is not None:
        raise RuntimeError("a rejected down-streak pause reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "iteration_id": iteration_id,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "lead_oos_before_judgment": entry["data"]["lead_oos_before_judgment"],
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
        "signal_sentence": gear._signal_sentence(previous_signal, signal),
        "signal": entry["signal"],
        "coverage": coverage,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
