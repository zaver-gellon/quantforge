#!/usr/bin/env python3
"""One walk-forward test: leave equities for a bond or gold trend.

The assets, the windows, the bands, and the keep rule are fixed in
``quantforge.strategies.idle`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The live book stays half in SPY after the 200-day band has moved the UPRO
sleeve to cash. Earlier overlays only resized that sleeve. This test, once a
diversifier's own average exists, moves the core and the sleeve together.

Run from the repo root:

    python studies/spy_upro_trend/idle_diversifier.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.data import YahooSource, load_bars
from quantforge.engine.backtest import run_idle_diversifier
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.idle import (
    ASSETS,
    BANDS,
    OFF_NAME,
    WINDOWS,
    align_ohlc,
    cells,
    idle_signal_kwargs,
    idle_state,
    kept_idle_choice,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-10-29-trailing-drawdown-stop"
ITERATION_ID = "2026-10-30-idle-diversifier"
ITERATION_DATE = "2026-10-30"
STUDY = "spy_upro_trend"
FAMILY = "idle_diversifier"

HYPOTHESIS = (
    "The live book leaves the SPY half invested after the 200-day band has moved "
    "the UPRO sleeve to cash. That is why the credit-spread hedge, the volatility "
    "scale, the multi-lookback vote, and the trailing-peak stop left the "
    "out-of-sample max drawdown near -46.8%: the trough is the equity bear after "
    "the sleeve is already in cash. This test does not try another faster exit "
    "for UPRO. Once a diversifier's own average exists, a day when SPY's 200-day "
    "2% band is off moves the whole book, core and sleeve, out of equities. The "
    "book holds the diversifier while that asset's filter is on, and T-bills "
    "while it is off. While SPY's band is on, the book is the unchanged trend "
    "core, including the three-down-close dip on the SPY half. Until the "
    "diversifier's average exists, the book stays that trend core, so the years "
    "before the ETF lists are not a separate cash-timing rule. Assets are GLD, "
    "IEF, and TLT, alphabetical, so a neighbor is one step on that list. Windows "
    "are 100 and 200 sessions. Bands are 0 and 2%, the same hysteresis as the "
    "equity filter, evaluated daily. A variant's walk-forward history starts the "
    "session after its filter first becomes ready. Earlier sessions are the trend "
    "core and do not count toward the 756-session minimum. Full-sample rows still "
    "include them. The walk-forward may also select the current book. The "
    "stitched path is kept only when it clears the materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_idle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(asset: str, window: int, band: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "symbol": asset,
        "window": int(window),
        "band": float(band),
        "apply": "next_open",
        "action": "risk_off_whole_book",
        "when": "equity_trend_off_and_filter_ready",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
        "rebalance": "quarterly",
        "history_starts": "session_after_filter_ready",
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
        note = "The prior trailing-drawdown iteration row was not in iterations.json, so the refresh has no comparison row."
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


def _simulate(study, bundle, trend, div_open, div_close, div_on, ready, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_idle_diversifier(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        div_open, div_close, trend, div_on, ready, dip, bundle["masks"]["quarterly"],
        0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
        div_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    return returns


def _walk_history(returns: pd.Series, ready: pd.Series) -> tuple[pd.Series, str]:
    """Drop sessions before the filter can change the book. Those days are the trend core."""
    prior = ready.reindex(returns.index).fillna(False).astype(bool).shift(1)
    prior = prior.fillna(False)
    if not bool(prior.any()):
        raise RuntimeError("diversifier filter never became ready")
    first = prior[prior].index[0]
    scored = returns.copy()
    scored.loc[scored.index < first] = float("nan")
    return scored, str(pd.Timestamp(first).date())


def _note(result: dict, kept: bool, last_pick: str, refresh: dict) -> str:
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
        f"Last walk-forward pick: {last_pick}."
    )
    if kept:
        ending = (
            "The stitched bond or gold risk-off rule clears the materiality bar. "
            "It becomes the live book."
        )
    else:
        ending = (
            "The stitched bond or gold risk-off rule does not clear the materiality bar. "
            "The live book stays the plain trend core."
        )
    return f"{stats} {ending}"


def _fmt_weights(weights: dict) -> str:
    order = ["SPY", "UPRO", "SSO", "BIL", "GLD", "IEF", "TLT"]
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


def _write_live_signal(bundle, frames) -> dict:
    as_of = bundle["spy_close"].index[-1]
    hedge_stress, hedge_rule, hedge_info = kept_hedge_stress(
        HERE / "iterations.json", bundle["spy_close"].index, as_of=as_of,
    )
    scale, vol_info = kept_sleeve_scale(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    dd_scale, dd_info = kept_drawdown_scale(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    if dd_scale is not None:
        scale, vol_info = dd_scale, None
    vote_trend, vote_info = kept_vote_trend(HERE / "iterations.json", bundle["spy_close"], as_of=as_of)
    closes = {symbol: frames[symbol]["close"] for symbol in ASSETS if symbol in frames}
    idle_kwargs = idle_signal_kwargs(HERE / "iterations.json", bundle["spy_close"], as_of, closes)
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=vote_trend, vote_info=vote_info,
        dd_info=dd_info,
        **idle_kwargs,
    )
    allowed = ALLOWED_TREND_TICKERS
    if idle_kwargs.get("idle_symbol"):
        allowed = allowed | frozenset({idle_kwargs["idle_symbol"]})
    write_signal(payload, HERE / "signals.json", allowed_tickers=allowed)
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if idle_kwargs.get("idle_info"):
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
    load_bars(
        YahooSource(),
        ["SPY", "SSO", "UPRO", "^IRX", *ASSETS],
        start="1993-01-01",
        refresh=True,
    )
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    frames = load_bars(YahooSource(), list(ASSETS), start="1993-01-01")
    trend = bundle["trends"][(200, 0.02, "daily")]
    sso = synth[2]
    upro = synth[3]
    baseline, _base_turn = hedge._simulate(study, bundle, trend, sample_index, sso, upro)
    ready_off = pd.Series(False, index=bundle["cal"])
    div_off = pd.Series(False, index=bundle["cal"])
    identity = _simulate(
        study, bundle, trend,
        bundle["spy_open"], bundle["spy_close"], div_off, ready_off,
        sample_index, sso, upro,
    )
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"an unready diversifier does not reproduce the trend core (max abs return gap {gap})")

    paths = {OFF_NAME: baseline}
    scored_paths = {OFF_NAME: baseline}
    coverage = []
    shares = {}
    history_start = {}
    prepared = {}
    for asset in ASSETS:
        div_open, div_close = align_ohlc(frames[asset]["open"], frames[asset]["close"], bundle["cal"])
        listed = frames[asset]["close"].dropna()
        coverage.append({
            "symbol": asset,
            "start": str(listed.index[0].date()),
            "end": str(listed.index[-1].date()),
            "n_obs": int(len(listed)),
        })
        prepared[asset] = (div_open, div_close)
    for asset, window, band in cells():
        div_open, div_close = prepared[asset]
        div_on, ready = idle_state(frames[asset]["close"], window, band, bundle["cal"])
        name = variant_name(asset, window, band)
        paths[name] = _simulate(
            study, bundle, trend, div_open, div_close, div_on, ready, sample_index, sso, upro,
        )
        scored_paths[name], history_start[name] = _walk_history(paths[name], ready)
        prior_off = ~trend.reindex(sample_index).shift(1).fillna(False).astype(bool)
        prior_ready = ready.reindex(sample_index).shift(1).fillna(False).astype(bool)
        prior_div = div_on.reindex(sample_index).shift(1).fillna(False).astype(bool)
        shares[name] = {
            "share_equity_off_and_ready": float((prior_off & prior_ready).mean()),
            "share_in_diversifier": float((prior_off & prior_ready & prior_div).mean()),
        }
    if max(float((paths[variant_name(asset, window, band)] - baseline).abs().max()) for asset, window, band in cells()) < 1e-12:
        raise RuntimeError("every diversifier cell reproduced the current book")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        scored_paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    rf = bundle["rf"]
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    for asset, window, band in cells():
        name = variant_name(asset, window, band)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(asset, window, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for asset, window, band in cells():
        name = variant_name(asset, window, band)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(asset, window, band),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(asset, window, band): (asset, window, band) for asset, window, band in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = {}
    else:
        cell_path = paths[last_pick]
        neighbor_paths = {
            variant_name(*cell): paths[variant_name(*cell)]
            for cell in adjacent_cells(by_name[last_pick], [ASSETS, WINDOWS, BANDS])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="idle_diversifier_walk_forward",
        params={
            "bar": "materiality_v1",
            "assets": list(ASSETS),
            "windows": list(WINDOWS),
            "bands": list(BANDS),
            "include_off": True,
            "score": "trailing_excess_sharpe",
            "first_year": study.WF_FIRST_YEAR,
            "min_obs": study.MIN_OBS,
            "history_starts": "session_after_filter_ready",
            "last_pick": last_pick,
        },
        sample_start=str(walk.returns.index[0].date()),
        sample_end=str(walk.returns.index[-1].date()),
        metrics={**result["oos"]["candidate"], "materiality_pass": kept},
        counts_for_deflation=False,
    )
    ledger.save()

    year_counts = {name: 0 for name in paths}
    for pick in walk.picks.values():
        year_counts[pick] = year_counts.get(pick, 0) + 1
    base_summary = summarize(baseline, rf=rf.reindex(baseline.index))
    variants = [{
        "name": OFF_NAME,
        "counts_for_deflation": False,
        "cagr": base_summary["cagr"],
        "sharpe": base_summary["sharpe"],
        "max_drawdown": base_summary["max_drawdown"],
        "years_selected": year_counts.get(OFF_NAME, 0),
        "note": "The current book. Already counted as dip_3_down_closes.",
    }]
    for asset, window, band in cells():
        name = variant_name(asset, window, band)
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": True,
            "symbol": asset,
            "window": window,
            "band": band,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "history_start": history_start[name],
            **shares[name],
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Risk-off bond or gold trend",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "idle_diversifier",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh),
        "data": {
            "assets": list(ASSETS),
            "windows": list(WINDOWS),
            "bands": list(BANDS),
            "coverage": coverage,
            "price_sample_end": sample_end,
            "history_start": history_start,
            "shares": shares,
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
    signal = _write_live_signal(bundle, frames)
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
    again = kept_idle_choice(HERE / "iterations.json", bundle["spy_close"].index[-1])
    if kept and last_pick != OFF_NAME:
        if again is None:
            raise RuntimeError("a kept diversifier did not reload from the iteration log")
    elif again is not None:
        raise RuntimeError("a rejected diversifier reloaded as if it were kept")
    print(json.dumps({
        "decision": entry["decision"],
        "kept": kept,
        "last_pick": last_pick,
        "n_trials": n_trials,
        "dispersion": dispersion,
        "oos": result["oos"],
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
        "history_start": history_start,
        "shares": shares,
        "price_sample_end": sample_end,
        "checks": result["checks"],
        "signal_sentence": _signal_sentence(previous_signal, signal),
        "signal": entry["signal"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
