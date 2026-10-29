#!/usr/bin/env python3
"""One walk-forward test: cut the UPRO sleeve after a trailing-peak drawdown.

The windows, the cuts, the re-entry fraction, and the keep rule are fixed in
``quantforge.strategies.drawdown`` and ``quantforge.eval.materiality``.
Do not add a cell after seeing this sample.

The 200-day band leaves the 3x sleeve on through the first part of a crash.
Volatility scaling cut that sleeve in high-vol rallies as well as in breaks,
and the multi-lookback vote was never selected. A trailing peak cuts the
sleeve only after SPY has fallen a set amount from its own high, and it puts
the sleeve back on once half of that drawdown has healed and the trend is
still on. The SPY core, the three-down-close dip, and the quarterly mix are
unchanged.

Run from the repo root:

    python studies/spy_upro_trend/drawdown_stop.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from quantforge.engine.backtest import run_vol_sleeve
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.drawdown import (
    CUTS,
    OFF_NAME,
    REENTRY_FRACTION,
    WINDOWS,
    cells,
    kept_drawdown_scale,
    sleeve_allowed,
    sleeve_scale,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
PRIOR_ID = "2026-10-27-multi-lookback-vote"
ITERATION_ID = "2026-10-29-trailing-drawdown-stop"
ITERATION_DATE = "2026-10-29"
STUDY = "spy_upro_trend"
FAMILY = "drawdown_stop"

HYPOTHESIS = (
    "The live sleeve stays on until SPY closes through its 200-day band. That lag "
    "is why the trend-core drawdown is still about -46.8%, and why the book trailed "
    "SPY in 2020 and 2022. The 2026-10-26 volatility scale cut the sleeve whenever "
    "realized volatility was high, including in high-vol rallies, and gained 2.6 "
    "drawdown points, short of 3, with a negative improvement Sharpe. The 2026-10-27 "
    "multi-lookback vote was never selected, so the stitched path matched the current "
    "book, and the vote cells themselves had deeper drawdowns. A trailing-peak stop "
    "cuts only the UPRO sleeve after SPY has already fallen from its own high, so a "
    "high-vol rally that is still near that high stays invested. The peak is the "
    "maximum close over 63, 126, or 252 sessions. The sleeve goes to T-bills when the "
    "close is 8%, 12%, or 16% below that peak, and it returns when the drawdown has "
    "healed to half the cut and the 200-day trend is still on. The half-cut re-entry "
    "is fixed, not a searched axis. The single-window grid already contains SMA 100, "
    "150, and 200; this stop does not replace that filter. The SPY core, the "
    "three-down-close dip, and the quarterly mix are unchanged. The walk-forward may "
    "also select the current book. The stitched path is kept only when it clears the "
    "materiality bar."
)


def _load_hedge():
    path = HERE / "credit_hedge.py"
    spec = importlib.util.spec_from_file_location("credit_hedge_for_drawdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cell_params(window: int, cut: float) -> dict:
    return {
        "base": "dip_3_down_closes",
        "peak_window": int(window),
        "cut": float(cut),
        "reentry_fraction": REENTRY_FRACTION,
        "peak": "trailing_max_close",
        "apply": "next_open",
        "action": "cut_upro_sleeve_to_tbill",
        "core_weight": 0.5,
        "sleeve_weight": 0.5,
        "dip_rule": "3_down_closes",
        "trend": "sma200_band2_daily",
        "rebalance": "quarterly",
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
        note = "The prior vote iteration row was not in iterations.json, so the refresh has no comparison row."
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


def _scaled(study, bundle, trend, scale, index, sso, upro):
    dip = dip_active(bundle["spy_close"], "3_down_closes").reindex(bundle["cal"]).fillna(False)
    result = run_vol_sleeve(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], scale,
        0.5, 0.5, bundle["rf"],
        spy_cost=study.BASE_COST, sso_cost=study.BASE_COST, upro_cost=study.BASE_COST,
    )
    returns = study._window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    return returns, turnover


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
        ending = "The stitched trailing-peak stop clears the materiality bar. It becomes the live sleeve rule."
    else:
        ending = (
            "The stitched trailing-peak stop does not clear the materiality bar. "
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
    payload = build_trend_signal(
        TrendCoreSpec(), bundle["spy_close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=scale, vol_info=vol_info,
        trend_override=vote_trend, vote_info=vote_info,
        dd_info=dd_info,
    )
    write_signal(payload, HERE / "signals.json")
    title = "Trend-core (2% band, dip overlay on the SPY half)"
    if dd_info:
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
    hedge = _load_hedge()
    study = hedge._load_study()
    bundle, synth, sample_index = hedge._prepare_bundle(study)
    trend = bundle["trends"][(200, 0.02, "daily")]
    sso = synth[2]
    upro = synth[3]
    baseline, _base_turn = hedge._simulate(study, bundle, trend, sample_index, sso, upro)
    ones = pd.Series(1.0, index=bundle["cal"])
    identity, _id_turn = _scaled(study, bundle, trend, ones, sample_index, sso, upro)
    gap = float((identity - baseline).abs().max())
    if gap > 1e-8:
        raise RuntimeError(f"a unit scale does not reproduce the trend core (max abs return gap {gap})")

    paths = {OFF_NAME: baseline}
    cut_share = {}
    for window, cut in cells():
        allowed = sleeve_allowed(bundle["spy_close"], window, cut).reindex(bundle["cal"]).fillna(True)
        scale = sleeve_scale(bundle["spy_close"], window, cut).reindex(bundle["cal"]).fillna(1.0)
        name = variant_name(window, cut)
        paths[name], _turn = _scaled(study, bundle, trend, scale, sample_index, sso, upro)
        trend_on = trend.reindex(sample_index).fillna(False).astype(bool)
        stopped = ~allowed.reindex(sample_index).fillna(True).astype(bool)
        cut_share[name] = float((trend_on & stopped).mean())
    if max(float((paths[variant_name(window, cut)] - baseline).abs().max()) for window, cut in cells()) < 1e-12:
        raise RuntimeError("every drawdown cell reproduced the current book; the stop never changed a return")

    decisions = study._year_starts(sample_index, study.WF_FIRST_YEAR)
    walk = study.walk_forward_select(
        paths, decisions, study._score_factory(bundle["rf"]), min_obs=study.MIN_OBS,
    )
    current_oos = baseline.reindex(walk.returns.index)
    if current_oos.isna().any() or not current_oos.index.equals(walk.returns.index):
        raise RuntimeError("the current book is missing walk-forward sessions")

    spy_r = study._window(bundle["spy_close"].pct_change(), sample_index)
    rf = bundle["rf"]
    ledger = TrialLedger(LEDGER_PATH)
    sample_end = str(sample_index[-1].date())
    full = {}
    for window, cut in cells():
        name = variant_name(window, cut)
        full[name] = summarize(
            paths[name], rf=rf.reindex(paths[name].index), benchmark=spy_r.reindex(paths[name].index),
        )
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(window, cut),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=full[name], counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for window, cut in cells():
        name = variant_name(window, cut)
        scored = dict(full[name])
        window_rf = rf.reindex(paths[name].index)
        scored["psr"] = probabilistic_sharpe(paths[name], rf=window_rf)
        scored["dsr"] = deflated_sharpe(paths[name], n_trials, dispersion, rf=window_rf)
        full[name] = scored
        ledger.record(
            study=STUDY, family=FAMILY, name=name, params=_cell_params(window, cut),
            sample_start=study.SAMPLE_START, sample_end=sample_end,
            metrics=scored, counts_for_deflation=True,
        )

    by_name = {variant_name(window, cut): (window, cut) for window, cut in cells()}
    last_pick = list(walk.picks.values())[-1]
    if last_pick == OFF_NAME:
        cell_path = None
        neighbor_paths = {}
    else:
        cell_path = paths[last_pick]
        neighbor_paths = {
            variant_name(*cell): paths[variant_name(*cell)]
            for cell in adjacent_cells(by_name[last_pick], [WINDOWS, CUTS])
        }
    result = evaluate_materiality(
        walk.returns, current_oos, neighbor_paths, cell_path,
        n_trials, dispersion, rf=rf,
    )
    kept = bool(result["pass"])
    ledger.record(
        study=STUDY, family="audit", name="drawdown_stop_walk_forward",
        params={
            "bar": "materiality_v1",
            "windows": list(WINDOWS),
            "cuts": list(CUTS),
            "reentry_fraction": REENTRY_FRACTION,
            "include_off": True,
            "score": "trailing_excess_sharpe",
            "first_year": study.WF_FIRST_YEAR,
            "min_obs": study.MIN_OBS,
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
    for window, cut in cells():
        name = variant_name(window, cut)
        row = full[name]
        variants.append({
            "name": name,
            "counts_for_deflation": True,
            "window": window,
            "cut": cut,
            "reentry_fraction": REENTRY_FRACTION,
            "cagr": row.get("cagr"),
            "sharpe": row.get("sharpe"),
            "max_drawdown": row.get("max_drawdown"),
            "dsr": row.get("dsr"),
            "years_selected": year_counts.get(name, 0),
            "share_trend_on_and_stopped": cut_share[name],
        })

    previous_signal = _committed_signal()
    refresh = _lead_refresh(result["oos"]["current"], baseline.reindex(walk.returns.index), rf)
    entry = {
        "id": ITERATION_ID,
        "date": ITERATION_DATE,
        "title": "Trailing-peak stop on the UPRO sleeve",
        "hypothesis": HYPOTHESIS,
        "status": "tested",
        "kept": kept,
        "overlay": "drawdown_stop",
        "decision": "kept" if kept else "rejected",
        "decision_note": _note(result, kept, last_pick, refresh),
        "data": {
            "windows": list(WINDOWS),
            "cuts": list(CUTS),
            "reentry_fraction": REENTRY_FRACTION,
            "price_sample_end": sample_end,
            "share_trend_on_and_stopped": cut_share,
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
        "trade_reason": signal.get("trade_reason"),
    }
    hedge._write_outputs(entry)
    again, again_info = kept_drawdown_scale(
        HERE / "iterations.json", bundle["spy_close"], as_of=bundle["spy_close"].index[-1],
    )
    if kept and last_pick != OFF_NAME:
        if again is None or again_info is None:
            raise RuntimeError("a kept stop did not reload from the iteration log")
    elif again is not None:
        raise RuntimeError("a rejected stop reloaded as if it were kept")
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
        "share_trend_on_and_stopped": cut_share,
        "price_sample_end": sample_end,
        "checks": result["checks"],
        "signal_sentence": _signal_sentence(previous_signal, signal),
        "signal": entry["signal"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
