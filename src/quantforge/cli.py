"""Command line entry points: fetch, signals, study, ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quantforge import __version__
from quantforge.data import get_source, load_bars
from quantforge.data.base import DataError
from quantforge.eval.ledger import TrialLedger
from quantforge.signals.emit import write_signal
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.engine.leverage import FINANCING_SPREAD, LEVERAGED_ETFS, synthesize_leveraged
from quantforge.strategies.dipbuy import build_dip_signal
from quantforge.strategies.confirm import kept_confirm_trend
from quantforge.strategies.corehold import kept_core_hold
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.drawdown import kept_drawdown_scale
from quantforge.strategies.drift import has_kept_drift, kept_drift_mix, lagged_tbill
from quantforge.strategies.reentry import kept_fast_reentry
from quantforge.strategies.ema import kept_ema_trend
from quantforge.strategies.sleevegear import kept_sleeve_gear
from quantforge.strategies.idle import kept_idle_choice, idle_signal_kwargs
from quantforge.strategies.sleevediv import kept_sleeve_cash_choice, sleeve_cash_signal_kwargs
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
from quantforge.strategies.billgear import align_bill_yield, has_kept_bill, kept_bill_gear
from quantforge.strategies.pause import has_kept_streak_pause, kept_streak_pause
from quantforge.strategies.cushion import kept_cushion
from quantforge.strategies.volgear import kept_vol_gear
from quantforge.strategies.volscale import kept_sleeve_scale
from quantforge.strategies.vote import kept_vote_trend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantforge", description="Walk-forward backtests and live target weights.")
    parser.add_argument("--version", action="version", version=f"quantforge {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="download bars into the local parquet cache")
    fetch.add_argument("source", choices=["yahoo", "coinbase"])
    fetch.add_argument("symbols", nargs="+")
    fetch.add_argument("--start", default="1998-01-01")
    fetch.add_argument("--end", default=None)
    fetch.add_argument("--interval", default="1d")
    fetch.add_argument("--cache", default=None)
    fetch.add_argument("--refresh", action="store_true")

    signals = sub.add_parser("signals", help="write target weights for the next NYSE session")
    signals.add_argument("--strategy", default="trend-core", choices=["trend-core", "flagship", "dip-buy", "apex"])
    signals.add_argument("--out", default=None)
    signals.add_argument("--cache", default=None)
    signals.add_argument("--start", default="1998-01-01")
    signals.add_argument("--end", default=None)
    signals.add_argument("--study-dir", default="studies/spy_upro_trend")

    study = sub.add_parser("study", help="run a study end to end")
    study.add_argument("name", choices=["spy-upro-trend", "flagship", "apex"])

    ledger = sub.add_parser("ledger", help="summarize the multiple-testing ledger")
    ledger.add_argument("--path", default="studies/trial_ledger.json")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "fetch":
            return _fetch(args)
        if args.cmd == "signals":
            return _signals(args)
        if args.cmd == "study":
            return _study(args)
        if args.cmd == "ledger":
            return _ledger(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _fetch(args) -> int:
    source = get_source(args.source)
    frames = load_bars(
        source,
        args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        cache_dir=args.cache,
        refresh=args.refresh,
    )
    for symbol, frame in frames.items():
        print(f"{symbol} {len(frame)} bars {frame.index[0].date()} -> {frame.index[-1].date()}")
    return 0


def _signals(args) -> int:
    if args.strategy == "flagship":
        return _flagship_from_study(args)
    if args.strategy == "apex":
        return _apex_from_study(args)
    cache = args.cache
    frames = load_bars(get_source("yahoo"), ["SPY"], start=args.start, end=args.end, cache_dir=cache)
    close = frames["SPY"]["close"]
    study = Path(args.study_dir)
    if args.strategy == "trend-core":
        stress, rule, info = kept_hedge_stress(study / "iterations.json", close.index)
        scale, vol_info = kept_sleeve_scale(study / "iterations.json", close, as_of=close.index[-1])
        dd_scale, dd_info = kept_drawdown_scale(study / "iterations.json", close, as_of=close.index[-1])
        if dd_scale is not None:
            scale, vol_info = dd_scale, None
        vote_trend, vote_info = kept_vote_trend(study / "iterations.json", close, as_of=close.index[-1])
        confirm_trend, confirm_info = kept_confirm_trend(study / "iterations.json", close, as_of=close.index[-1])
        if confirm_trend is not None and vote_trend is not None:
            raise DataError("the iteration log kept both a vote and a confirmation rule")
        if confirm_trend is not None:
            vote_trend, vote_info = confirm_trend, None
        ema_trend, ema_info = kept_ema_trend(study / "iterations.json", close, as_of=close.index[-1])
        if ema_trend is not None and vote_trend is not None:
            raise DataError("the iteration log kept both an exponential filter and another trend override")
        if ema_trend is not None:
            vote_trend = ema_trend
        idle_choice = kept_idle_choice(study / "iterations.json", close.index[-1])
        idle_closes = {}
        if idle_choice is not None:
            idle_frames = load_bars(
                get_source("yahoo"), [idle_choice["symbol"]],
                start=args.start, end=args.end, cache_dir=cache,
            )
            idle_closes[idle_choice["symbol"]] = idle_frames[idle_choice["symbol"]]["close"]
        idle_kwargs = idle_signal_kwargs(
            study / "iterations.json", close, close.index[-1], idle_closes,
        )
        sleeve_cash_choice = kept_sleeve_cash_choice(study / "iterations.json", close.index[-1])
        sleeve_cash_closes = {}
        if sleeve_cash_choice is not None:
            if idle_choice is not None:
                raise DataError("the iteration log kept both a whole-book diversifier and a sleeve diversifier")
            sleeve_frames = load_bars(
                get_source("yahoo"), [sleeve_cash_choice["symbol"]],
                start=args.start, end=args.end, cache_dir=cache,
            )
            sleeve_cash_closes[sleeve_cash_choice["symbol"]] = sleeve_frames[sleeve_cash_choice["symbol"]]["close"]
        sleeve_cash_kwargs = sleeve_cash_signal_kwargs(
            study / "iterations.json", close, close.index[-1], sleeve_cash_closes,
        )
        core_hold, core_hold_info = kept_core_hold(study / "iterations.json", close, as_of=close.index[-1])
        reentry_hold, reentry_info = kept_fast_reentry(study / "iterations.json", close, as_of=close.index[-1])
        if reentry_hold is not None and core_hold is not None:
            raise DataError("the iteration log kept both an off-trend core hold and a fast re-entry")
        if reentry_hold is not None:
            core_hold, core_hold_info = reentry_hold, reentry_info
        sleeve_fast, gear_info = kept_sleeve_gear(study / "iterations.json", close, as_of=close.index[-1])
        sleeve_upro_share, vol_gear_info = kept_vol_gear(study / "iterations.json", close, as_of=close.index[-1])
        if sleeve_upro_share is not None and sleeve_fast is not None:
            raise DataError("the iteration log kept both a vol gear and a sleeve gear")
        if sleeve_upro_share is not None and scale is not None:
            raise DataError("the iteration log kept both a vol gear and a sleeve scale")
        cushion_share, cushion_info = kept_cushion(study / "iterations.json", close, as_of=close.index[-1])
        if cushion_share is not None and sleeve_upro_share is not None:
            raise DataError("the iteration log kept both a cushion scale and a vol gear")
        if cushion_share is not None and sleeve_fast is not None:
            raise DataError("the iteration log kept both a cushion scale and a sleeve gear")
        if cushion_share is not None and scale is not None:
            raise DataError("the iteration log kept both a cushion scale and a sleeve scale")
        bill_share = None
        bill_info = None
        if has_kept_bill(study / "iterations.json"):
            irx_frames = load_bars(
                get_source("yahoo"), ["^IRX"],
                start=args.start, end=args.end, cache_dir=cache,
            )
            bill_share, bill_info = kept_bill_gear(
                study / "iterations.json",
                align_bill_yield(irx_frames["^IRX"]["close"], close.index),
                as_of=close.index[-1],
            )
        if bill_share is not None and cushion_share is not None:
            raise DataError("the iteration log kept both a bill-yield gear and a cushion scale")
        if bill_share is not None and sleeve_upro_share is not None:
            raise DataError("the iteration log kept both a bill-yield gear and a vol gear")
        if bill_share is not None and sleeve_fast is not None:
            raise DataError("the iteration log kept both a bill-yield gear and a sleeve gear")
        if bill_share is not None and scale is not None:
            raise DataError("the iteration log kept both a bill-yield gear and a sleeve scale")
        pause_share = None
        pause_info = None
        if has_kept_streak_pause(study / "iterations.json"):
            pause_share, pause_info = kept_streak_pause(
                study / "iterations.json", close, as_of=close.index[-1],
            )
        if pause_share is not None and bill_share is not None:
            raise DataError("the iteration log kept both a down-streak pause and a bill-yield gear")
        if pause_share is not None and cushion_share is not None:
            raise DataError("the iteration log kept both a down-streak pause and a cushion scale")
        if pause_share is not None and sleeve_upro_share is not None:
            raise DataError("the iteration log kept both a down-streak pause and a vol gear")
        if pause_share is not None and sleeve_fast is not None:
            raise DataError("the iteration log kept both a down-streak pause and a sleeve gear")
        if pause_share is not None and scale is not None:
            raise DataError("the iteration log kept both a down-streak pause and a sleeve scale")
        drift_kwargs = {}
        if has_kept_drift(study / "iterations.json"):
            names = ["SPY", "SSO", "UPRO", "^IRX"]
            loaded = load_bars(
                get_source("yahoo"), names,
                start=args.start, end=args.end, cache_dir=cache,
            )
            spy_frame = loaded["SPY"]
            irx = lagged_tbill(loaded["^IRX"]["close"], spy_frame.index)
            synth = {}
            for symbol, leverage in (("SSO", 2), ("UPRO", 3)):
                meta = LEVERAGED_ETFS[symbol]
                synth[leverage] = synthesize_leveraged(
                    spy_frame["open"], spy_frame["close"], irx,
                    leverage, meta["expense_ratio"], FINANCING_SPREAD,
                )
            drift_mix, drift_info = kept_drift_mix(
                study / "iterations.json",
                spy_frame["open"], spy_frame["close"],
                synth[2]["open"], synth[2]["close"],
                synth[3]["open"], synth[3]["close"],
                irx, as_of=spy_frame.index[-1],
            )
            drift_kwargs = {"drift_mix": drift_mix, "drift_info": drift_info}
        payload = build_trend_signal(
            TrendCoreSpec(), close, strategy_id="trend_core",
            hedge_on=stress, hedge_rule=rule, hedge_info=info,
            sleeve_scale=scale, vol_info=vol_info,
            trend_override=vote_trend, vote_info=vote_info,
            confirm_info=confirm_info,
            dd_info=dd_info,
            core_hold=core_hold, core_hold_info=core_hold_info,
            sleeve_fast=sleeve_fast, sleeve_gear_info=gear_info,
            sleeve_upro_share=sleeve_upro_share, vol_gear_info=vol_gear_info,
            cushion_share=cushion_share, cushion_info=cushion_info,
            bill_share=bill_share, bill_info=bill_info,
            pause_share=pause_share, pause_info=pause_info,
            ema_info=ema_info,
            **idle_kwargs,
            **drift_kwargs,
            **sleeve_cash_kwargs,
        )
        dest = Path(args.out) if args.out else study / "signals.json"
        allowed = ALLOWED_TREND_TICKERS
        if idle_kwargs.get("idle_symbol"):
            allowed = allowed | frozenset({idle_kwargs["idle_symbol"]})
        if sleeve_cash_kwargs.get("sleeve_cash_symbol"):
            allowed = allowed | frozenset({sleeve_cash_kwargs["sleeve_cash_symbol"]})
        write_signal(payload, dest, allowed_tickers=allowed)
        print(f"wrote {dest} for {payload['date']} {payload['weights']}")
        return 0
    else:
        payload = build_dip_signal(close, strategy_id="dip_buy")
        dest = Path(args.out) if args.out else study / "signals_dipbuy.json"
    write_signal(payload, dest)
    print(f"wrote {dest} for {payload['date']} {payload['weights']}")
    return 0


def _flagship_from_study(args) -> int:
    """Replay the recommended spec stored by the study. Does not refit it."""
    study = Path(args.study_dir)
    summary_path = study / "results.json"
    flagship_summary = Path("studies/flagship/results.json")
    if args.strategy == "flagship" and flagship_summary.exists():
        summary_path = flagship_summary
    if not summary_path.exists():
        raise DataError(f"no {summary_path}; run `quantforge study spy-upro-trend` first")
    summary = json.loads(summary_path.read_text())
    if summary.get("live_signal"):
        payload = summary["live_signal"]
        dest = Path(args.out) if args.out else Path(args.study_dir) / "signals_flagship.json"
        negative = any(value < 0 for value in payload.get("weights", {}).values())
        gross = max(1.0, float(payload.get("gross") or 1.0))
        write_signal(payload, dest, allowed_tickers=None, max_gross=gross, allow_negative=negative)
        print(f"wrote {dest} for {payload['date']} {payload['weights']}")
        return 0
    params = summary.get("recommendation", {}).get("signal_params")
    if not params:
        raise DataError("results.json has no recommendation.signal_params or live_signal")
    frames = load_bars(get_source("yahoo"), ["SPY"], start=args.start, end=args.end, cache_dir=args.cache)
    spec = TrendCoreSpec(
        core_weight=params["core_weight"],
        leverage=params["leverage"],
        window=params["window"],
        band=params["band"],
        freq=params["freq"],
        rebalance=params["rebalance"],
        sleeve_weight=params["sleeve_weight"],
        cash_symbol=params.get("cash_symbol", "BIL"),
        core_symbol=params.get("core_symbol", "SPY"),
    )
    payload = build_trend_signal(spec, frames["SPY"]["close"], strategy_id="flagship")
    payload["endorsed"] = bool(summary["recommendation"].get("edge_survives"))
    payload["recommendation"] = summary["recommendation"].get("action")
    dest = Path(args.out) if args.out else study / "signals_flagship.json"
    write_signal(payload, dest)
    print(f"wrote {dest} for {payload['date']} {payload['weights']}")
    return 0


def _apex_from_study(args) -> int:
    """Replay the apex signal stored by the study. Does not refit it."""
    summary_path = Path("studies/apex/results.json")
    if not summary_path.exists():
        raise DataError(f"no {summary_path}; run `quantforge study apex` first")
    summary = json.loads(summary_path.read_text())
    payload = summary.get("live_signal")
    if not payload:
        raise DataError("apex results.json has no live_signal")
    dest = Path(args.out) if args.out else Path("studies/apex/signals.json")
    negative = any(value < 0 for value in payload.get("weights", {}).values())
    gross = max(1.0, float(payload.get("gross") or 1.0))
    write_signal(payload, dest, allowed_tickers=None, max_gross=gross, allow_negative=negative)
    print(f"wrote {dest} for {payload['date']} {payload['weights']}")
    return 0


def _study(args) -> int:
    root = _repo_root()
    folder = {"spy-upro-trend": "spy_upro_trend", "flagship": "flagship", "apex": "apex"}[args.name]
    script = root / "studies" / folder / "run.py"
    if not script.exists():
        raise DataError(f"study script not found at {script}")
    import runpy

    sys.path.insert(0, str(root / "src"))
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _ledger(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"no ledger at {path}")
        return 0
    book = TrialLedger(path)
    print(f"trials {book.n_trials()} counting / {len(book.trials)} stored  dispersion {book.sharpe_dispersion():.6f}")
    for trial in sorted(book.counting(), key=lambda item: item["metrics"].get("sharpe", 0), reverse=True)[:15]:
        metrics = trial["metrics"]
        print(f"  {trial['name']}  sharpe {metrics.get('sharpe', float('nan')):.2f}  cagr {metrics.get('cagr', float('nan')):.3f}")
    return 0


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        if (parent / "studies" / "spy_upro_trend" / "run.py").exists():
            return parent
    return Path.cwd()


if __name__ == "__main__":
    raise SystemExit(main())
