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
from quantforge.strategies.dipbuy import build_dip_signal
from quantforge.strategies.credit import kept_hedge_stress
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal
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
        vote_trend, vote_info = kept_vote_trend(study / "iterations.json", close, as_of=close.index[-1])
        payload = build_trend_signal(
            TrendCoreSpec(), close, strategy_id="trend_core",
            hedge_on=stress, hedge_rule=rule, hedge_info=info,
            sleeve_scale=scale, vol_info=vol_info,
            trend_override=vote_trend, vote_info=vote_info,
        )
        dest = Path(args.out) if args.out else study / "signals.json"
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
