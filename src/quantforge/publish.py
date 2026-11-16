"""Refresh prices and write the paper-trading signals.

Run from the repo root:

    python -m quantforge.publish --out signals/

The lead file keeps the object the trend bot already reads: ``date``,
``weights``, and ``diagnostics``. The other files are one strategy each.
``signals.json`` in the output directory is the combined object. Weights are
recomputed from the stored walk-forward cell and the latest close. The cell
is not refit here. Re-run ``studies/spy_upro_trend/faster.py`` when a new
calendar year starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from quantforge.cli import _signals
from quantforge.data import CoinbaseSource, YahooSource, load_bars
from quantforge.data.base import DataError
from quantforge.signals.emit import write_signal
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal
from quantforge.strategies.fastsignal import boost_signal, btc_signal, meanrev_signal, nasdaq_signal, rotation_signal
from quantforge.strategies.panel import build_panel

YAHOO_SYMBOLS = [
    "SPY", "SSO", "UPRO", "QQQ", "QLD", "TQQQ", "TLT", "TMF", "GLD", "UGL", "BIL", "^IRX",
]
FAST_TICKERS = frozenset({
    "SPY", "SSO", "UPRO", "QQQ", "QLD", "TQQQ", "TLT", "TMF", "GLD", "UGL", "BIL", "BTC-USD",
})


def load_spec(path: Path) -> dict:
    if not path.exists():
        raise DataError(f"no {path}; run python studies/spy_upro_trend/faster.py first")
    return json.loads(path.read_text())


def annotate_lead(payload: dict, spec: dict) -> dict:
    """Add the publisher fields without removing the keys the trend bot reads."""
    lead = spec.get("lead") or {}
    out = dict(payload)
    diagnostics = dict(out.get("diagnostics") or {})
    sma = diagnostics.get("sma")
    band = diagnostics.get("band", 0.02)
    close_px = diagnostics.get("reference_close")
    levels = {}
    if sma is not None and close_px is not None:
        levels = {
            "entry": f"SPY close above {float(sma) * (1.0 + float(band)):.4f}",
            "entry_level": float(sma) * (1.0 + float(band)),
            "entry_value": float(close_px),
            "exit": f"SPY close below {float(sma) * (1.0 - float(band)):.4f}",
            "exit_level": float(sma) * (1.0 - float(band)),
            "stop": "no trailing stop; the SPY half switches to SSO after three down closes",
            "stop_level": None,
            "reference_close": float(close_px),
        }
    out["strategy_id"] = "trend-core"
    out["as_of"] = out.get("signal_date")
    out["levels"] = levels
    out["validated"] = bool(lead.get("validated", True))
    out["validated_reason"] = lead.get("validated_reason", "approved_live_book")
    out["oos"] = lead.get("oos") or {}
    return out


def build_fast_signals(panel: dict, spec: dict) -> dict[str, dict]:
    """Recompute each stored cell on the panel. Does not walk-forward again."""
    strategies = spec.get("strategies") or {}
    found = {}
    spy = panel["spy"]
    if "meanrev" in strategies:
        cell = strategies["meanrev"]["cell"]
        payload = meanrev_signal(spy, cell)
        found["meanrev"] = _decorate(payload, strategies["meanrev"])
    if "rotation" in strategies:
        closes = pd.DataFrame({
            symbol: panel["stitched"][symbol]["close"]
            for symbol in ("UPRO", "TQQQ", "TMF", "UGL")
        })
        cell = strategies["rotation"]["cell"]
        payload = rotation_signal(closes, panel["rf"], cell)
        found["rotation"] = _decorate(payload, strategies["rotation"])
    if "nasdaq-vol" in strategies:
        cell = strategies["nasdaq-vol"]["cell"]
        payload = nasdaq_signal(
            panel["under"]["QQQ"]["close"],
            panel["stitched"]["TQQQ"]["close"],
            cell,
        )
        found["nasdaq-vol"] = _decorate(payload, strategies["nasdaq-vol"])
    if "spy-rsi-boost" in strategies:
        cell = strategies["spy-rsi-boost"]["cell"]
        payload = boost_signal(spy["close"], cell)
        found["spy-rsi-boost"] = _decorate(payload, strategies["spy-rsi-boost"])
    if "btc-trend" in strategies:
        if panel.get("btc") is None or panel["btc"].dropna().empty:
            raise DataError("the spec includes btc-trend but the panel has no BTC-USD closes")
        cell = strategies["btc-trend"]["cell"]
        close = panel["btc"]["close"] if isinstance(panel["btc"], pd.DataFrame) else panel["btc"]
        payload = btc_signal(close, cell)
        found["btc-trend"] = _decorate(payload, strategies["btc-trend"])
    return found


def _decorate(payload: dict, row: dict) -> dict:
    out = dict(payload)
    out["validated"] = bool(row.get("validated", False))
    out["validated_reason"] = row.get("validated_reason", "materiality_bar")
    out["oos"] = row.get("oos") or {}
    out["cell"] = row.get("cell")
    return out


def check_selection_year(spec: dict, as_of: str) -> None:
    year = spec.get("selection_year")
    if year is None:
        raise DataError("the spec has no selection_year")
    close_year = int(str(as_of)[:4])
    if close_year > int(year):
        raise DataError(
            f"the latest close {as_of} is in {close_year}, after the stored "
            f"{year} walk-forward pick. Re-run studies/spy_upro_trend/faster.py "
            "before publishing."
        )


def write_bundle(out_dir: Path, lead: dict, fast: dict[str, dict]) -> dict:
    """Write ``trend-core.json``, one file per faster book, and ``signals.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_signal(lead, allowed_tickers=ALLOWED_TREND_TICKERS)
    write_signal(lead, out_dir / "trend-core.json", allowed_tickers=ALLOWED_TREND_TICKERS)
    for name, payload in fast.items():
        validate_signal(payload, allowed_tickers=FAST_TICKERS)
        write_signal(payload, out_dir / f"{name}.json", allowed_tickers=FAST_TICKERS)
    combined = {
        "date": lead["date"],
        "as_of": lead.get("as_of") or lead.get("signal_date"),
        "strategies": {"trend-core": lead, **fast},
    }
    dest = out_dir / "signals.json"
    dest.write_text(json.dumps(combined, indent=2) + "\n")
    return combined


def _lead_payload(study_dir: Path, cache: str | None, dest: Path) -> dict:
    args = argparse.Namespace(
        cmd="signals",
        strategy="trend-core",
        out=str(dest),
        cache=cache,
        start="1993-01-01",
        end=None,
        study_dir=str(study_dir),
    )
    code = _signals(args)
    if code != 0:
        raise DataError("the trend-core signal builder failed")
    return json.loads(dest.read_text())


def publish(out_dir: Path, spec_path: Path, study_dir: Path, cache: str | None = None) -> dict:
    spec = load_spec(spec_path)
    frames = load_bars(YahooSource(), YAHOO_SYMBOLS, start="1993-01-01", cache_dir=cache, refresh=True)
    if (spec.get("strategies") or {}).get("btc-trend"):
        try:
            crypto = load_bars(
                CoinbaseSource(), ["BTC-USD"], start="2015-01-01", cache_dir=cache, refresh=True,
            )
        except DataError as exc:
            raise DataError(f"BTC-USD is in the spec but Coinbase did not refresh: {exc}") from exc
        frames["BTC-USD"] = crypto["BTC-USD"]
    panel = build_panel(frames)
    as_of = str(pd.Timestamp(panel["clock"][-1]).date())
    check_selection_year(spec, as_of)
    lead_raw = _lead_payload(study_dir, cache, study_dir / "signals.json")
    if lead_raw.get("signal_date") != as_of:
        raise DataError(
            f"trend-core signal close {lead_raw.get('signal_date')} does not match the panel close {as_of}"
        )
    lead = annotate_lead(lead_raw, spec)
    fast = build_fast_signals(panel, spec)
    # The canonical trend file stays the object the bot reads, including the extra keys.
    write_signal(lead, study_dir / "signals.json", allowed_tickers=ALLOWED_TREND_TICKERS)
    return write_bundle(out_dir, lead, fast)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write paper-trading signals from the latest close.")
    parser.add_argument("--out", default="signals", help="directory for one JSON per strategy")
    parser.add_argument("--spec", default="studies/faster/live.json")
    parser.add_argument("--study-dir", default="studies/spy_upro_trend")
    parser.add_argument("--cache", default=None)
    args = parser.parse_args(argv)
    try:
        combined = publish(Path(args.out), Path(args.spec), Path(args.study_dir), cache=args.cache)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    as_of = combined.get("as_of")
    session = combined.get("date")
    print(f"wrote {args.out} for the {session} open from the {as_of} close")
    for name, payload in combined["strategies"].items():
        weights = ", ".join(f"{ticker} {value:.4f}" for ticker, value in sorted(payload["weights"].items()))
        flag = "validated" if payload.get("validated") else "not validated"
        print(f"  {name}: {weights} ({flag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
