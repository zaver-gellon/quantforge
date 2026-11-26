"""Refresh prices and write the paper-trading signals.

Run from the repo root:

    python -m quantforge.publish --out signals/

The lead file keeps the object the trend bot already reads: ``date``,
``weights``, and ``diagnostics``. The other files are one strategy each.
``signals.json`` in the output directory is the combined object. Weights are
recomputed from the stored walk-forward cell and the latest close. The cell
is not refit here. Re-run ``studies/spy_upro_trend/faster.py`` when a new
calendar year starts.

The same run writes ``risk.json`` next to those files. ``--positions``
(default ``<out>/positions.json``) is either a JSON list of ``{account,
ticker, qty, fill, stop}`` or an object with ``positions``, an optional
``equity`` map, and an optional ``as_of``. A missing file writes an empty
list and a note. A malformed file is named in ``flags`` and the signal
files are still written.
"""

from __future__ import annotations

import argparse
import json
import math
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

# Distance from a hard stop back to SPY. UPRO and SSO are the SPY multiples
# the desk holds. Anything else, including a fund levered to another index, is 1x.
STOP_LEVERAGE = {"UPRO": 3.0, "SSO": 2.0, "SPY": 1.0}
# A stop whose implied SPY level is above the exit, or within this fraction
# below it, can fire before the tested close-based exit.
TIGHT_BAND = 0.02


def load_spec(path: Path) -> dict:
    if not path.exists():
        raise DataError(f"no {path}; run python studies/spy_upro_trend/faster.py first")
    return json.loads(path.read_text())


def _dip_stop(diagnostics: dict) -> str:
    """Stop line for the published trend book. The live dip is three down closes."""
    rule = diagnostics.get("dip_mr_rule")
    if rule and not diagnostics.get("dip_mr_live", True):
        threshold = diagnostics.get("dip_mr_threshold")
        return (
            "no trailing stop; the SPY half switches to SSO when Wilder RSI(2) "
            f"is below {threshold} and leaves when the close is above the 5-day average "
            "or RSI(2) is above 70"
        )
    return "no trailing stop; the SPY half switches to SSO after three down closes"


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
            "stop": _dip_stop(diagnostics),
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


def leverage_for(ticker: str) -> float:
    """SPY multiple used to turn a stop percent into a SPY move."""
    return STOP_LEVERAGE.get(str(ticker).upper(), 1.0)


def _round(value: float, places: int = 6) -> float:
    return round(float(value), places)


def _price(value) -> float | None:
    """A usable positive price, or ``None``."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _number(row: dict, key: str, index: int, path: Path) -> float:
    if key not in row:
        raise DataError(f"{path} row {index} is missing {key}")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{path} row {index} {key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f"{path} row {index} {key} must be finite")
    return number


def load_positions(path: Path) -> dict:
    """Read a positions file without raising.

    The file is either a JSON list of ``{account, ticker, qty, fill, stop}``
    or an object whose ``positions`` list holds those rows. An object may also
    carry ``equity`` (account to dollars) and ``as_of``. Equity may sit on the
    rows as well; the two must agree. Any other key is ignored. A missing file
    returns no rows and no error. A malformed file returns an ``error`` string
    so the publisher can record it and still write the signals.
    """
    found = {"rows": None, "as_of": None, "error": None}
    if not path.exists():
        return found
    as_of = None
    try:
        if not path.is_file():
            raise DataError(f"{path} is not a positions file")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise DataError(f"{path} is not valid JSON ({exc})") from exc
        if isinstance(payload, dict):
            stamp = payload.get("as_of")
            if isinstance(stamp, str) and stamp.strip():
                as_of = stamp.strip()
        found["rows"], found["as_of"] = _positions_payload(payload, path)
    except DataError as exc:
        found["rows"] = None
        found["as_of"] = as_of
        found["error"] = str(exc)
    return found


def _positions_payload(payload, path: Path) -> tuple[list[dict], str | None]:
    """Return ``(rows, as_of)`` from a list or from the desk's object wrapper."""
    if isinstance(payload, list):
        rows = _parse_rows(payload, path)
        _reject_mixed_equity(rows)
        return rows, None
    if not isinstance(payload, dict):
        raise DataError(f"{path} must be a JSON list of positions or an object with a positions list")
    stamp = payload.get("as_of")
    if stamp is not None and not isinstance(stamp, str):
        raise DataError(f"{path} as_of must be a string")
    as_of = stamp.strip() if isinstance(stamp, str) and stamp.strip() else None
    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise DataError(f"{path} positions must be a JSON list")
    rows = _parse_rows(positions, path)
    _apply_equity_map(rows, payload.get("equity"), path)
    _reject_mixed_equity(rows)
    return rows, as_of


def _parse_rows(payload: list, path: Path) -> list[dict]:
    rows = []
    for offset, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise DataError(f"{path} row {offset} must be an object")
        account = row.get("account")
        ticker = row.get("ticker")
        if not isinstance(account, str) or not account.strip():
            raise DataError(f"{path} row {offset} account must be a non-empty string")
        if not isinstance(ticker, str) or not ticker.strip():
            raise DataError(f"{path} row {offset} ticker must be a non-empty string")
        fill = _number(row, "fill", offset, path)
        if fill <= 0:
            raise DataError(f"{path} row {offset} fill must be positive")
        parsed = {
            "account": account.strip(),
            "ticker": ticker.strip().upper(),
            "qty": _number(row, "qty", offset, path),
            "fill": fill,
            "stop": _number(row, "stop", offset, path),
        }
        if "equity" in row and row["equity"] is not None:
            equity = _number(row, "equity", offset, path)
            if equity <= 0:
                raise DataError(f"{path} row {offset} equity must be positive")
            parsed["equity"] = equity
        rows.append(parsed)
    return rows


def _apply_equity_map(rows: list[dict], equity, path: Path) -> None:
    """Copy a top-level ``{account: dollars}`` map onto the rows."""
    if equity is None:
        return
    if not isinstance(equity, dict):
        raise DataError(f"{path} equity must be an object of account to dollars")
    amounts: dict[str, float] = {}
    for name, value in equity.items():
        if not isinstance(name, str) or not name.strip():
            raise DataError(f"{path} equity keys must be account names")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataError(f"{path} equity for {name} must be a number")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise DataError(f"{path} equity for {name} must be a positive number")
        amounts[name.strip()] = number
    for row in rows:
        mapped = amounts.get(row["account"])
        if mapped is None:
            continue
        current = row.get("equity")
        if current is not None and abs(float(current) - mapped) > 1e-6:
            raise DataError(f"account {row['account']} lists equity {current} and {mapped}")
        row["equity"] = mapped


def _reject_mixed_equity(rows: list[dict]) -> None:
    seen: dict[str, float] = {}
    for row in rows:
        if "equity" not in row:
            continue
        name = str(row["account"])
        equity = float(row["equity"])
        prior = seen.get(name)
        if prior is not None and abs(prior - equity) > 1e-6:
            raise DataError(f"account {name} lists equity {prior} and {equity}")
        seen[name] = equity


def _score_position(row: dict, spy_close: float | None, exit_level: float | None) -> dict:
    """One position: dollars at the stop, and where that stop sits in SPY."""
    qty = float(row["qty"])
    fill = float(row["fill"])
    stop = float(row["stop"])
    ticker = str(row["ticker"]).strip().upper()
    dollar = qty * (fill - stop)
    stop_pct = (fill - stop) / fill
    leverage = leverage_for(ticker)
    move = stop_pct / leverage
    implied = None if spy_close is None else spy_close * (1.0 - move)
    gap = None
    gap_pct = None
    too_tight = False
    if implied is not None and exit_level is not None:
        gap = implied - exit_level
        gap_pct = gap / exit_level
        # Above the exit, or within 2% below it, including the boundary.
        too_tight = gap_pct >= -TIGHT_BAND - 1e-9
    stored_leverage = int(leverage) if leverage == int(leverage) else leverage
    return {
        "account": row["account"],
        "ticker": ticker,
        "qty": qty,
        "fill": fill,
        "stop": stop,
        "dollar_risk": _round(dollar),
        "stop_pct": _round(stop_pct),
        "leverage": stored_leverage,
        "implied_spy_move": _round(move),
        "implied_spy": None if implied is None else _round(implied),
        "exit_gap": None if gap is None else _round(gap),
        "exit_gap_pct": None if gap_pct is None else _round(gap_pct),
        "too_tight": too_tight,
    }


def _account_totals(parsed: list[dict], scored: list[dict]) -> list[dict]:
    _reject_mixed_equity(parsed)
    order: list[str] = []
    buckets: dict[str, dict] = {}
    for raw, row in zip(parsed, scored):
        name = row["account"]
        bucket = buckets.get(name)
        if bucket is None:
            bucket = {"dollar": 0.0, "equity": None}
            buckets[name] = bucket
            order.append(name)
        bucket["dollar"] += float(row["qty"]) * (float(row["fill"]) - float(row["stop"]))
        if "equity" in raw and bucket["equity"] is None:
            bucket["equity"] = float(raw["equity"])
    totals = []
    for name in order:
        bucket = buckets[name]
        equity = bucket["equity"]
        risk = bucket["dollar"]
        totals.append({
            "account": name,
            "dollar_risk": _round(risk),
            "equity": equity,
            "risk_pct": None if equity is None else _round(risk / equity),
        })
    return totals


def _flag(row: dict, exit_level: float) -> str:
    gap = row["exit_gap"]
    if gap is not None and gap > 0:
        relation = "above"
    elif gap == 0:
        relation = "at"
    else:
        relation = "within 2% below"
    return (
        f"{row['ticker']} {row['account']}: hard stop implies SPY {row['implied_spy']:.2f}, "
        f"{relation} the trend-core exit {exit_level:.2f}"
    )


def assess_risk(
    rows: list[dict],
    spy_close: float | None,
    exit_level: float | None,
    as_of: str | None = None,
    session: str | None = None,
    note: str | None = None,
) -> dict:
    """Score each position against the trend-core exit.

    Dollar risk is ``qty * (fill - stop)``. ``stop_pct`` is the fraction of
    the fill between the fill and the stop. That fraction divided by leverage
    is the SPY move that hits the stop, and ``implied_spy`` applies the move
    to the latest SPY close. ``exit_gap`` is ``implied_spy`` minus the exit
    (negative means the stop is below the exit). ``too_tight`` is true when
    the implied level is above the exit or within 2% below it. ``risk_pct``
    is dollar risk divided by the account equity when ``equity`` is present.
    """
    close = _price(spy_close)
    exit_px = _price(exit_level)
    scored = [_score_position(row, close, exit_px) for row in rows]
    flags = [_flag(row, exit_px) for row in scored if row["too_tight"] and exit_px is not None]
    return {
        "date": session,
        "as_of": as_of,
        "spy_close": close,
        "exit_level": exit_px,
        "tight_band": TIGHT_BAND,
        "note": note,
        "positions": scored,
        "accounts": _account_totals(rows, scored),
        "flags": flags,
    }


def write_risk(out_dir: Path, lead: dict, positions_path: Path | None = None) -> dict:
    """Write ``risk.json`` beside the strategy files.

    A missing positions file is an empty list and a note. A malformed file
    does not raise: the error is a flag, and the caller can still write the
    signal files.
    """
    path = Path(positions_path) if positions_path is not None else Path(out_dir) / "positions.json"
    levels = lead.get("levels") or {}
    diagnostics = lead.get("diagnostics") or {}
    spy_close = levels.get("reference_close")
    if spy_close is None:
        spy_close = diagnostics.get("reference_close")
    exit_level = levels.get("exit_level")
    loaded = load_positions(path)
    notes = []
    rows = loaded["rows"]
    error = loaded["error"]
    if rows is None:
        rows = []
        if error is None:
            notes.append(f"no positions file at {path}")
    if rows and _price(spy_close) is None:
        notes.append("SPY close is missing, so implied stop levels were not computed")
    if rows and _price(exit_level) is None:
        notes.append("trend-core exit is missing, so too_tight was not judged")
    as_of = loaded["as_of"] or lead.get("as_of") or lead.get("signal_date")
    try:
        report = assess_risk(
            rows,
            spy_close,
            exit_level,
            as_of=as_of,
            session=lead.get("date"),
            note=" ".join(notes) if notes else None,
        )
    except DataError as exc:
        error = error or str(exc)
        report = assess_risk(
            [],
            spy_close,
            exit_level,
            as_of=as_of,
            session=lead.get("date"),
            note=" ".join(notes) if notes else None,
        )
    if error:
        report["flags"] = [error, *report["flags"]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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


def publish(
    out_dir: Path,
    spec_path: Path,
    study_dir: Path,
    cache: str | None = None,
    positions: str | Path | None = None,
) -> dict:
    # A bad positions file is recorded in risk.json after the signals are written.
    positions_path = Path(positions) if positions is not None else Path(out_dir) / "positions.json"
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
    combined = write_bundle(out_dir, lead, fast)
    write_risk(out_dir, lead, positions_path)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write paper-trading signals from the latest close.")
    parser.add_argument("--out", default="signals", help="directory for one JSON per strategy")
    parser.add_argument("--spec", default="studies/faster/live.json")
    parser.add_argument("--study-dir", default="studies/spy_upro_trend")
    parser.add_argument("--cache", default=None)
    parser.add_argument(
        "--positions",
        default=None,
        help="positions JSON (a list, or {positions, equity, as_of}); default is <out>/positions.json",
    )
    args = parser.parse_args(argv)
    try:
        combined = publish(
            Path(args.out),
            Path(args.spec),
            Path(args.study_dir),
            cache=args.cache,
            positions=args.positions,
        )
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
    risk = json.loads((Path(args.out) / "risk.json").read_text())
    dollars = sum(row["dollar_risk"] for row in risk.get("accounts") or [])
    count = len(risk.get("positions") or [])
    print(f"  risk: ${dollars:.2f} across {count} position(s)")
    if risk.get("note"):
        print(f"  risk: {risk['note']}")
    for item in risk.get("flags") or []:
        print(f"  flag: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
