#!/usr/bin/env python3
"""Return-max flagship. The survival rule below is fixed before this sample is scored.

    python studies/flagship/run.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantforge.calendar import next_nyse_session  # noqa: E402
from quantforge.data import CoinbaseSource, YahooSource, load_bars  # noqa: E402
from quantforge.data.base import DataError  # noqa: E402
from quantforge.engine.leverage import FINANCING_SPREAD, LEVERAGED_ETFS, synthesize_leveraged, tracking_stats  # noqa: E402
from quantforge.eval.bootstrap import block_bootstrap  # noqa: E402
from quantforge.eval.ledger import TrialLedger  # noqa: E402
from quantforge.eval.metrics import path_cagr, path_max_drawdown, summarize  # noqa: E402
from quantforge.eval.sharpe import deflated_sharpe, expected_max_z  # noqa: E402
from quantforge.eval.walkforward import walk_forward_select  # noqa: E402
from quantforge.signals.emit import write_signal  # noqa: E402
from quantforge.signals.schema import validate_signal  # noqa: E402

from candidates import (  # noqa: E402
    cta_books,
    crypto_books,
    kelly_overlay,
    momentum_books,
    risk_parity_books,
    trend_core_book,
)

HERE = Path(__file__).resolve().parent
STUDY = "flagship"
MIN_OBS = 756
WF_FIRST_YEAR = 2010
DD_CAPS = (0.30, 0.45, 0.60)
DSR_MIN = 0.95
BOOT_BEAT_MIN = 0.55
BOOT_PATHS = 20_000
BOOT_BLOCK = 63
BOOT_SEED = 7
BASKET = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"]
STOCKS = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL", "TXN", "QCOM", "AMZN",
    "JPM", "BAC", "WFC", "C", "AXP", "GS", "MS",
    "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY", "BMY", "AMGN",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "DIS",
    "GE", "BA", "CAT", "MMM", "HON", "UPS",
]
KELLY_BASES = (
    "trend_core",
    "cta_L1_sma200_band2",
    "cta_L2_sma200_band2",
    "rp_trend_G1",
    "xasset_top3_lb252_G1",
    "sector_top3_lb252_G1",
    "crypto_btc_sma",
)
VALIDATION = (
    ("SSO", "SPY", 2),
    ("UPRO", "SPY", 3),
    ("QLD", "QQQ", 2),
    ("TQQQ", "QQQ", 3),
    ("TMF", "TLT", 3),
    ("UGL", "GLD", 2),
    ("TNA", "IWM", 3),
)

PROTOCOL_MD = """
## Protocol

Precommitted before this sample was scored:

- Objective: maximize out-of-sample CAGR subject to a drawdown budget. The budgets are 30%, 45%, and 60%. A separate track maximizes trailing excess Sharpe and is not allowed to win the growth recommendation.
- Decision: first session of each year from 2010. Score only returns strictly before that session. A variant needs 756 sessions. The growth score is CAGR. A variant is eligible for a budget only if its expanding-window max drawdown is no worse than the cap. Ties break alphabetically. If nothing qualifies, the book is cash.
- Candidates, all counted as trials: the final SPY/UPRO trend-core book; a monthly managed-futures basket (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, DBC) that is long/flat the 1x, 2x, or 3x sleeve when the SMA trend is on, plus a long/short margin version at 1x, 2x, and 3x; inverse-vol risk parity with and without that trend, at 1x, 2x, and 3x gross; 12-1 and 6-1 cross-asset momentum in the top 2 or 3, at 1x and 2x; the same momentum on sector ETFs; BTC and ETH trend and BTC Donchian, plus BTC vol targets of 25% and 40% capped at 2x; half-Kelly (cap 3x, expanding, month-end) on a fixed list of those books. Concentrated stock momentum is estimated and counted, and it is ineligible for the recommendation because the list is names that still exist.
- Leverage above 1x on an unlevered basket is margin financed at the lagged T-bill plus 0.40%. Leveraged-ETF sleeves are the daily-reset model with a 0.95% expense ratio unless the fund is SSO, UPRO, QLD, or TQQQ, which use the sponsor expense ratios from the trend study. Equity trades pay 5 bps. Crypto pays 25 bps plus 3 bps.
- A budget path survives only if its out-of-sample CAGR beats SPY over the same sessions, its deflated Sharpe is at least 0.95 using the persistent ledger, its 5-year block-bootstrap probability of beating SPY is at least 0.55 (20,000 paths, 63-day blocks, seed 7), and its realized out-of-sample max drawdown is inside the cap. The recommendation is the surviving budget path with the highest out-of-sample CAGR. If none survive, the flagship file is 100% SPY and is not endorsed. The full-sample ranking is not a recommendation.
""".strip()

CAVEATS = [
    "Stock momentum uses a fixed list of companies that are still listed. Delisted names are missing, so that sleeve is biased upward. It is shown and counted, and it cannot be the recommendation.",
    "BTC and ETH are the assets that became the liquid majors. That choice is a form of selection. The sleeve is tested because those are the markets a book can actually trade, and the caveat stays attached to the result.",
    "Listed 3x funds for EAFE, emerging markets, Treasuries, and gold are not all modeled with their own expense ratios. Unspecified sleeves use 0.95% plus 40 bps of financing. The overlap table is the check where a fund was downloaded.",
    "Crypto is marked to the last Coinbase daily close at or before the NYSE session. A position that was already on earns that change; a new position does not earn it until the next session.",
    "The drawdown cap is applied to the past. The realized path can be worse. A breach fails the survival rule and the path is not recommended.",
    "Half-Kelly is the growth-optimal fraction of the base book's daily excess return, halved and capped at 3. It is not a promise that the estimate is stable.",
    "This is research on public prices. It is not a recommendation to use margin or leveraged ETFs.",
]


def _load_report():
    path = Path(__file__).with_name("report.py")
    spec = importlib.util.spec_from_file_location("flagship_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_default(value):
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    raise TypeError(f"not serializable: {type(value)}")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.dropna(subset=["open", "close"]).copy()
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _fetch_yahoo(symbols: list[str]) -> dict[str, pd.DataFrame]:
    source = YahooSource()
    found = {}
    for symbol in symbols:
        try:
            found.update(load_bars(source, [symbol], start="1993-01-01"))
            print(f"  {symbol} {found[symbol].index[0].date()} -> {found[symbol].index[-1].date()}", flush=True)
        except DataError as exc:
            print(f"  missing {symbol}: {exc}", flush=True)
    return found


def _fetch_crypto() -> dict[str, pd.DataFrame]:
    try:
        frames = load_bars(CoinbaseSource(), ["BTC-USD", "ETH-USD"], start="2015-01-01")
        print("  crypto via Coinbase", flush=True)
        return frames
    except DataError as exc:
        print(f"  Coinbase failed ({exc}); trying Yahoo", flush=True)
    found = {}
    source = YahooSource()
    for symbol in ("BTC-USD", "ETH-USD"):
        try:
            found.update(load_bars(source, [symbol], start="2014-01-01"))
        except DataError as exc:
            print(f"  missing {symbol}: {exc}", flush=True)
    return found


def _yield_series(close: pd.Series) -> pd.Series:
    series = close.dropna().astype(float)
    if float(series.median()) < 0.5:
        series = series * 100.0
    if float(series.median()) > 30:
        raise DataError(f"^IRX median {float(series.median()):.2f} is not a percent yield")
    return series


def _panel(frames: dict[str, pd.DataFrame], symbols: list[str], calendar: pd.DatetimeIndex, min_start="2007-01-01"):
    """Inner join after the latest listing that still begins by ``min_start``."""
    available = []
    dropped = []
    limit = pd.Timestamp(min_start)
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None:
            dropped.append(symbol)
            continue
        px = _clean(frame)
        if px.empty or px.index[0] > limit or px.index[-1] < calendar[-1] - pd.Timedelta(days=15):
            dropped.append(symbol)
            continue
        available.append((symbol, px))
    if len(available) < 2:
        return None, dropped
    start = max(px.index[0] for _, px in available)
    window = calendar[calendar >= start]
    cleaned = {}
    for symbol, px in available:
        aligned = px.reindex(window)
        if float(aligned["close"].notna().mean()) < 0.98:
            dropped.append(symbol)
            continue
        cleaned[symbol] = aligned.ffill(limit=5)
    if len(cleaned) < 2:
        return None, dropped
    close = pd.DataFrame({symbol: cleaned[symbol]["close"] for symbol in cleaned}).dropna(how="any")
    open_ = pd.DataFrame({symbol: cleaned[symbol]["open"] for symbol in cleaned}).reindex(close.index).dropna(how="any")
    close = close.reindex(open_.index)
    if len(close) < 252 * 8:
        return None, dropped + [f"(intersection {len(close)} days)"]
    return (open_, close), dropped


def _align(series: pd.Series, calendar: pd.DatetimeIndex, rf: pd.Series) -> pd.Series:
    out = pd.Series(series).reindex(calendar).astype(float)
    first = out.first_valid_index()
    if first is None:
        return out
    tail = out.loc[first:].copy()
    filler = rf.reindex(tail.index).fillna(0.0)
    tail = tail.fillna(filler)
    return tail


def _year_starts(index: pd.DatetimeIndex, first_year: int) -> list[pd.Timestamp]:
    starts = []
    for year in range(first_year, int(index[-1].year) + 1):
        hits = index[index.year == year]
        if len(hits):
            starts.append(pd.Timestamp(hits[0]))
    return starts


def _growth(history: pd.Series) -> float:
    value = path_cagr(history)
    return value if math.isfinite(value) else -1e9


def _sharpe_score(rf: pd.Series):
    def score(history: pd.Series) -> float:
        excess = history - rf.reindex(history.index).fillna(0.0)
        deviation = float(excess.std(ddof=1))
        if not math.isfinite(deviation) or deviation == 0.0:
            return -1e9
        return float(excess.mean() / deviation)
    return score


def _accept_dd(cap: float):
    def accept(history: pd.Series) -> bool:
        drawdown = path_max_drawdown(history)
        return math.isfinite(drawdown) and drawdown >= -cap
    return accept


def _frequency(picks: dict) -> list[tuple[str, int]]:
    names = sorted(set(picks.values()))
    return sorted(((name, sum(value == name for value in picks.values())) for name in names), key=lambda item: (-item[1], item[0]))


def _metrics(series: pd.Series, rf: pd.Series, spy: pd.Series) -> dict:
    return summarize(series, rf=rf.reindex(series.index), benchmark=spy.reindex(series.index))


def _validate(frames, rf) -> list[dict]:
    rows = []
    for etf, underlying, leverage in VALIDATION:
        if etf not in frames or underlying not in frames:
            continue
        under = _clean(frames[underlying])
        actual = _clean(frames[etf])
        rate = (rf.reindex(under.index).ffill()).fillna(0.0)
        meta_er = LEVERAGED_ETFS.get(etf, {}).get("expense_ratio", 0.0095)
        synth = synthesize_leveraged(under["open"], under["close"], rate, leverage, meta_er, FINANCING_SPREAD)
        stats = tracking_stats(synth["close"], actual["close"])
        stats["etf"] = etf
        rows.append(stats)
    return rows


def _signal(weights: dict, as_of, endorsed: bool, meta: dict) -> dict:
    clean = {ticker: float(value) for ticker, value in weights.items() if abs(float(value)) > 1e-6}
    negative = any(value < 0 for value in clean.values())
    gross = sum(abs(value) for value in clean.values())
    payload = {
        "date": next_nyse_session(as_of).strftime("%Y-%m-%d"),
        "weights": clean,
        "strategy": "flagship",
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_date": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "endorsed": endorsed,
        "gross": round(gross, 6),
        "diagnostics": meta,
    }
    validate_signal(payload, max_gross=max(1.0, gross), allow_negative=negative)
    return payload


def _verdict(frontier: list[dict], recommendation: dict) -> str:
    parts = []
    for row in frontier:
        parts.append(
            f"{row['track']} compounded at {row['cagr'] * 100:.1f}% with a realized max drawdown of "
            f"{row['max_drawdown'] * 100:.1f}% against SPY at {row['spy_cagr'] * 100:.1f}% "
            f"(DSR {row['dsr']:.2f}, 5y P(beat SPY) {row.get('p_beats_spy', float('nan')):.2f}, "
            f"survives={row['survives']})."
        )
    if recommendation["endorsed"]:
        ending = (
            f" The recommendation is `{recommendation['variant']}` from {recommendation['track']}. "
            "It is the highest out-of-sample CAGR among budget paths that cleared the precommitted bar."
        )
    else:
        ending = (
            f" No budget path survived. The highest out-of-sample CAGR was {recommendation['best_failed_track']} "
            f"at {recommendation['best_failed_cagr'] * 100:.1f}%, and it is not a recommendation. "
            "The flagship file is 100% SPY and is not endorsed."
        )
    return " ".join(parts) + ending


def main() -> None:
    started = time.time()
    print("fetching Yahoo bars", flush=True)
    symbols = list(dict.fromkeys(BASKET + SECTORS + STOCKS + ["SSO", "UPRO", "QLD", "TQQQ", "TMF", "UGL", "TNA", "^IRX"]))
    frames = _fetch_yahoo(symbols)
    if "SPY" not in frames or "^IRX" not in frames:
        raise DataError("SPY and ^IRX are required")
    print("fetching crypto", flush=True)
    crypto = _fetch_crypto()
    spy = _clean(frames["SPY"])
    calendar = spy.index
    irx = _yield_series(frames["^IRX"]["close"])
    rf = (irx.reindex(calendar).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    rf.index = calendar

    basket, basket_dropped = _panel(frames, BASKET, calendar)
    if basket is None:
        raise DataError(f"basket panel failed: {basket_dropped}")
    basket_open, basket_close = basket
    sectors, sector_dropped = _panel(frames, SECTORS, basket_close.index)
    stocks, stock_dropped = _panel(frames, STOCKS, basket_close.index)
    print(
        f"basket {list(basket_close.columns)} {basket_close.index[0].date()} -> {basket_close.index[-1].date()}",
        flush=True,
    )

    spy_book = spy.reindex(basket_close.index).dropna(subset=["open", "close"])
    candidates = [
        trend_core_book(
            spy_book["open"], spy_book["close"], rf.reindex(spy_book.index).fillna(0.0),
        )
    ]
    print(f"trend-core {time.time() - started:.0f}s", flush=True)
    candidates.extend(cta_books(basket_open, basket_close, rf.reindex(basket_close.index).fillna(0.0)))
    print(f"cta {time.time() - started:.0f}s", flush=True)
    candidates.extend(risk_parity_books(basket_open, basket_close, rf.reindex(basket_close.index).fillna(0.0)))
    candidates.extend(momentum_books(
        basket_open, basket_close, rf.reindex(basket_close.index).fillna(0.0),
        "xasset", note="Cross-asset 6-1 and 12-1 momentum. Non-positive scores stay in cash.",
    ))
    if sectors is not None:
        sec_o, sec_c = sectors
        candidates.extend(momentum_books(
            sec_o, sec_c, rf.reindex(sec_c.index).fillna(0.0), "sector",
            lookbacks=(252,), tops=(2, 3), grosses=(1, 2),
            note="Sector-ETF momentum. Constituents can die without the sector disappearing.",
        ))
    if stocks is not None:
        stk_o, stk_c = stocks
        candidates.extend(momentum_books(
            stk_o, stk_c, rf.reindex(stk_c.index).fillna(0.0), "stock",
            lookbacks=(252,), tops=(3, 5, 10), grosses=(1, 1.5), eligible=False,
            note="Survivorship-biased. Ineligible for the recommendation.",
        ))
    candidates.extend(crypto_books(crypto, basket_close.index, rf))
    print(f"sleeves {time.time() - started:.0f}s ({len(candidates)} books)", flush=True)
    by_name = {item.name: item for item in candidates}
    for base_name in KELLY_BASES:
        base = by_name.get(base_name)
        if base is None:
            print(f"  kelly base missing: {base_name}", flush=True)
            continue
        overlay = kelly_overlay(base, rf)
        candidates.append(overlay)
        by_name[overlay.name] = overlay

    master = basket_close.index
    spy_r = spy["close"].pct_change().reindex(master)
    cash = rf.reindex(master).fillna(0.0)
    variants = {"CASH": cash, "BH_SPY": spy_r.dropna()}
    for item in candidates:
        if not item.eligible:
            continue
        variants[item.name] = _align(item.returns, master, cash)
    biased = {"CASH": cash}
    for item in candidates:
        biased[item.name] = _align(item.returns, master, cash)

    sample = spy_r.dropna().index
    decisions = _year_starts(sample, WF_FIRST_YEAR)
    print(f"selecting from {decisions[0].date()} across {len(variants) - 2} eligible books", flush=True)

    ledger = TrialLedger(ROOT / "studies" / "trial_ledger.json")
    sample_end = str(sample[-1].date())
    for item in candidates:
        metrics = _metrics(item.returns.dropna(), rf, spy_r)
        metrics["turnover_per_year"] = item.turnover_per_year
        ledger.record(
            study=STUDY, family=item.family, name=item.name, params=item.params,
            sample_start=str(pd.Timestamp(item.returns.dropna().index[0]).date()),
            sample_end=sample_end, metrics=metrics, counts_for_deflation=item.counts,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()

    def _pack(track: str, result, cap: float | None, recommendable: bool):
        series = result.returns.dropna()
        stats = _metrics(series, rf, spy_r.reindex(series.index))
        stats["dsr"] = deflated_sharpe(series, n_trials, dispersion, rf=rf.reindex(series.index))
        matched = _metrics(spy_r.reindex(series.index).dropna(), rf, spy_r.reindex(series.index))
        boot = {
            "1y": block_bootstrap(series, 252, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_r),
            "5y": block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_r),
        }
        p_beats = boot["5y"].get("p_beats_benchmark")
        inside = True if cap is None else stats["max_drawdown"] >= -cap
        survives = bool(
            recommendable
            and cap is not None
            and stats["cagr"] > matched["cagr"]
            and stats["dsr"] >= DSR_MIN
            and p_beats is not None
            and p_beats >= BOOT_BEAT_MIN
            and inside
        )
        return {
            "track": track,
            "cap": cap,
            "recommendable": recommendable,
            "cagr": stats["cagr"],
            "vol": stats["vol"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "ir_t": stats.get("ir_t"),
            "dsr": stats["dsr"],
            "spy_cagr": matched["cagr"],
            "start": stats.get("start"),
            "end": stats.get("end"),
            "p_beats_spy": p_beats,
            "p_loss": boot["5y"].get("p_loss"),
            "bootstrap": boot,
            "survives": survives,
            "picks": list(result.picks.items()),
            "frequency": _frequency(result.picks),
            "live_variant": list(result.picks.values())[-1],
        }

    frontier = []
    pick_rows = []
    for cap in DD_CAPS:
        selected = walk_forward_select(
            variants, decisions, _growth, min_obs=MIN_OBS, fallback="CASH", accept_fn=_accept_dd(cap),
        )
        row = _pack(f"growth_{int(cap * 100)}", selected, cap, True)
        frontier.append(row)
        pick_rows.append({"track": row["track"], "picks": row["picks"], "frequency": row["frequency"]})
        print(f"  {row['track']} cagr {row['cagr']:.3f} dd {row['max_drawdown']:.3f} live {row['live_variant']}", flush=True)
    sharpe_path = walk_forward_select(variants, decisions, _sharpe_score(rf), min_obs=MIN_OBS, fallback="CASH")
    sharpe_row = _pack("sharpe", sharpe_path, None, False)
    frontier.append(sharpe_row)
    pick_rows.append({"track": "sharpe", "picks": sharpe_row["picks"], "frequency": sharpe_row["frequency"]})
    if stocks is not None:
        disclosed = walk_forward_select(
            biased, decisions, _growth, min_obs=MIN_OBS, fallback="CASH", accept_fn=_accept_dd(0.45),
        )
        disclosed_row = _pack("growth_45_including_stocks", disclosed, 0.45, False)
        frontier.append(disclosed_row)
        pick_rows.append({
            "track": disclosed_row["track"],
            "picks": disclosed_row["picks"],
            "frequency": disclosed_row["frequency"],
        })

    survivors = [row for row in frontier if row["survives"]]
    if survivors:
        winner = max(survivors, key=lambda row: row["cagr"])
        live_name = winner["live_variant"]
        endorsed = True
        track = winner["track"]
    else:
        winner = None
        live_name = "BH_SPY"
        endorsed = False
        track = None
    growth_rows = [row for row in frontier if row["recommendable"]]
    best_failed = max(growth_rows, key=lambda row: row["cagr"])
    if endorsed and live_name in by_name:
        live_weights = by_name[live_name].weights
    elif endorsed and live_name == "BH_SPY":
        live_weights = {"SPY": 1.0}
    elif endorsed and live_name == "CASH":
        live_weights = {"BIL": 1.0}
    else:
        live_weights = {"SPY": 1.0}
    recommendation = {
        "endorsed": endorsed,
        "track": track,
        "variant": live_name if endorsed else "BH_SPY",
        "best_failed_track": None if endorsed else best_failed["track"],
        "best_failed_cagr": None if endorsed else best_failed["cagr"],
        "best_failed_variant": None if endorsed else best_failed["live_variant"],
    }
    signal = _signal(
        live_weights,
        basket_close.index[-1],
        endorsed,
        {
            "track": track,
            "variant": live_name,
            "rule": "oos_growth_under_drawdown_budget",
            "candidate_if_unendorsed": None if endorsed else best_failed["live_variant"],
        },
    )
    negative = any(value < 0 for value in signal["weights"].values())
    gross = max(1.0, float(signal["gross"]))
    write_signal(signal, HERE / "signals_flagship.json", allowed_tickers=None, max_gross=gross, allow_negative=negative)
    trend_dir = ROOT / "studies" / "spy_upro_trend"
    write_signal(signal, trend_dir / "signals_flagship.json", allowed_tickers=None, max_gross=gross, allow_negative=negative)

    insample = []
    for item in candidates:
        stats = _metrics(item.returns.dropna(), rf, spy_r)
        insample.append({
            "name": item.name,
            "family": item.family,
            "eligible": item.eligible,
            "cagr": stats.get("cagr"),
            "vol": stats.get("vol"),
            "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
            "ir_t": stats.get("ir_t"),
            "note": item.note,
        })
    insample.sort(key=lambda row: row["cagr"] if isinstance(row["cagr"], float) and math.isfinite(row["cagr"]) else -1e9, reverse=True)

    for item in candidates:
        stats = _metrics(item.returns.dropna(), rf, spy_r)
        stats["dsr"] = deflated_sharpe(item.returns.dropna(), n_trials, dispersion, rf=rf)
        ledger.record(
            study=STUDY, family=item.family, name=item.name, params=item.params,
            sample_start=str(pd.Timestamp(item.returns.dropna().index[0]).date()),
            sample_end=sample_end, metrics=stats, counts_for_deflation=item.counts,
        )
    ledger.save()

    results = {
        "protocol_md": PROTOCOL_MD,
        "caveats": CAVEATS,
        "sample": {
            "start": str(sample[0].date()),
            "end": str(sample[-1].date()),
            "n_days": int(len(sample)),
        },
        "walk_forward_start": WF_FIRST_YEAR,
        "min_obs": MIN_OBS,
        "n_trials": n_trials,
        "multiple_testing_bar_t": expected_max_z(n_trials),
        "sharpe_dispersion_daily": dispersion,
        "universe": list(basket_close.columns),
        "dropped": basket_dropped + (sector_dropped or []) + (stock_dropped or []),
        "frontier": frontier,
        "picks": pick_rows,
        "insample": insample,
        "bootstrap_block": BOOT_BLOCK,
        "bootstrap_paths": BOOT_PATHS,
        "bootstrap_seed": BOOT_SEED,
        "recommendation": recommendation,
        "live_signal": signal,
        "validation": _validate(frames, rf),
        "verdict": "",
        "elapsed_seconds": time.time() - started,
    }
    results["verdict"] = _verdict(frontier, recommendation)
    _write_json(HERE / "results.json", results)
    (HERE / "RESULTS.md").write_text(_load_report().render_report(results))
    print(results["verdict"], flush=True)
    print(f"done in {time.time() - started:.0f}s endorsed={endorsed} variant={recommendation['variant']}", flush=True)


if __name__ == "__main__":
    main()
