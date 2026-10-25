#!/usr/bin/env python3
"""Apex multi-strategy book. The survival rule below is fixed before this sample is scored.

    python studies/apex/run.py
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

from quantforge.apex.alphas import alpha_panel, xs_zscore  # noqa: E402
from quantforge.apex.combine import combine_equal, gate_names, meta_label_scores, walk_forward_predictions  # noqa: E402
from quantforge.apex.impact import capacity_notionals, impact_matrix  # noqa: E402
from quantforge.apex.portfolio import monthly_targets  # noqa: E402
from quantforge.apex.regime import hmm_scale, vol_trend_scale  # noqa: E402
from quantforge.apex.riskmodel import drawdown_exposure  # noqa: E402
from quantforge.apex.universe import CORE_ETFS, DOLLAR_SYMBOLS, SECTOR, STOCKS, YIELD_SYMBOLS  # noqa: E402
from quantforge.apex.validate import cpcv_sharpes  # noqa: E402
from quantforge.calendar import next_nyse_session  # noqa: E402
from quantforge.data import CoinbaseSource, YahooSource, load_bars  # noqa: E402
from quantforge.data.base import DataError  # noqa: E402
from quantforge.engine.backtest import run_backtest, run_trend_core  # noqa: E402
from quantforge.engine.costs import equity_costs  # noqa: E402
from quantforge.engine.leverage import FINANCING_SPREAD, LEVERAGED_ETFS, synthesize_leveraged  # noqa: E402
from quantforge.engine.overlay import scale_exposure  # noqa: E402
from quantforge.engine.schedule import month_ends, rebalance_mask  # noqa: E402
from quantforge.eval.bootstrap import block_bootstrap  # noqa: E402
from quantforge.eval.ledger import TrialLedger  # noqa: E402
from quantforge.eval.metrics import path_max_drawdown, summarize  # noqa: E402
from quantforge.eval.sharpe import deflated_sharpe, expected_max_z  # noqa: E402
from quantforge.eval.walkforward import walk_forward_select  # noqa: E402
from quantforge.signals.emit import write_signal  # noqa: E402
from quantforge.signals.schema import validate_signal  # noqa: E402
from quantforge.strategies.basket import momentum_weights  # noqa: E402
from quantforge.strategies.dipbuy import dip_active  # noqa: E402
from quantforge.strategies.trend import trend_state  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = "apex"
MIN_OBS = 756
WF_FIRST_YEAR = 2011
DSR_MIN = 0.95
BOOT_BEAT_MIN = 0.55
BOOT_PATHS = 20_000
BOOT_BLOCK = 63
BOOT_SEED = 7
AUM = 5_000.0
STRESS_AUM = 10_000_000.0
DD_LIMIT_PRIMARY = 0.40
DD_LIMIT_LONG = 0.35
COMPONENT_HURDLE = 0.02
ML_HURDLE = 0.05
SHORT_BORROW = 0.005 / 252.0
ALPHA_NAMES = [
    "tsmom", "xsmom", "residual", "reversal", "carry",
    "lowvol", "bab", "value", "pairs", "vrp", "macro",
]
PRIMARY = [
    "er_budget_long", "er_budget_ls",
    "er_hrp_long", "er_mv_long",
    "gated_budget_long", "gated_budget_ls",
    "ridge_budget_long", "ridge_budget_ls",
    "gbdt_budget_long", "gbdt_budget_ls",
    "hmm_budget_long", "ridge_meta_long",
    "er_long_crypto",
]
LONG_ONLY = [name for name in PRIMARY if "_ls" not in name]

PROTOCOL_MD = """
## Protocol

Precommitted before this sample was scored:

- Score: trailing excess Sharpe versus T-bills, looked up on the first session of each year from 2011, using only returns strictly before that session. A book needs 756 sessions. Ties break alphabetically. The fallback is the equal-risk long book.
- The primary set is the equal-risk risk-budget book (long-only and long/short), HRP and mean-variance long-only, the same long-only and long/short books with alphas gated on trailing Sharpe, ridge and gradient boosting (long-only and long/short), the long-only book with a two-state HMM instead of the volatility/trend rule, ridge with meta-label sizing, and the long-only book plus a BTC trend satellite capped at 15%. Leave-one-out alphas, the stock sleeve, and the cost-stress rerun are counted or disclosed and are not in the selector.
- Machine learning is one of those books. It is used when the selector's past window prefers it. A separate horse race asks whether the fixed ridge or gradient-boosting path beats fixed equal-risk by 0.05 Sharpe on the out-of-sample window. That horse race does not override the selector.
- A component "added value" when the fixed equal-risk long Sharpe on the out-of-sample window exceeds the leave-one-out Sharpe by at least 0.02. That table is a diagnostic. It does not pick a new book after the fact.
- Endorsement requires all of: out-of-sample CAGR above SPY and above the final SPY/UPRO trend-core on the same sessions, deflated Sharpe at least 0.95 on the persistent ledger, 5-year block-bootstrap probability at least 0.55 of beating each of those two benchmarks (20,000 paths, 63-day blocks, seed 7, $5,000), and a realized max drawdown no worse than 40% for the primary path or 35% for the long-only path. If both paths qualify, the higher CAGR is the recommendation. If neither qualifies, the bot file is 100% SPY and is not endorsed.
- Costs: 5 bps one-way, plus square-root impact at a $5,000 account, plus 50 bps a year of borrow on short notional and 40 bps of financing above T-bills when the book borrows cash. Rebalance is monthly. The signal is fixed at the close and filled at the next open.
- The stock sleeve is a fixed list of companies that are still listed. It is ineligible for the recommendation.
""".strip()

CAVEATS = [
    "Single-name momentum uses today's survivors. Delisted companies are missing, so that sleeve is biased upward and cannot be the recommendation.",
    "BTC is the crypto asset that became the liquid major. The satellite is a form of selection. It is eligible only as a capped overlay, and the caveat stays attached to the result.",
    "Equity carry is a five-year reversal, not a dividend yield. The adjusted-price cache does not keep a point-in-time dividend. Commodity carry is absent because the free daily bars do not include a futures curve.",
    "The HMM and the gradient-boosting model are fit only on data available at each refit. A missing alpha is a neutral score, not a zero-imputed fundamental.",
    "Drawdown throttling scales the finished daily path. It is causal, and it is not a second pass of the fill simulator.",
    "Square-root impact uses a coefficient of 0.5. That is a model, not a fit to our fills. On liquid ETFs a $5,000 account pays almost nothing. The $10 million rerun is the capacity check.",
    "Hierarchical risk parity and mean-variance are projected onto the position limits by clipping. They are not the output of a commercial quadratic solver.",
    "This is research on public prices. It is not a recommendation to use leverage, shorts, or leveraged ETFs.",
]


def _load_report():
    path = Path(__file__).with_name("report.py")
    spec = importlib.util.spec_from_file_location("apex_report", path)
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
    out = frame.dropna(subset=["close"]).copy()
    if "open" not in out.columns:
        out["open"] = out["close"]
    out = out.dropna(subset=["open", "close"])
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if out.index.tz is not None:
        out.index = out.index.tz_convert("America/New_York").tz_localize(None)
    out.index = out.index.normalize()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out[~out.index.duplicated(keep="last")].sort_index()


def _fetch_yahoo(symbols: list[str]) -> dict[str, pd.DataFrame]:
    source = YahooSource()
    found = {}
    for symbol in symbols:
        try:
            found.update(load_bars(source, [symbol], start="1993-01-01"))
            frame = found[symbol]
            print(f"  {symbol} {frame.index[0].date()} -> {frame.index[-1].date()}", flush=True)
        except DataError as exc:
            print(f"  missing {symbol}: {exc}", flush=True)
    return found


def _fetch_crypto() -> dict[str, pd.DataFrame]:
    try:
        frames = load_bars(CoinbaseSource(), ["BTC-USD"], start="2015-01-01")
        print("  BTC via Coinbase", flush=True)
        return frames
    except DataError as exc:
        print(f"  Coinbase failed ({exc}); trying Yahoo", flush=True)
    try:
        return load_bars(YahooSource(), ["BTC-USD"], start="2014-01-01")
    except DataError as exc:
        print(f"  missing BTC-USD: {exc}", flush=True)
        return {}


def _yield_series(close: pd.Series) -> pd.Series:
    series = close.dropna().astype(float)
    if float(series.median()) < 0.5:
        series = series * 100.0
    if float(series.median()) > 30 and series.name != "^VIX":
        raise DataError(f"yield median {float(series.median()):.2f} is not a percent yield")
    return series


def _panel(frames: dict[str, pd.DataFrame], symbols: list[str], min_start="2007-07-01"):
    available = []
    dropped = []
    limit = pd.Timestamp(min_start)
    end = max(frames[s].index[-1] for s in symbols if s in frames)
    for symbol in symbols:
        frame = frames.get(symbol)
        if frame is None:
            dropped.append(symbol)
            continue
        px = _clean(frame)
        if px.empty or px.index[0] > limit or px.index[-1] < end - pd.Timedelta(days=15):
            dropped.append(symbol)
            continue
        available.append((symbol, px))
    if len(available) < 8:
        raise DataError(f"core panel is too thin: {dropped}")
    start = max(px.index[0] for _, px in available)
    calendar = available[0][1].index
    calendar = calendar[calendar >= start]
    # Use the intersection of the available calendars via SPY if present.
    anchor = dict(available).get("SPY")
    if anchor is not None:
        calendar = anchor.index[anchor.index >= start]
    cleaned = {}
    for symbol, px in available:
        aligned = px.reindex(calendar)
        if float(aligned["close"].notna().mean()) < 0.98:
            dropped.append(symbol)
            continue
        cleaned[symbol] = aligned.ffill(limit=5)
    close = pd.DataFrame({s: cleaned[s]["close"] for s in cleaned}).dropna(how="any")
    open_ = pd.DataFrame({s: cleaned[s]["open"] for s in cleaned}).reindex(close.index)
    volume = pd.DataFrame({s: cleaned[s]["volume"] for s in cleaned}).reindex(close.index).fillna(0.0)
    open_ = open_.dropna(how="any")
    close = close.reindex(open_.index)
    volume = volume.reindex(open_.index)
    return open_, close, volume, dropped


def _lag(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return series.reindex(index).ffill().shift(1)


def _sharpe(series: pd.Series, rf: pd.Series) -> float:
    hist = pd.Series(series).dropna().astype(float)
    if len(hist) < 20:
        return float("nan")
    excess = hist - rf.reindex(hist.index).fillna(0.0)
    deviation = float(excess.std(ddof=1))
    if not math.isfinite(deviation) or deviation <= 0:
        return float("nan")
    return float(excess.mean() / deviation * np.sqrt(252.0))


def _score_factory(rf: pd.Series):
    def score(history: pd.Series) -> float:
        value = _sharpe(history, rf)
        return value if math.isfinite(value) else -1e9
    return score


def _year_starts(index: pd.DatetimeIndex, first_year: int) -> list[pd.Timestamp]:
    starts = []
    last_year = int(index[-1].year)
    for year in range(first_year, last_year + 1):
        hits = index[index.year == year]
        if len(hits):
            starts.append(pd.Timestamp(hits[0]))
    return starts


def _clip_weights(weights: pd.DataFrame, gross: float, long_only: bool) -> pd.DataFrame:
    frame = weights.astype(float).copy()
    if long_only:
        frame = frame.clip(lower=0.0)
    exposure = frame.abs().sum(axis=1)
    overflow = exposure > gross + 1e-8
    if bool(overflow.any()):
        frame.loc[overflow] = frame.loc[overflow].div(exposure[overflow] / gross, axis=0)
    return frame


def _run_book(weights, opens, closes, rf, gross, long_only, volume, aum):
    weights = _clip_weights(weights.reindex(index=opens.index, columns=opens.columns).fillna(0.0), gross, long_only)
    impact = impact_matrix(weights, closes, volume.reindex(index=closes.index, columns=closes.columns), aum)
    reb = month_ends(opens.index)
    cost = equity_costs(list(opens.columns), commission_bps=5.0, slippage_bps=0.0)
    borrow = (FINANCING_SPREAD / 252.0) if gross > 1 else 0.0
    short = SHORT_BORROW if not long_only else 0.0
    result = run_backtest(
        opens, closes, weights, reb, cost, rf.reindex(opens.index).fillna(0.0),
        max_gross=gross, allow_short=not long_only, borrow_spread=borrow,
        short_borrow=short, impact=impact,
    )
    return result, weights


def _apply_dd(returns: pd.Series, rf: pd.Series) -> pd.Series:
    exposure = drawdown_exposure(returns)
    return scale_exposure(returns, exposure, rf.reindex(returns.index).fillna(0.0), cost=0.0005)


def _crypto_weight(close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    state = trend_state(close, 200, 0.02, "daily")
    vol = close.pct_change().rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252.0)
    raw = state.astype(float) * (0.10 / vol.replace(0.0, np.nan))
    raw = raw.clip(upper=0.15).fillna(0.0)
    mask = month_ends(close.index)
    decided = raw.where(mask.reindex(raw.index).fillna(False))
    carried = decided.ffill().fillna(0.0)
    return carried.reindex(index).fillna(0.0).clip(lower=0.0, upper=0.15)


def _blend(base: pd.Series, sleeve: pd.Series, weight: pd.Series, cost: float) -> pd.Series:
    w = weight.reindex(base.index).fillna(0.0).shift(1).fillna(0.0)
    other = sleeve.reindex(base.index)
    w = w.where(other.notna(), 0.0)
    other = other.fillna(0.0)
    turn = w.diff().abs().fillna(w.abs())
    return ((1.0 - w) * base.fillna(0.0) + w * other - turn * cost).where(base.notna())


def _overlay_weights(base: pd.DataFrame, weight: pd.Series, symbol: str) -> pd.DataFrame:
    w = weight.reindex(base.index).fillna(0.0).clip(lower=0.0, upper=0.5)
    out = base.mul(1.0 - w, axis=0)
    out[symbol] = w
    return out


def _trend_core(spy: pd.DataFrame, rf: pd.Series) -> pd.Series:
    sso = synthesize_leveraged(
        spy["open"], spy["close"], rf.reindex(spy.index).fillna(0.0),
        2, LEVERAGED_ETFS["SSO"]["expense_ratio"], FINANCING_SPREAD,
    )
    upro = synthesize_leveraged(
        spy["open"], spy["close"], rf.reindex(spy.index).fillna(0.0),
        3, LEVERAGED_ETFS["UPRO"]["expense_ratio"], FINANCING_SPREAD,
    )
    trend = trend_state(spy["close"], 200, 0.02, "daily")
    dip = dip_active(spy["close"], "3_down_closes")
    mix = rebalance_mask(spy.index, "quarterly")
    result = run_trend_core(
        spy["open"], spy["close"], sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, mix, 0.5, 0.5, rf.reindex(spy.index).fillna(0.0),
        spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    return result.returns


def _book_spec(name: str) -> dict:
    long_only = "_ls" not in name
    method = "budget"
    if "hrp" in name:
        method = "hrp"
    elif "_mv_" in name or name.startswith("er_mv"):
        method = "mv"
    regime = "hmm" if name.startswith("hmm") else "vol"
    if name.startswith("no_regime"):
        regime = None
    scores = "all"
    if name.startswith("gated"):
        scores = "gated"
    elif name.startswith("ridge_meta"):
        scores = "meta"
    elif name.startswith("ridge"):
        scores = "ridge"
    elif name.startswith("gbdt"):
        scores = "gbdt"
    elif name.startswith("loo_"):
        scores = name
    return {
        "method": method,
        "long_only": long_only,
        "gross": 1.0 if long_only else 1.5,
        "max_weight": 0.25 if long_only else 0.20,
        "regime": regime,
        "vol_target": None if name.startswith("no_vol") else (0.10 if long_only else 0.12),
        "kelly": None if name.startswith("no_kelly") else 0.5,
        "blend": 1.0 if name.startswith("no_turnover") else 0.5,
        "dd": not name.startswith("no_dd"),
        "scores": scores,
    }


def _signal(weights: dict, as_of, endorsed: bool, meta: dict) -> dict:
    clean = {}
    for ticker, value in weights.items():
        number = round(float(value), 6)
        if abs(number) > 1e-6:
            clean[ticker] = number
    if not clean:
        clean = {"BIL": 1.0}
    negative = any(value < 0 for value in clean.values())
    gross = sum(abs(value) for value in clean.values())
    payload = {
        "date": next_nyse_session(as_of).strftime("%Y-%m-%d"),
        "weights": clean,
        "strategy": "apex",
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_date": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "endorsed": endorsed,
        "gross": round(gross, 6),
        "diagnostics": meta,
    }
    validate_signal(payload, allowed_tickers=None, max_gross=max(1.0, gross), allow_negative=negative)
    return payload


def _last_weights(frame: pd.DataFrame) -> dict:
    row = frame.iloc[-1]
    return {str(k): float(v) for k, v in row.items() if abs(float(v)) > 1e-8}


def _stress(series: pd.Series, start: str, end: str) -> dict | None:
    window = series.loc[start:end].dropna()
    if len(window) < 5:
        return None
    return {
        "total": float((1.0 + window).prod() - 1.0),
        "max_drawdown": path_max_drawdown(window),
        "days": int(len(window)),
    }


def _verdict(primary: dict, long_only: dict, recommendation: dict) -> str:
    def one(row):
        return (
            f"{row['track']} compounded at {row['cagr'] * 100:.1f}% with Sharpe {row['sharpe']:.2f} and "
            f"max drawdown {row['max_drawdown'] * 100:.1f}%, against SPY at {row['spy_cagr'] * 100:.1f}% and "
            f"trend-core at {row['trend_cagr'] * 100:.1f}% (DSR {row['dsr']:.2f}, "
            f"5y P(>SPY) {row['p_beats_spy']:.2f}, 5y P(>trend) {row['p_beats_trend']:.2f}, "
            f"survives={row['survives']})."
        )
    ending = (
        f" The endorsed book is `{recommendation['variant']}`."
        if recommendation["endorsed"]
        else " Neither walk-forward path cleared the precommitted bar. The bot file is 100% SPY and is not endorsed."
    )
    return one(primary) + " " + one(long_only) + ending


def main() -> None:
    started = time.time()
    print("fetching Yahoo bars", flush=True)
    symbols = list(dict.fromkeys(CORE_ETFS + STOCKS + YIELD_SYMBOLS + DOLLAR_SYMBOLS + ["SPY"]))
    frames = _fetch_yahoo(symbols)
    if "SPY" not in frames or "^IRX" not in frames or "^VIX" not in frames or "^TNX" not in frames:
        raise DataError("SPY, ^IRX, ^VIX, and ^TNX are required")
    print("fetching crypto", flush=True)
    crypto = _fetch_crypto()
    opens, closes, volume, dropped = _panel(frames, CORE_ETFS)
    print(f"panel {list(closes.columns)} {closes.index[0].date()} -> {closes.index[-1].date()}", flush=True)
    spy_full = _clean(frames["SPY"])
    irx = _yield_series(frames["^IRX"]["close"])
    calendar = spy_full.index.union(closes.index)
    rf_full = (irx.reindex(calendar).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    rf = rf_full.reindex(closes.index).fillna(0.0)

    yields = {}
    for symbol in YIELD_SYMBOLS:
        if symbol in frames and symbol != "^VIX":
            yields[symbol] = _lag(_yield_series(frames[symbol]["close"]), closes.index)
    vix = _lag(frames["^VIX"]["close"].astype(float), closes.index)
    tnx = yields.get("^TNX", pd.Series(index=closes.index, dtype=float))
    curve = tnx - yields.get("^IRX", 0.0)
    credit = (closes["HYG"] / closes["HYG"].shift(63) - closes["LQD"] / closes["LQD"].shift(63)).shift(1) if "HYG" in closes and "LQD" in closes else pd.Series(np.nan, index=closes.index)
    if "UUP" in frames:
        dollar_px = _clean(frames["UUP"])["close"].reindex(closes.index).ffill()
        dollar = dollar_px.pct_change(63).shift(1)
    else:
        dollar = pd.Series(np.nan, index=closes.index)
        dropped.append("UUP")

    print("alphas", flush=True)
    panels = alpha_panel(closes, closes["SPY"], yields, vix, curve, credit, dollar)
    zmap = {name: xs_zscore(panel) for name, panel in panels.items()}
    combined = combine_equal(panels, ALPHA_NAMES)
    loo = {name: combine_equal(panels, [other for other in ALPHA_NAMES if other != name]) for name in ALPHA_NAMES}

    print("standalone alphas", flush=True)
    standalone_returns = {}
    for name in ALPHA_NAMES:
        weights = monthly_targets(
            zmap[name], closes, method="budget", long_only=False, gross=1.0, max_weight=0.3,
            sector_map=SECTOR, sector_cap=0.7, regime=None, vol_target=None, kelly_fraction=None,
            turnover_blend=1.0, rf=rf,
        )
        result, _weights = _run_book(weights, opens, closes, rf, 1.0, False, volume, AUM)
        standalone_returns[name] = result.returns
        print(f"  {name} {_sharpe(result.returns, rf):.2f}", flush=True)

    print("gated scores and models", flush=True)
    gated = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for decision in closes.index[month_ends(closes.index).to_numpy()]:
        keep = gate_names(standalone_returns, decision)
        row = combine_equal({name: panels[name] for name in keep}, keep)
        if decision in row.index:
            gated.loc[decision] = row.loc[decision]
    ridge = walk_forward_predictions(panels, closes, ALPHA_NAMES, model="ridge", min_rows=180)
    print(f"  ridge {time.time() - started:.0f}s", flush=True)
    gbdt = walk_forward_predictions(panels, closes, ALPHA_NAMES, model="gbdt", min_rows=180)
    print(f"  gbdt {time.time() - started:.0f}s", flush=True)
    meta = meta_label_scores(ridge, closes, min_rows=80)
    ridge = ridge.reindex(closes.index)
    gbdt = gbdt.reindex(closes.index)
    meta = meta.reindex(closes.index)

    vol_regime = vol_trend_scale(closes["SPY"])
    refits = _year_starts(closes.index, int(closes.index[0].year))
    hmm = hmm_scale(closes["SPY"], refits)
    print(f"regimes {time.time() - started:.0f}s", flush=True)

    score_frames = {"all": combined, "gated": gated, "ridge": ridge, "gbdt": gbdt, "meta": meta}
    for name, frame in loo.items():
        score_frames[f"loo_{name}"] = frame
    regimes = {"vol": vol_regime, "hmm": hmm, None: None}

    variant_names = list(PRIMARY) + [f"loo_{name}" for name in ALPHA_NAMES]
    variant_names += ["no_regime_long", "no_dd_long", "no_vol_target_long", "no_kelly_long", "no_turnover_long"]
    # Crypto enters the primary set only when the series exists. Drop it otherwise.
    btc_weight = None
    btc_returns = None
    if "BTC-USD" in crypto:
        btc = _clean(crypto["BTC-USD"])
        btc_close = btc["close"]
        btc_close.index = pd.DatetimeIndex(pd.to_datetime(btc_close.index)).tz_localize(None).normalize()
        btc_close = btc_close[~btc_close.index.duplicated(keep="last")].sort_index()
        mark = btc_close.reindex(closes.index.union(btc_close.index)).sort_index().ffill(limit=4).reindex(closes.index)
        btc_returns = mark.pct_change()
        btc_weight = _crypto_weight(mark.dropna(), closes.index)
    else:
        variant_names = [name for name in variant_names if name != "er_long_crypto"]

    print("building books", flush=True)
    store = {}
    weight_frames = {}
    for name in variant_names:
        spec = _book_spec(name)
        scores = score_frames[spec["scores"]]
        weights = monthly_targets(
            scores, closes, method=spec["method"], long_only=spec["long_only"], gross=spec["gross"],
            max_weight=spec["max_weight"], sector_map=SECTOR, sector_cap=0.50,
            regime=regimes[spec["regime"]], vol_target=spec["vol_target"],
            kelly_fraction=spec["kelly"], turnover_blend=spec["blend"], rf=rf,
        )
        result, weights = _run_book(weights, opens, closes, rf, spec["gross"], spec["long_only"], volume, AUM)
        series = _apply_dd(result.returns, rf) if spec["dd"] else result.returns
        store[name] = {"returns": series, "turnover": result.turnover, "eligible": name in PRIMARY, "spec": spec}
        weight_frames[name] = weights
        print(f"  {name} sharpe {_sharpe(series, rf):.2f}", flush=True)

    if btc_weight is not None and "er_budget_long" in store:
        blended = _blend(store["er_budget_long"]["returns"], btc_returns, btc_weight, 0.0028)
        store["er_long_crypto"] = {
            "returns": blended, "turnover": store["er_budget_long"]["turnover"],
            "eligible": True, "spec": _book_spec("er_long_crypto"),
        }
        weight_frames["er_long_crypto"] = _overlay_weights(weight_frames["er_budget_long"], btc_weight, "BTC-USD")

    stock_note = "not built"
    if sum(symbol in frames for symbol in STOCKS) >= 20:
        stock_names = [symbol for symbol in STOCKS if symbol in frames]
        stock_close = pd.DataFrame({s: _clean(frames[s])["close"] for s in stock_names})
        stock_open = pd.DataFrame({s: _clean(frames[s])["open"] for s in stock_names})
        both = stock_close.dropna(axis=1, thresh=int(0.9 * len(closes))).dropna(how="any")
        if len(both) > 252 * 8:
            stock_open = stock_open.reindex(both.index)[both.columns].dropna(how="any")
            stock_close = both.reindex(stock_open.index)
            sw = momentum_weights(stock_close, 252, 21, 5, 1.0, True)
            s_reb = month_ends(stock_close.index)
            s_cost = equity_costs(list(stock_close.columns), commission_bps=5.0)
            s_rf = rf_full.reindex(stock_close.index).fillna(0.0)
            stock_bt = run_backtest(
                stock_open, stock_close, sw.reindex(stock_close.index).fillna(0.0),
                s_reb, s_cost, s_rf, max_gross=1.0,
            )
            sleeve_w = pd.Series(0.15, index=closes.index)
            sleeve_w = sleeve_w.where(month_ends(closes.index).reindex(closes.index).fillna(False)).ffill().fillna(0.0)
            blended = _blend(store["er_budget_long"]["returns"], stock_bt.returns, sleeve_w, 0.0005)
            store["er_long_stocks"] = {
                "returns": blended, "turnover": store["er_budget_long"]["turnover"],
                "eligible": False, "spec": {"note": "survivorship"},
            }
            stock_note = f"{stock_close.shape[1]} names from {stock_close.index[0].date()}"
    print(f"stocks {stock_note}", flush=True)

    trend = _trend_core(spy_full, rf_full)
    spy_r = spy_full["close"].pct_change()

    eligible = {name: store[name]["returns"] for name in PRIMARY if name in store}
    long_names = {name: eligible[name] for name in LONG_ONLY if name in eligible}
    sample = closes["SPY"].pct_change().dropna().index
    decisions = _year_starts(sample, WF_FIRST_YEAR)
    print(f"selecting from {decisions[0].date()} across {len(eligible)} books", flush=True)
    primary_wf = walk_forward_select(eligible, decisions, _score_factory(rf), min_obs=MIN_OBS, fallback="er_budget_long")
    long_wf = walk_forward_select(long_names, decisions, _score_factory(rf), min_obs=MIN_OBS, fallback="er_budget_long")

    ledger = TrialLedger(ROOT / "studies" / "trial_ledger.json")
    sample_end = str(sample[-1].date())

    def _record(name: str, series: pd.Series, family: str, params: dict, counts: bool, metrics: dict | None = None):
        clean = series.dropna()
        if clean.empty:
            return
        stats = metrics or summarize(clean, rf=rf.reindex(clean.index), benchmark=spy_r.reindex(clean.index))
        ledger.record(
            study=STUDY, family=family, name=name, params=params,
            sample_start=str(pd.Timestamp(clean.index[0]).date()),
            sample_end=sample_end, metrics=stats, counts_for_deflation=counts,
        )

    for name, payload in store.items():
        _record(name, payload["returns"], "apex_book", {"name": name}, True)
    for name, series in standalone_returns.items():
        _record(f"standalone_{name}", series, "alpha_standalone", {"alpha": name}, True)
    _record("trend_core", trend.reindex(closes.index).dropna(), "benchmark", {"book": "trend_core"}, True)
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()

    def _pack(track: str, result, dd_limit: float):
        series = result.returns.dropna()
        common = series.index.intersection(spy_r.dropna().index).intersection(trend.dropna().index)
        series = series.reindex(common).dropna()
        spy_w = spy_r.reindex(series.index).dropna()
        series = series.reindex(spy_w.index)
        tc_w = trend.reindex(series.index).dropna()
        series = series.reindex(tc_w.index)
        spy_w = spy_r.reindex(series.index)
        tc_w = trend.reindex(series.index)
        stats = summarize(series, rf=rf.reindex(series.index), benchmark=spy_w)
        stats["dsr"] = deflated_sharpe(series, n_trials, dispersion, rf=rf.reindex(series.index))
        spy_stats = summarize(spy_w, rf=rf.reindex(series.index))
        tc_stats = summarize(tc_w, rf=rf.reindex(series.index), benchmark=spy_w)
        boot_spy = block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_w)
        boot_tc = block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=tc_w)
        boot_spy_1 = block_bootstrap(series, 252, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_w)
        p_spy = boot_spy.get("p_beats_benchmark")
        p_tc = boot_tc.get("p_beats_benchmark")
        beats_spy = bool(stats["cagr"] > spy_stats["cagr"])
        beats_tc = bool(stats["cagr"] > tc_stats["cagr"])
        survives = bool(
            beats_spy and beats_tc and stats["dsr"] >= DSR_MIN
            and p_spy is not None and p_spy >= BOOT_BEAT_MIN
            and p_tc is not None and p_tc >= BOOT_BEAT_MIN
            and stats["max_drawdown"] >= -dd_limit
        )
        picks = list(result.picks.items())
        freq = {}
        for _year, picked in picks:
            freq[picked] = freq.get(picked, 0) + 1
        return {
            "track": track,
            "live_variant": list(result.picks.values())[-1],
            "cagr": stats["cagr"],
            "vol": stats["vol"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "ir_t": stats.get("ir_t"),
            "dsr": stats["dsr"],
            "spy_cagr": spy_stats["cagr"],
            "trend_cagr": tc_stats["cagr"],
            "p_beats_spy": p_spy,
            "p_beats_trend": p_tc,
            "beats_spy": beats_spy,
            "beats_trend": beats_tc,
            "survives": survives,
            "picks": picks,
            "frequency": sorted(freq.items(), key=lambda item: (-item[1], item[0])),
            "bootstrap": {"1y": boot_spy_1, "5y_spy": boot_spy, "5y_trend": boot_tc},
            "returns_start": str(series.index[0].date()),
            "returns_end": str(series.index[-1].date()),
            "_series": series,
            "_spy": spy_w,
            "_trend": tc_w,
        }

    primary = _pack("primary", primary_wf, DD_LIMIT_PRIMARY)
    cash = _pack("long_only", long_wf, DD_LIMIT_LONG)
    print(
        f"primary {primary['live_variant']} cagr {primary['cagr']:.3f} survives {primary['survives']}",
        flush=True,
    )
    print(
        f"long_only {cash['live_variant']} cagr {cash['cagr']:.3f} survives {cash['survives']}",
        flush=True,
    )

    survivors = [row for row in (primary, cash) if row["survives"]]
    if survivors:
        winner = max(survivors, key=lambda row: row["cagr"])
        endorsed = True
        live_name = winner["live_variant"]
        live_track = winner["track"]
    else:
        endorsed = False
        live_name = "BH_SPY"
        live_track = None
        winner = None

    window = primary["_series"].index
    full_sharpe = _sharpe(store["er_budget_long"]["returns"].reindex(window), rf)
    components = []
    for name in ALPHA_NAMES:
        without = _sharpe(store[f"loo_{name}"]["returns"].reindex(window), rf)
        delta = full_sharpe - without
        components.append({
            "name": name,
            "sharpe_full": full_sharpe,
            "sharpe_without": without,
            "delta": delta,
            "added": bool(math.isfinite(delta) and delta >= COMPONENT_HURDLE),
        })
    for label, other in (
        ("ridge", "ridge_budget_long"),
        ("gbdt", "gbdt_budget_long"),
        ("hmm", "hmm_budget_long"),
        ("hrp", "er_hrp_long"),
        ("mean_variance", "er_mv_long"),
        ("meta_label", "ridge_meta_long"),
        ("drawdown_throttle", "no_dd_long"),
        ("vol_target", "no_vol_target_long"),
        ("kelly", "no_kelly_long"),
        ("turnover_penalty", "no_turnover_long"),
        ("regime", "no_regime_long"),
    ):
        if other not in store:
            continue
        without = _sharpe(store[other]["returns"].reindex(window), rf)
        # For the construction alternatives, delta is "alternative minus baseline".
        if label in ("ridge", "gbdt", "hmm", "hrp", "mean_variance", "meta_label"):
            delta = without - full_sharpe
            hurdle = ML_HURDLE if label in ("ridge", "gbdt") else COMPONENT_HURDLE
            components.append({
                "name": label,
                "sharpe_full": without,
                "sharpe_without": full_sharpe,
                "delta": delta,
                "added": bool(math.isfinite(delta) and delta >= hurdle),
            })
        else:
            delta = full_sharpe - without
            components.append({
                "name": label,
                "sharpe_full": full_sharpe,
                "sharpe_without": without,
                "delta": delta,
                "added": bool(math.isfinite(delta) and delta >= COMPONENT_HURDLE),
            })
    if "er_long_crypto" in store:
        crypto_sharpe = _sharpe(store["er_long_crypto"]["returns"].reindex(window), rf)
        delta = crypto_sharpe - full_sharpe
        components.append({
            "name": "btc_satellite",
            "sharpe_full": crypto_sharpe,
            "sharpe_without": full_sharpe,
            "delta": delta,
            "added": bool(math.isfinite(delta) and delta >= COMPONENT_HURDLE),
        })

    def _horse(name: str, series: pd.Series, benchmark_spy, benchmark_tc):
        stats = summarize(series, rf=rf.reindex(series.index), benchmark=benchmark_spy)
        stats["name"] = name
        stats["dsr"] = deflated_sharpe(series, n_trials, dispersion, rf=rf.reindex(series.index)) if name not in ("SPY",) else None
        boot_s = block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=benchmark_spy)
        boot_t = block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=benchmark_tc)
        stats["p_beats_spy"] = boot_s.get("p_beats_benchmark")
        stats["p_beats_trend"] = boot_t.get("p_beats_benchmark")
        return stats

    spy_w = primary["_spy"]
    tc_w = primary["_trend"]
    horse = [
        _horse("SPY", spy_w, spy_w, tc_w),
        _horse("trend_core", tc_w, spy_w, tc_w),
        _horse("primary_walk_forward", primary["_series"], spy_w, tc_w),
        _horse("long_only_walk_forward", cash["_series"].reindex(window).dropna(), spy_w, tc_w),
        _horse("equal_risk_long", store["er_budget_long"]["returns"].reindex(window).dropna(), spy_w, tc_w),
        _horse("ridge_long", store["ridge_budget_long"]["returns"].reindex(window).dropna(), spy_w, tc_w),
        _horse("gbdt_long", store["gbdt_budget_long"]["returns"].reindex(window).dropna(), spy_w, tc_w),
    ]

    stress_rows = []
    periods = (
        ("2008", "2008-01-01", "2009-03-31"),
        ("2020", "2020-02-01", "2020-04-30"),
        ("2022", "2022-01-01", "2022-12-31"),
    )
    books = {
        "primary_walk_forward": primary["_series"],
        "equal_risk_long": store["er_budget_long"]["returns"],
        "SPY": spy_r,
        "trend_core": trend,
    }
    for label, start, end in periods:
        for book, series in books.items():
            stats = _stress(series, start, end)
            if stats is None:
                continue
            stress_rows.append({"window": label, "book": book, **stats})

    regime_rows = []
    applied = vol_regime.reindex(window).shift(1)
    path = primary["_series"]
    labels = pd.Series("unclassified", index=path.index)
    labels = labels.where(~(applied >= 0.9), "calm")
    labels = labels.where(~((applied >= 0.55) & (applied < 0.9)), "hot")
    labels = labels.where(~(applied < 0.55), "down")
    for label, chunk in path.groupby(labels):
        clean = chunk.dropna()
        if clean.empty:
            continue
        regime_rows.append({
            "regime": label,
            "days": int(len(clean)),
            "mean": float(clean.mean()),
            "total": float((1.0 + clean).prod() - 1.0),
        })

    base_weights = weight_frames["er_budget_long"]
    impact_small = impact_matrix(base_weights, closes, volume, AUM)
    impact_large = impact_matrix(base_weights, closes, volume, STRESS_AUM)
    traded = base_weights.diff().abs().sum(axis=1)
    small_bps = (impact_small.where(base_weights.diff().abs() > 0) * 1e4).stack()
    large_bps = (impact_large.where(base_weights.diff().abs() > 0) * 1e4).stack()
    stress_result, _w = _run_book(base_weights, opens, closes, rf, 1.0, True, volume, STRESS_AUM)
    spy_adv = float((closes["SPY"] * volume["SPY"]).rolling(21).median().iloc[-1])
    spy_vol = float(closes["SPY"].pct_change().rolling(21).std(ddof=1).iloc[-1])
    capacity = {
        "aum": AUM,
        "stress_aum": STRESS_AUM,
        "median_impact_bps": float(small_bps.replace(0.0, np.nan).median()) if len(small_bps) else 0.0,
        "stress_median_impact_bps": float(large_bps.replace(0.0, np.nan).median()) if len(large_bps) else 0.0,
        "base_cagr": summarize(store["er_budget_long"]["returns"].dropna(), rf=rf).get("cagr"),
        "stress_cagr": summarize(stress_result.returns.dropna(), rf=rf).get("cagr"),
        "spy_adv": spy_adv,
        "spy_capacity_10bps": capacity_notionals(spy_adv, spy_vol, (10.0,)).get(10.0, 0.0),
        "turnover_per_year": float(traded.reindex(window).fillna(0.0).sum() / max(len(window) / 252.0, 1e-9) / 2.0),
    }

    cpcv = {
        "primary_walk_forward": {k: v for k, v in cpcv_sharpes(primary["_series"]).items() if k != "sharpes"},
        "equal_risk_long": {k: v for k, v in cpcv_sharpes(store["er_budget_long"]["returns"].reindex(window)).items() if k != "sharpes"},
        "SPY": {k: v for k, v in cpcv_sharpes(spy_w).items() if k != "sharpes"},
    }

    standalone_rows = []
    for name, series in standalone_returns.items():
        stats = summarize(series.reindex(window).dropna(), rf=rf)
        standalone_rows.append({
            "name": name, "cagr": stats.get("cagr"), "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
        })

    insample = []
    for name, payload in store.items():
        stats = summarize(payload["returns"].dropna(), rf=rf, benchmark=spy_r)
        insample.append({
            "name": name, "eligible": payload["eligible"],
            "cagr": stats.get("cagr"), "sharpe": stats.get("sharpe"),
            "max_drawdown": stats.get("max_drawdown"),
        })
    insample.sort(key=lambda row: row["cagr"] if isinstance(row["cagr"], float) and math.isfinite(row["cagr"]) else -1e9, reverse=True)

    for name, payload in store.items():
        clean = payload["returns"].dropna()
        stats = summarize(clean, rf=rf.reindex(clean.index), benchmark=spy_r.reindex(clean.index))
        stats["dsr"] = deflated_sharpe(clean, n_trials, dispersion, rf=rf.reindex(clean.index))
        _record(name, clean, "apex_book", {"name": name}, True, stats)
    ledger.save()

    if endorsed:
        signal_weights = _last_weights(weight_frames[live_name])
    else:
        signal_weights = {"SPY": 1.0}
    added = [row["name"] for row in components if row["added"]]
    meta_diag = {
        "track": live_track,
        "variant": live_name if endorsed else primary["live_variant"],
        "endorsed_variant": live_name if endorsed else None,
        "beats_spy": bool(winner["beats_spy"]) if winner else False,
        "beats_trend_core": bool(winner["beats_trend"]) if winner else False,
        "components_that_added_value": added,
        "ml_beats_equal_risk": any(row["name"] in ("ridge", "gbdt") and row["added"] for row in components),
        "rule": "walk_forward_excess_sharpe",
    }
    signal = _signal(signal_weights, closes.index[-1], endorsed, meta_diag)
    model_weights = _last_weights(weight_frames[primary["live_variant"]])
    model_signal = _signal(
        model_weights, closes.index[-1], bool(primary["survives"]),
        {**meta_diag, "variant": primary["live_variant"], "book": "primary_walk_forward"},
    )
    long_weights = _last_weights(weight_frames[cash["live_variant"]])
    long_signal = _signal(
        long_weights, closes.index[-1], bool(cash["survives"]),
        {**meta_diag, "variant": cash["live_variant"], "book": "long_only_walk_forward"},
    )
    write_signal(signal, HERE / "signals.json", allowed_tickers=None, max_gross=max(1.0, signal["gross"]), allow_negative=any(v < 0 for v in signal["weights"].values()))
    write_signal(model_signal, HERE / "signals_model.json", allowed_tickers=None, max_gross=max(1.0, model_signal["gross"]), allow_negative=any(v < 0 for v in model_signal["weights"].values()))
    write_signal(long_signal, HERE / "signals_long_only.json", allowed_tickers=None, max_gross=max(1.0, long_signal["gross"]), allow_negative=any(v < 0 for v in long_signal["weights"].values()))

    def _public(row):
        return {key: value for key, value in row.items() if not key.startswith("_")}

    boot_rows = []
    for track, row in (("primary", primary), ("long_only", cash)):
        boot_rows.append({
            "track": track, "horizon": "1y",
            "median": row["bootstrap"]["1y"]["median"],
            "p10": row["bootstrap"]["1y"]["p10"],
            "p90": row["bootstrap"]["1y"]["p90"],
            "p_loss": row["bootstrap"]["1y"]["p_loss"],
            "p_beats_spy": row["bootstrap"]["1y"].get("p_beats_benchmark"),
            "p_beats_trend": None,
        })
        boot_rows.append({
            "track": track, "horizon": "5y",
            "median": row["bootstrap"]["5y_spy"]["median"],
            "p10": row["bootstrap"]["5y_spy"]["p10"],
            "p90": row["bootstrap"]["5y_spy"]["p90"],
            "p_loss": row["bootstrap"]["5y_spy"]["p_loss"],
            "p_beats_spy": row["p_beats_spy"],
            "p_beats_trend": row["p_beats_trend"],
        })

    recommendation = {
        "endorsed": endorsed,
        "track": live_track,
        "variant": live_name,
        "primary_variant": primary["live_variant"],
        "long_only_variant": cash["live_variant"],
    }
    results = {
        "protocol_md": PROTOCOL_MD,
        "caveats": CAVEATS,
        "sample": {
            "start": str(closes.index[0].date()),
            "end": str(closes.index[-1].date()),
            "n_days": int(len(closes)),
        },
        "universe": list(closes.columns),
        "dropped": dropped,
        "n_trials": n_trials,
        "multiple_testing_bar_t": expected_max_z(n_trials),
        "sharpe_dispersion_daily": dispersion,
        "horse_race": [{k: v for k, v in row.items() if k in {
            "name", "cagr", "vol", "sharpe", "max_drawdown", "ir_t", "dsr", "p_beats_spy", "p_beats_trend",
        }} for row in horse],
        "tracks": [_public(primary), _public(cash)],
        "components": components,
        "standalone": standalone_rows,
        "stress": stress_rows,
        "bootstrap_rows": boot_rows,
        "bootstrap_block": BOOT_BLOCK,
        "bootstrap_paths": BOOT_PATHS,
        "bootstrap_seed": BOOT_SEED,
        "cpcv": cpcv,
        "capacity": capacity,
        "regime": regime_rows,
        "insample": insample,
        "recommendation": recommendation,
        "live_signal": signal,
        "model_signal": model_signal,
        "long_only_signal": long_signal,
        "stocks": stock_note,
        "verdict": "",
        "elapsed_seconds": time.time() - started,
    }
    results["verdict"] = _verdict(primary, cash, recommendation)
    _write_json(HERE / "results.json", results)
    (HERE / "RESULTS.md").write_text(_load_report().render_report(results))
    (HERE / "signals.md").write_text(
        "\n".join([
            "# Apex signals",
            "",
            "`signals.json` is the bot file. It is the endorsed book, or 100% SPY when nothing was endorsed.",
            "`signals_model.json` is the primary walk-forward book's latest target, endorsed only if that path cleared the bar.",
            "`signals_long_only.json` is the long-only walk-forward book for a cash account.",
            "",
            f"Session: {signal['date']}",
            f"Endorsed: {signal['endorsed']}",
            f"Weights: {signal['weights']}",
            "",
        ])
    )
    print(results["verdict"], flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
