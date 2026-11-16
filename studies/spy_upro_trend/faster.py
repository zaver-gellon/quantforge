#!/usr/bin/env python3
"""Why the 2x trend book trails SPY over short windows, and three faster books.

The grids, the cost, and the keep rule are fixed in the strategy modules and
in ``quantforge.eval.materiality``. Do not add a cell after seeing this sample.

Run from the repo root:

    python studies/spy_upro_trend/faster.py
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.data import CoinbaseSource, YahooSource, load_bars
from quantforge.data.base import DataError
from quantforge.engine.backtest import run_backtest, run_trend_core
from quantforge.engine.costs import CostModel
from quantforge.engine.schedule import rebalance_mask, week_ends
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.materiality import adjacent_cells, evaluate_materiality
from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, probabilistic_sharpe
from quantforge.eval.walkforward import walk_forward_select
from quantforge.signals.emit import write_signal, write_signals_note
from quantforge.signals.schema import ALLOWED_TREND_TICKERS
from quantforge.publish import annotate_lead, build_fast_signals, write_bundle
from quantforge.strategies.boost import (
    FRACTIONS as BOOST_FRACTIONS,
    HOLDS as BOOST_HOLDS,
    THRESHOLDS as BOOST_THRESHOLDS,
    book_frame,
    boost_episodes,
    boost_flag,
    cash_frame,
    cell_name as boost_name,
    cells as boost_cells,
    close_to_close_book,
    episode_summary,
    named_cell,
    overlay_frame,
    parse_cell as parse_boost,
)
from quantforge.strategies.btctrend import PAIRS as BTC_PAIRS
from quantforge.strategies.btctrend import cell_name as btc_name
from quantforge.strategies.btctrend import cells as btc_cells
from quantforge.strategies.btctrend import trend_on as btc_trend_on
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.exposure import LIVE_EXPOSURE, ROLL_WINDOWS, gross_exposure, mix_weights, trail_rates
from quantforge.strategies.fastsignal import book_weights
from quantforge.strategies.meanrev import ENTRIES, HOLDS, VEHICLES
from quantforge.strategies.meanrev import cell_name as meanrev_name
from quantforge.strategies.meanrev import cells as meanrev_cells
from quantforge.strategies.meanrev import entry_mask, episode_stats, regime_on, target_flag
from quantforge.strategies.nasdaqvol import BANDS as NASDAQ_BANDS
from quantforge.strategies.nasdaqvol import TARGETS as NASDAQ_TARGETS
from quantforge.strategies.nasdaqvol import cell_name as nasdaq_name
from quantforge.strategies.nasdaqvol import cells as nasdaq_cells
from quantforge.strategies.nasdaqvol import tqqq_weight
from quantforge.strategies.panel import build_panel
from quantforge.strategies.rotation import BONDS, CASH, EQUITIES, GOLD, LOOKBACKS
from quantforge.strategies.rotation import cell_name as rotation_name
from quantforge.strategies.rotation import cells as rotation_cells
from quantforge.strategies.rotation import cash_trailing_return, trailing_return, weekly_pick
from quantforge.strategies.trend import trend_state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER_PATH = ROOT / "studies" / "trial_ledger.json"
OUT_DIR = ROOT / "studies" / "faster"
SPEC_PATH = OUT_DIR / "live.json"
SIGNAL_DIR = ROOT / "signals"
STUDY = "spy_upro_trend"
SAMPLE_START = "2000-01-03"
FIRST_YEAR = 2006
MIN_OBS = 756
BASE_COST = 0.0005
STRESS_COST = 0.0010
CRYPTO_COST = 0.0028
PRIOR_ID = "2026-11-13-dip-hold"
EXPECTED_CLOSE = "2026-11-13"
REBALANCE_AXIS = ("daily", "monthly", "quarterly")
QQQ_BANDS = (0.0, 0.02, 0.04)
YAHOO = [
    "SPY", "SSO", "UPRO", "QQQ", "QLD", "TQQQ", "TLT", "TMF", "GLD", "UGL", "BIL", "^IRX",
]

HYPOTHESIS_GAP = (
    "The live book is 50% SPY and 50% UPRO while SPY is inside its 200-day 2% band, "
    "so the equity exposure is 2x whenever the trend is on and the dip is off. "
    "A 2x book trails SPY over short windows when the market is down or choppy, "
    "even if the long-run compound growth is higher. This run measures that at "
    "1.0x, 1.5x, 2.0x, and 2.5x, plus a 100% SPY/BIL band with no dip. "
    "Two changes are then scored on the materiality bar. The first replaces the "
    "quarterly mix reset with a monthly reset. Daily is on the axis so monthly "
    "has a neighbor; quarterly is the live book and is not a new trial. "
    "The second replaces SPY/SSO/UPRO with QQQ/QLD/TQQQ under the same dip and "
    "a 200-day band of 0%, 2%, or 4%. The 2% cell is the named swap. "
    "The live book changes only if a walk-forward path clears the bar."
)

HYPOTHESIS_FAST = (
    "Three additional paper books, none of which replaces the trend account "
    "unless its own note says the materiality bar cleared and the desk adopts it. "
    "Mean reversion enters at the close while SPY is above its 200-day average, "
    "on RSI(2) below 5, 10, or 15, on IBS below 0.2, or on a close below the "
    "5-day lower band, and holds SPY, SSO, or UPRO for 3, 5, or 10 sessions "
    "or until SPY closes above its 5-day average. Weekly rotation picks the "
    "best of UPRO or TQQQ, TMF, and UGL when that return beats T-bills over "
    "21, 42, or 63 sessions, otherwise BIL. UGL is 2x gold. Nasdaq vol holds "
    "TQQQ in a 200-day band of 0% or 2%, sized to 20%, 30%, or 40% realized "
    "vol with a 100% cap. Bitcoin, if the coin prints, is long when a fast "
    "average is above a slow average (10/30, 20/50, 50/200) and otherwise BIL."
)


def _plain(value):
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")


def _score_factory(rf: pd.Series):
    def score(history: pd.Series) -> float:
        excess = history - rf.reindex(history.index).fillna(0.0)
        deviation = float(excess.std(ddof=1))
        if not math.isfinite(deviation) or deviation == 0.0:
            return -1e9
        return float(excess.mean() / deviation)
    return score


def _year_starts(index: pd.DatetimeIndex, min_obs: int = MIN_OBS, first_year: int = FIRST_YEAR) -> list:
    starts = []
    last_year = int(index[-1].year)
    first = max(first_year, int(index[0].year))
    for year in range(first, last_year + 1):
        hits = index[index.year == year]
        if len(hits) == 0:
            continue
        start = pd.Timestamp(hits[0])
        if int((index < start).sum()) >= min_obs:
            starts.append(start)
    if not starts:
        raise RuntimeError("no walk-forward date has enough history")
    return starts


def _finite_index(parts: list[pd.Series]) -> pd.DatetimeIndex:
    index = parts[0].dropna().index
    for series in parts[1:]:
        index = index.intersection(series.dropna().index)
    return pd.DatetimeIndex(index).sort_values()


def _cost_model(symbols: list[str], cost: float) -> CostModel:
    return CostModel(commission_bps={symbol: cost * 10_000.0 for symbol in symbols})


def _run_assets(opens: pd.DataFrame, closes: pd.DataFrame, targets: pd.DataFrame, reb: pd.Series, rf: pd.Series, cost: float):
    symbols = list(opens.columns)
    return run_backtest(
        opens, closes, targets.reindex(columns=symbols).fillna(0.0),
        reb, _cost_model(symbols, cost), rf.reindex(opens.index).fillna(0.0),
    )


def _simulate_core(open1, close1, open2, close2, open3, close3, trend, dip, mask, core, sleeve, rf, cost):
    index = _finite_index([open1, close1, open2, close2, open3, close3, rf])
    result = run_trend_core(
        open1.reindex(index), close1.reindex(index),
        open2.reindex(index), close2.reindex(index),
        open3.reindex(index), close3.reindex(index),
        trend.reindex(index).fillna(False),
        dip.reindex(index).fillna(False),
        mask.reindex(index).fillna(False),
        core, sleeve, rf.reindex(index).fillna(0.0),
        spy_cost=cost, sso_cost=cost, upro_cost=cost,
    )
    returns = result.returns.dropna()
    turnover = result.turnover.reindex(returns.index).fillna(0.0)
    return returns, turnover


def _pack(returns: pd.Series, rf: pd.Series, benchmark: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    window_rf = rf.reindex(returns.index).fillna(0.0)
    bench = None if benchmark is None else benchmark.reindex(returns.index)
    stats = summarize(returns, rf=window_rf, benchmark=bench, turnover=turnover)
    keep = (
        "cagr", "vol", "sharpe", "sharpe_per_period", "max_drawdown", "worst_year",
        "worst_year_return", "n_days", "start", "end", "turnover_per_year", "total_return",
    )
    return {key: stats.get(key) for key in keep}


def _oos(returns: pd.Series, start) -> pd.Series:
    return returns.loc[pd.Timestamp(start):].dropna()


def _without_best_year(returns: pd.Series, rf: pd.Series) -> dict:
    yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    year = int(yearly.idxmax())
    kept = returns.loc[returns.index.year != year]
    stats = _pack(kept, rf)
    stats["year"] = year
    stats["year_return"] = float(yearly.loc[year])
    return stats


def _active_from_flag(flag: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return flag.reindex(index).fillna(False).astype(bool)


def _holding_stats(flag: pd.Series, index: pd.DatetimeIndex) -> dict:
    return episode_stats(_active_from_flag(flag, index))


def _lead_slice(lead: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = lead.reindex(index).astype(float)
    if aligned.isna().any():
        raise RuntimeError("the lead book is missing sessions the candidate traded")
    return aligned


def main() -> None:
    print("refreshing Yahoo prices", flush=True)
    frames = load_bars(YahooSource(), YAHOO, start="1993-01-01", refresh=True)
    btc_note = None
    try:
        print("refreshing BTC-USD", flush=True)
        crypto = load_bars(CoinbaseSource(), ["BTC-USD"], start="2015-01-01", refresh=True)
        frames["BTC-USD"] = crypto["BTC-USD"]
    except DataError as exc:
        btc_note = f"BTC-USD was not loaded ({exc}). The Bitcoin book was not run."
        print(btc_note, flush=True)
    panel = build_panel(frames)
    clock = panel["clock"]
    last_close = pd.Timestamp(clock[-1])
    as_of = str(last_close.date())
    session = next_nyse_session(last_close).strftime("%Y-%m-%d")
    rf = panel["rf"]
    spy_close = panel["spy"]["close"]
    spy_cc = spy_close.pct_change()
    trend = trend_state(spy_close, 200, 0.02, "daily")
    dip = dip_active(spy_close, "3_down_closes")
    print(f"SPY {spy_close.index[0].date()} -> {as_of} ({len(spy_close)} sessions)", flush=True)

    def spy_book(core, sleeve, freq, cost, band=0.02, with_dip=True):
        state = trend if abs(band - 0.02) < 1e-12 else trend_state(spy_close, 200, band, "daily")
        mask = rebalance_mask(clock, freq)
        overlay = dip if with_dip else pd.Series(False, index=clock)
        return _simulate_core(
            panel["spy"]["open"], spy_close,
            panel["stitched"]["SSO"]["open"], panel["stitched"]["SSO"]["close"],
            panel["stitched"]["UPRO"]["open"], panel["stitched"]["UPRO"]["close"],
            state, overlay, mask, core, sleeve, rf, cost,
        )

    print("lead book and exposure grid", flush=True)
    lead_r, lead_turn = spy_book(0.5, 0.5, "quarterly", BASE_COST)
    sample = lead_r.loc[SAMPLE_START:]
    decisions = _year_starts(sample.index)
    oos_start = decisions[0]
    lead_oos = _oos(sample, oos_start)
    spy_oos = spy_cc.reindex(lead_oos.index)
    lead_stats = _pack(lead_oos, rf, spy_oos, lead_turn.reindex(lead_oos.index))

    exposure_rows = []
    exposure_paths = {}
    for exposure in (1.0, 1.5, 2.0, 2.5):
        core, sleeve = mix_weights(exposure)
        if abs(exposure - LIVE_EXPOSURE) < 1e-12:
            returns, turnover = lead_r, lead_turn
        else:
            returns, turnover = spy_book(core, sleeve, "quarterly", BASE_COST)
        path = _oos(returns.loc[SAMPLE_START:], oos_start).reindex(lead_oos.index)
        if path.isna().any():
            raise RuntimeError(f"exposure {exposure} is missing lead-book sessions")
        stats = _pack(path, rf, spy_oos.reindex(path.index), turnover.reindex(path.index))
        trail = trail_rates(path, spy_oos.reindex(path.index), ROLL_WINDOWS)
        name = f"{exposure:.1f}x"
        exposure_rows.append({
            "name": name,
            "exposure": exposure,
            "core": core,
            "sleeve": sleeve,
            "gross": gross_exposure(core, sleeve),
            "oos": stats,
            "trail": trail,
            "counts_for_deflation": abs(exposure - LIVE_EXPOSURE) > 1e-12,
        })
        exposure_paths[name] = (returns.loc[SAMPLE_START:], stats)
        print(f"  {name} cagr {stats['cagr']:.4f} sharpe {stats['sharpe']:.3f} dd {stats['max_drawdown']:.3f}", flush=True)

    bil_state = trend.astype(float)
    bil_index = _finite_index([panel["spy"]["open"], spy_close, rf])
    bil_reb = trend.reindex(bil_index).fillna(False).astype(bool)
    bil_reb = bil_reb.ne(bil_reb.shift(1)).fillna(True)
    bil_targets = pd.DataFrame({"SPY": trend.reindex(bil_index).fillna(False).astype(float)})
    bil_run = _run_assets(
        panel["spy"]["open"].reindex(bil_index).to_frame("SPY"),
        spy_close.reindex(bil_index).to_frame("SPY"),
        bil_targets, bil_reb, rf.reindex(bil_index), BASE_COST,
    )
    bil_oos = _oos(bil_run.returns.dropna().loc[SAMPLE_START:], oos_start).reindex(lead_oos.index)
    bil_stats = _pack(bil_oos, rf, spy_oos, bil_run.turnover.reindex(bil_oos.index))
    bil_trail = trail_rates(bil_oos, spy_oos, ROLL_WINDOWS)
    exposure_rows.append({
        "name": "SPY/BIL",
        "exposure": 1.0,
        "core": 0.0,
        "sleeve": 1.0,
        "gross": 1.0,
        "oos": bil_stats,
        "trail": bil_trail,
        "counts_for_deflation": True,
        "note": "100% SPY when the 2% band is on, cash when it is off. No dip.",
    })

    recent = lead_r.loc["2026-10-26":as_of].dropna()
    recent_spy = spy_cc.reindex(recent.index).dropna()
    recent = recent.reindex(recent_spy.index)
    recent_block = {
        "start": str(recent.index[0].date()) if len(recent) else None,
        "end": str(recent.index[-1].date()) if len(recent) else None,
        "n_days": int(len(recent)),
        "lead_return": float((1.0 + recent).prod() - 1.0) if len(recent) else None,
        "spy_return": float((1.0 + recent_spy).prod() - 1.0) if len(recent_spy) else None,
        "trend_on_share": float(trend.reindex(recent.index).mean()) if len(recent) else None,
        "dip_on_share": float(dip.reindex(recent.index).mean()) if len(recent) else None,
    }
    if recent_block["lead_return"] is not None and recent_block["spy_return"] is not None:
        recent_block["spy_multiple"] = 1.0 + recent_block["spy_return"]
        recent_block["lead_if_spy_ended_at_4927"] = 4927.0 * (1.0 + recent_block["lead_return"]) / recent_block["spy_multiple"]

    print("rebalance and QQQ grids", flush=True)
    rebalance_paths = {}
    rebalance_turns = {}
    for freq in REBALANCE_AXIS:
        if freq == "quarterly":
            rebalance_paths[freq] = lead_r.loc[SAMPLE_START:]
            rebalance_turns[freq] = lead_turn.reindex(rebalance_paths[freq].index).fillna(0.0)
        else:
            returns, turnover = spy_book(0.5, 0.5, freq, BASE_COST)
            rebalance_paths[freq] = returns.loc[SAMPLE_START:]
            rebalance_turns[freq] = turnover.reindex(rebalance_paths[freq].index).fillna(0.0)
    qqq_close = panel["under"]["QQQ"]["close"]
    qqq_trend = {band: trend_state(qqq_close, 200, band, "daily") for band in QQQ_BANDS}
    qqq_dip = dip_active(qqq_close, "3_down_closes")
    qqq_mask = rebalance_mask(clock, "quarterly")
    qqq_paths = {}
    qqq_turns = {}
    for band in QQQ_BANDS:
        returns, turnover = _simulate_core(
            panel["under"]["QQQ"]["open"], qqq_close,
            panel["stitched"]["QLD"]["open"], panel["stitched"]["QLD"]["close"],
            panel["stitched"]["TQQQ"]["open"], panel["stitched"]["TQQQ"]["close"],
            qqq_trend[band], qqq_dip, qqq_mask, 0.5, 0.5, rf, BASE_COST,
        )
        name = f"qqq_band{int(round(band * 100)):02d}"
        qqq_paths[name] = returns.loc[SAMPLE_START:]
        qqq_turns[name] = turnover.reindex(qqq_paths[name].index).fillna(0.0)

    print("mean reversion", flush=True)
    regime = regime_on(spy_close, 200)
    high = panel["spy"]["high"]
    low = panel["spy"]["low"]
    mean_paths = {}
    mean_turns = {}
    mean_flags = {}
    mean_stress = {}
    vehicle_px = {
        "SPY": (panel["spy"]["open"], spy_close),
        "SSO": (panel["stitched"]["SSO"]["open"], panel["stitched"]["SSO"]["close"]),
        "UPRO": (panel["stitched"]["UPRO"]["open"], panel["stitched"]["UPRO"]["close"]),
    }
    for entry, hold, vehicle in meanrev_cells():
        flag = target_flag(spy_close, entry_mask(spy_close, high, low, entry), regime, hold)
        name = meanrev_name(entry, hold, vehicle)
        opened, closed = vehicle_px[vehicle]
        index = _finite_index([opened, closed, rf])
        weight = flag.reindex(index).fillna(False).astype(float).to_frame(vehicle)
        reb = pd.Series(True, index=index)
        base = _run_assets(opened.reindex(index).to_frame(vehicle), closed.reindex(index).to_frame(vehicle), weight, reb, rf, BASE_COST)
        stress = _run_assets(opened.reindex(index).to_frame(vehicle), closed.reindex(index).to_frame(vehicle), weight, reb, rf, STRESS_COST)
        mean_paths[name] = base.returns.dropna().loc[SAMPLE_START:]
        mean_turns[name] = base.turnover.reindex(mean_paths[name].index).fillna(0.0)
        mean_flags[name] = flag
        mean_stress[name] = stress.returns.dropna().loc[SAMPLE_START:]

    print("weekly rotation", flush=True)
    rot_index = _finite_index([
        panel["stitched"]["UPRO"]["close"], panel["stitched"]["TQQQ"]["close"],
        panel["stitched"]["TMF"]["close"], panel["stitched"]["UGL"]["close"], rf,
    ])
    rot_index = rot_index[rot_index >= pd.Timestamp(SAMPLE_START)]
    rot_closes = pd.DataFrame({
        symbol: panel["stitched"][symbol]["close"].reindex(rot_index) for symbol in ("UPRO", "TQQQ", "TMF", "UGL")
    })
    rot_opens = pd.DataFrame({
        symbol: panel["stitched"][symbol]["open"].reindex(rot_index) for symbol in ("UPRO", "TQQQ", "TMF", "UGL")
    })
    rot_week = week_ends(rot_index)
    rot_paths = {}
    rot_turns = {}
    rot_picks = {}
    rot_stress = {}
    for equity, lookback in rotation_cells():
        scores = {
            equity: trailing_return(rot_closes[equity], lookback),
            BONDS: trailing_return(rot_closes[BONDS], lookback),
            GOLD: trailing_return(rot_closes[GOLD], lookback),
        }
        hurdle = cash_trailing_return(rf.reindex(rot_index), lookback)
        pick = weekly_pick(scores, hurdle, rot_index)
        name = rotation_name(equity, lookback)
        assets = [equity, BONDS, GOLD]
        targets = pd.DataFrame({symbol: (pick == symbol).astype(float) for symbol in assets}, index=rot_index)
        base = _run_assets(rot_opens[assets], rot_closes[assets], targets, rot_week, rf.reindex(rot_index), BASE_COST)
        stress = _run_assets(rot_opens[assets], rot_closes[assets], targets, rot_week, rf.reindex(rot_index), STRESS_COST)
        rot_paths[name] = base.returns.dropna()
        rot_turns[name] = base.turnover.reindex(rot_paths[name].index).fillna(0.0)
        rot_picks[name] = pick
        rot_stress[name] = stress.returns.dropna()

    print("nasdaq vol target", flush=True)
    nq_index = _finite_index([
        panel["under"]["QQQ"]["close"], panel["stitched"]["TQQQ"]["open"], panel["stitched"]["TQQQ"]["close"], rf,
    ])
    nq_paths = {}
    nq_turns = {}
    nq_weights = {}
    nq_stress = {}
    qqq_on = panel["under"]["QQQ"]["close"].reindex(nq_index)
    tqqq_o = panel["stitched"]["TQQQ"]["open"].reindex(nq_index)
    tqqq_c = panel["stitched"]["TQQQ"]["close"].reindex(nq_index)
    for band, target in nasdaq_cells():
        weight = tqqq_weight(qqq_on, tqqq_c, band, target)
        name = nasdaq_name(band, target)
        targets = weight.to_frame("TQQQ")
        reb = pd.Series(True, index=nq_index)
        base = _run_assets(tqqq_o.to_frame("TQQQ"), tqqq_c.to_frame("TQQQ"), targets, reb, rf.reindex(nq_index), BASE_COST)
        stress = _run_assets(tqqq_o.to_frame("TQQQ"), tqqq_c.to_frame("TQQQ"), targets, reb, rf.reindex(nq_index), STRESS_COST)
        nq_paths[name] = base.returns.dropna().loc[SAMPLE_START:]
        nq_turns[name] = base.turnover.reindex(nq_paths[name].index).fillna(0.0)
        nq_weights[name] = weight
        nq_stress[name] = stress.returns.dropna().loc[SAMPLE_START:]

    btc_paths = {}
    btc_turns = {}
    btc_flags = {}
    btc_stress = {}
    btc_crypto = {}
    if panel.get("btc") is not None and "BTC-USD" in frames:
        print("bitcoin trend", flush=True)
        btc_px = panel["btc"].dropna(subset=["open", "close"])
        btc_px = btc_px.loc[btc_px.index >= pd.Timestamp("2015-01-01")]
        for fast, slow in btc_cells():
            flag = btc_trend_on(btc_px["close"], fast, slow)
            name = btc_name(fast, slow)
            weight = flag.astype(float).to_frame("BTC-USD")
            reb = pd.Series(True, index=btc_px.index)
            opened = btc_px["open"].to_frame("BTC-USD")
            closed = btc_px["close"].to_frame("BTC-USD")
            base = _run_assets(opened, closed, weight, reb, rf.reindex(btc_px.index), BASE_COST)
            stress = _run_assets(opened, closed, weight, reb, rf.reindex(btc_px.index), STRESS_COST)
            crypto = _run_assets(opened, closed, weight, reb, rf.reindex(btc_px.index), CRYPTO_COST)
            btc_paths[name] = base.returns.dropna()
            btc_turns[name] = base.turnover.reindex(btc_paths[name].index).fillna(0.0)
            btc_flags[name] = flag
            btc_stress[name] = stress.returns.dropna()
            btc_crypto[name] = crypto.returns.dropna()

    ledger = TrialLedger(LEDGER_PATH)
    sample_end = as_of

    def record_path(family, name, params, returns, turnover, count):
        stats = _pack(returns, rf, spy_cc.reindex(returns.index), turnover)
        ledger.record(
            study=STUDY, family=family, name=name, params=params,
            sample_start=str(returns.index[0].date()), sample_end=sample_end,
            metrics=stats, counts_for_deflation=count,
        )
        return stats

    print("RSI(2) boost", flush=True)
    boost_index = _finite_index([
        panel["spy"]["open"], spy_close, panel["stitched"]["UPRO"]["open"],
        panel["stitched"]["UPRO"]["close"], panel["stitched"]["SSO"]["open"],
        panel["stitched"]["SSO"]["close"], rf,
    ])
    boost_flags = {}
    boost_next = {}
    boost_close_paths = {}
    boost_turns = {}
    boost_opens = pd.DataFrame({
        "SPY": panel["spy"]["open"].reindex(boost_index),
        "UPRO": panel["stitched"]["UPRO"]["open"].reindex(boost_index),
    })
    boost_closes = pd.DataFrame({
        "SPY": spy_close.reindex(boost_index),
        "UPRO": panel["stitched"]["UPRO"]["close"].reindex(boost_index),
    })
    boost_reb = pd.Series(True, index=boost_index)
    for threshold, fraction, hold in boost_cells():
        flag = boost_flag(spy_close, threshold, hold)
        name = boost_name(threshold, fraction, hold)
        weights = book_frame(flag.reindex(boost_index).fillna(False), fraction)
        nxt = _run_assets(boost_opens, boost_closes, weights, boost_reb, rf.reindex(boost_index), BASE_COST)
        closed = close_to_close_book(weights, boost_closes, rf.reindex(boost_index), BASE_COST)
        boost_flags[name] = flag
        boost_next[name] = nxt.returns.dropna().loc[SAMPLE_START:]
        boost_close_paths[name] = closed.loc[SAMPLE_START:]
        boost_turns[name] = nxt.turnover.reindex(boost_next[name].index).fillna(0.0)
    named = named_cell()
    stand_w = cash_frame(boost_flags[named].reindex(boost_index).fillna(False))
    stand_opens = boost_opens[["UPRO"]]
    stand_closes = boost_closes[["UPRO"]]
    stand_next = _run_assets(stand_opens, stand_closes, stand_w, boost_reb, rf.reindex(boost_index), BASE_COST)
    stand_close = close_to_close_book(stand_w, stand_closes, rf.reindex(boost_index), BASE_COST)
    overlay_w = overlay_frame(trend, dip, boost_flags[named], 1.0 / 3.0)
    overlay_w = overlay_w.reindex(boost_index).fillna(0.0)
    changed = overlay_w.diff().abs().sum(axis=1).fillna(1.0) > 1e-8
    overlay_reb = (changed | rebalance_mask(boost_index, "quarterly")).fillna(True)
    overlay_reb.iloc[0] = True
    overlay_opens = pd.DataFrame({
        "SPY": boost_opens["SPY"],
        "SSO": panel["stitched"]["SSO"]["open"].reindex(boost_index),
        "UPRO": boost_opens["UPRO"],
    })
    overlay_closes = pd.DataFrame({
        "SPY": boost_closes["SPY"],
        "SSO": panel["stitched"]["SSO"]["close"].reindex(boost_index),
        "UPRO": boost_closes["UPRO"],
    })
    overlay_next = _run_assets(overlay_opens, overlay_closes, overlay_w, overlay_reb, rf.reindex(boost_index), BASE_COST)
    overlay_close = close_to_close_book(overlay_w, overlay_closes, rf.reindex(boost_index), BASE_COST)

    print("recording trials", flush=True)
    for row in exposure_rows:
        if row["name"] == "SPY/BIL":
            record_path(
                "exposure", "spy_bil_band2",
                {"book": "SPY/BIL", "band": 0.02, "dip": False, "rebalance": "on_flip"},
                bil_run.returns.dropna().loc[SAMPLE_START:], bil_run.turnover, True,
            )
        elif row["counts_for_deflation"]:
            full, _stats = exposure_paths[row["name"]]
            record_path(
                "exposure", row["name"],
                {"exposure": row["exposure"], "core": row["core"], "sleeve": row["sleeve"], "dip": True, "band": 0.02},
                full, None, True,
            )
    for freq, path in rebalance_paths.items():
        if freq == "quarterly":
            continue
        record_path(
            "rebalance", f"reb_{freq}",
            {"rebalance": freq, "core": 0.5, "sleeve": 0.5, "band": 0.02, "dip": True},
            path, rebalance_turns[freq], True,
        )
    for name, path in qqq_paths.items():
        record_path(
            "qqq_trend", name,
            {"band": int(name[-2:]) / 100.0, "core": 0.5, "sleeve": 0.5, "dip": True, "rebalance": "quarterly"},
            path, qqq_turns[name], True,
        )
    for entry, hold, vehicle in meanrev_cells():
        name = meanrev_name(entry, hold, vehicle)
        record_path(
            "meanrev", name,
            {"entry": entry, "max_hold": hold, "vehicle": vehicle, "regime": "close_above_sma200"},
            mean_paths[name], mean_turns[name], True,
        )
    for equity, lookback in rotation_cells():
        name = rotation_name(equity, lookback)
        record_path(
            "rotation", name,
            {"equity": equity, "lookback": lookback, "bonds": BONDS, "gold": GOLD, "gold_leverage": 2},
            rot_paths[name], rot_turns[name], True,
        )
    for band, target in nasdaq_cells():
        name = nasdaq_name(band, target)
        record_path(
            "nasdaq_vol", name,
            {"band": band, "vol_target": target, "vol_window": 20, "cap": 1.0},
            nq_paths[name], nq_turns[name], True,
        )
    for threshold, fraction, hold in boost_cells():
        name = boost_name(threshold, fraction, hold)
        params = {"threshold": threshold, "fraction": fraction, "max_hold": hold, "exit_rsi": 70.0, "regime": "sma200"}
        record_path(
            "spy_rsi_boost", f"{name}_next_open", {**params, "fill": "next_open"},
            boost_next[name], boost_turns[name], True,
        )
        record_path(
            "spy_rsi_boost", f"{name}_close", {**params, "fill": "close"},
            boost_close_paths[name], None, True,
        )
    record_path(
        "spy_rsi_boost", "standalone_upro_next_open",
        {"rule": "upro_or_cash", "cell": named, "fill": "next_open"},
        stand_next.returns.dropna().loc[SAMPLE_START:], stand_next.turnover, True,
    )
    record_path(
        "spy_rsi_boost", "standalone_upro_close",
        {"rule": "upro_or_cash", "cell": named, "fill": "close"},
        stand_close.loc[SAMPLE_START:], None, True,
    )
    record_path(
        "spy_rsi_boost", "lead_overlay_next_open",
        {"rule": "lead_spy_half", "fraction_of_core": 1.0 / 3.0, "cell": named, "fill": "next_open"},
        overlay_next.returns.dropna().loc[SAMPLE_START:], overlay_next.turnover, True,
    )
    record_path(
        "spy_rsi_boost", "lead_overlay_close",
        {"rule": "lead_spy_half", "fraction_of_core": 1.0 / 3.0, "cell": named, "fill": "close"},
        overlay_close.loc[SAMPLE_START:], None, True,
    )
    for fast, slow in btc_cells():
        name = btc_name(fast, slow)
        if name not in btc_paths:
            continue
        record_path(
            "btc_trend", name,
            {"fast": fast, "slow": slow, "fallback": "BIL"},
            btc_paths[name], btc_turns[name], True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    print(f"ledger trials {n_trials} dispersion {dispersion:.6f}", flush=True)

    def attach_dsr(family, name, params, returns, turnover, count=True):
        stats = record_path(family, name, params, returns, turnover, count)
        window_rf = rf.reindex(returns.index)
        stats["psr"] = probabilistic_sharpe(returns, rf=window_rf)
        stats["dsr"] = deflated_sharpe(returns, n_trials, dispersion, rf=window_rf)
        ledger.record(
            study=STUDY, family=family, name=name, params=params,
            sample_start=str(returns.index[0].date()), sample_end=sample_end,
            metrics=stats, counts_for_deflation=count,
        )
        return stats

    def run_walk(family, paths, turns, stress, neighbor_fn, flag_fn, current_name=None):
        decisions_i = _year_starts(next(iter(paths.values())).index)
        walk = walk_forward_select(paths, decisions_i, _score_factory(rf), min_obs=MIN_OBS)
        current = _lead_slice(lead_oos, walk.returns.index)
        last = list(walk.picks.values())[-1]
        if current_name is not None and last == current_name:
            cell_path = None
            neighbors = None
        else:
            cell_path = paths[last]
            neighbors = {item: paths[item] for item in neighbor_fn(last)}
        result = evaluate_materiality(
            walk.returns, current, neighbors, cell_path, n_trials, dispersion, rf=rf,
        )
        stress_walk = _stitch(stress, walk.picks, walk.returns.index)
        oos_turn = _stitch(turns, walk.picks, walk.returns.index)
        oos_stats = _pack(walk.returns, rf, spy_cc.reindex(walk.returns.index), oos_turn)
        window_rf = rf.reindex(walk.returns.index)
        oos_stats["dsr"] = float(deflated_sharpe(walk.returns, n_trials, dispersion, rf=window_rf))
        oos_stats["psr"] = float(probabilistic_sharpe(walk.returns, rf=window_rf))
        flag = None if flag_fn is None else flag_fn(last, walk.returns.index)
        hold = episode_stats(flag) if flag is not None else {
            "trades_per_year": oos_stats.get("turnover_per_year"),
            "avg_holding_days": None,
            "n_trades": None,
        }
        removed = _without_best_year(walk.returns, rf)
        both = pd.concat([current.rename("lead"), walk.returns.rename("book")], axis=1).dropna()
        blend = 0.5 * both["lead"] + 0.5 * both["book"]
        row = {
            "family": family,
            "picks": [[str(date), pick] for date, pick in walk.picks.items()],
            "result": result,
            "last": last,
            "oos": oos_stats,
            "stress": _pack(stress_walk, rf),
            "hold": hold,
            "removed": removed,
            "correlation": float(both["lead"].corr(both["book"])) if len(both) > 5 else None,
            "blend": _pack(blend, rf),
            "validated": bool(result["pass"]),
            "oos_index_start": str(walk.returns.index[0].date()),
            "oos_index_end": str(walk.returns.index[-1].date()),
        }
        ledger.record(
            study=STUDY, family="audit", name=f"{family}_walk_forward",
            params={"last_pick": last, "bar": "materiality_v1"},
            sample_start=row["oos_index_start"], sample_end=row["oos_index_end"],
            metrics={**{k: oos_stats[k] for k in ("cagr", "sharpe", "max_drawdown", "dsr")}, "materiality_pass": row["validated"]},
            counts_for_deflation=False,
        )
        print(
            f"  {family} pick {last} cagr {oos_stats['cagr']:.4f} "
            f"sharpe {oos_stats['sharpe']:.3f} dd {oos_stats['max_drawdown']:.3f} "
            f"validated {row['validated']}",
            flush=True,
        )
        return row

    def rebalance_neighbors(name):
        freq = name
        return [
            item[0] for item in adjacent_cells((freq,), [list(REBALANCE_AXIS)])
        ]

    reb_row = run_walk(
        "rebalance", rebalance_paths, rebalance_turns,
        {freq: rebalance_paths[freq] for freq in rebalance_paths},
        rebalance_neighbors,
        None,
        current_name="quarterly",
    )

    qqq_names = [f"qqq_band{int(round(band * 100)):02d}" for band in QQQ_BANDS]
    qqq_row = run_walk(
        "qqq_trend", qqq_paths, qqq_turns, qqq_paths,
        lambda name: [item[0] for item in adjacent_cells((name,), [qqq_names])],
        None,
    )

    from quantforge.strategies.meanrev import parse_cell as parse_mean

    mean_row = run_walk(
        "meanrev", mean_paths, mean_turns, mean_stress,
        lambda name: [meanrev_name(*item) for item in adjacent_cells(parse_mean(name), [list(ENTRIES), list(HOLDS), list(VEHICLES)])],
        lambda name, index: _active_from_flag(mean_flags[name], index),
    )
    from quantforge.strategies.rotation import parse_cell as parse_rot

    rot_row = run_walk(
        "rotation", rot_paths, rot_turns, rot_stress,
        lambda name: [rotation_name(*item) for item in adjacent_cells(parse_rot(name), [list(EQUITIES), list(LOOKBACKS)])],
        lambda name, index: (rot_picks[name].reindex(index) != CASH).astype(bool),
    )
    from quantforge.strategies.nasdaqvol import parse_cell as parse_nq

    def _nq_flag(name, index):
        weight = nq_weights[name].reindex(index).fillna(0.0)
        return weight > 1e-8

    nq_row = run_walk(
        "nasdaq_vol", nq_paths, nq_turns, nq_stress,
        lambda name: [nasdaq_name(*item) for item in adjacent_cells(parse_nq(name), [list(NASDAQ_BANDS), list(NASDAQ_TARGETS)])],
        _nq_flag,
    )
    btc_row = None
    if btc_paths:
        from quantforge.strategies.btctrend import parse_cell as parse_btc

        btc_row = run_walk(
            "btc_trend", btc_paths, btc_turns, btc_stress,
            lambda name: [btc_name(*item[0]) for item in adjacent_cells((parse_btc(name),), [list(BTC_PAIRS)])],
            lambda name, index: _active_from_flag(btc_flags[name], index),
        )
        picks = {date: name for date, name in btc_row["picks"]}
        crypto_index = btc_paths[btc_row["last"]].loc[pd.Timestamp(btc_row["oos_index_start"]):].index
        crypto_walk = _stitch(btc_crypto, picks, crypto_index).dropna()
        btc_row["crypto_cost"] = _pack(crypto_walk, rf)
        ledger.record(
            study=STUDY, family="audit", name="btc_trend_crypto_cost",
            params={"cell": btc_row["last"], "one_way": CRYPTO_COST},
            sample_start=btc_row["oos_index_start"], sample_end=sample_end,
            metrics=btc_row["crypto_cost"], counts_for_deflation=False,
        )

    # Rebalance and QQQ were recorded twice conceptually: record_path earlier and attach_dsr in run_walk.
    # attach_dsr updates the same trial id. Quarterly was included in run_walk's attach_dsr with counts True.
    # That would add the live quarterly book as a new trial. Fix: the params differ by family so it is a new id.
    # Do not count quarterly. Re-record it as an audit.
    ledger.record(
        study=STUDY, family="rebalance", name="quarterly",
        params={"rebalance": "quarterly", "core": 0.5, "sleeve": 0.5, "band": 0.02, "dip": True, "counts": False},
        sample_start=str(rebalance_paths["quarterly"].index[0].date()), sample_end=sample_end,
        metrics=_pack(rebalance_paths["quarterly"], rf, spy_cc.reindex(rebalance_paths["quarterly"].index), rebalance_turns["quarterly"]),
        counts_for_deflation=False,
    )

    boost_walk = run_walk(
        "spy_rsi_boost", boost_next, boost_turns, boost_next,
        lambda name: [
            boost_name(*item)
            for item in adjacent_cells(parse_boost(name), [list(BOOST_THRESHOLDS), list(BOOST_FRACTIONS), list(BOOST_HOLDS)])
        ],
        lambda name, index: _active_from_flag(boost_flags[name], index),
    )
    replication = _replication(
        named, boost_flags, boost_next, boost_close_paths, boost_turns,
        stand_next.returns.dropna().loc[SAMPLE_START:], stand_close.loc[SAMPLE_START:],
        overlay_next.returns.dropna().loc[SAMPLE_START:], overlay_close.loc[SAMPLE_START:],
        lead_oos, spy_cc, rf, oos_start, n_trials, dispersion, boost_walk,
    )

    kept_rebalance = bool(reb_row["validated"])
    kept_qqq = bool(qqq_row["validated"])
    kept_any = kept_rebalance or kept_qqq

    spec_strategies = {
        "meanrev": _spec_row("meanrev", mean_row),
        "rotation": _spec_row("rotation", rot_row),
        "nasdaq-vol": _spec_row("nasdaq-vol", nq_row),
    }
    if btc_row is not None:
        spec_strategies["btc-trend"] = _spec_row("btc-trend", btc_row)
    spec_strategies["spy-rsi-boost"] = {
        "id": "spy-rsi-boost",
        "cell": named,
        "validated": bool(boost_walk["validated"]),
        "validated_reason": "materiality_bar" if boost_walk["validated"] else "did_not_clear_materiality_bar",
        "oos": _public_oos(replication["named_next_open_oos"]),
        "published_rule": "named_not_walk_forward_pick",
        "walk_forward_pick": boost_walk["last"],
    }
    spec = {
        "selection_year": int(as_of[:4]),
        "sample_end": as_of,
        "next_session": session,
        "lead": {
            "validated": True,
            "validated_reason": "approved_live_book" if not kept_any else "materiality_bar",
            "oos": _public_oos(lead_stats),
            "kept_rebalance": reb_row["last"] if kept_rebalance else None,
            "kept_qqq": qqq_row["last"] if kept_qqq else None,
        },
        "strategies": spec_strategies,
    }
    _write_json(SPEC_PATH, spec)

    signal_rows = []
    fast_payloads = build_fast_signals(panel, spec)
    for key, row in (
        ("meanrev", mean_row),
        ("rotation", rot_row),
        ("nasdaq-vol", nq_row),
    ):
        signal_rows.append(_strategy_result(key, row, fast_payloads[key]))
    if btc_row is not None:
        signal_rows.append(_strategy_result("btc-trend", btc_row, fast_payloads["btc-trend"]))
    signal_rows.append(_strategy_result("spy-rsi-boost", _boost_as_walk(boost_walk, replication), fast_payloads["spy-rsi-boost"]))

    # Lead signal uses the unchanged builder unless a gap fix cleared the bar.
    from quantforge.cli import _signals
    import argparse
    lead_dest = HERE / "signals.json"
    _signals(argparse.Namespace(
        cmd="signals", strategy="trend-core", out=str(lead_dest), cache=None,
        start="1993-01-01", end=None, study_dir=str(HERE),
    ))
    lead_payload = annotate_lead(json.loads(lead_dest.read_text()), spec)
    if kept_any:
        lead_payload["diagnostics"]["gap_fix_kept"] = {
            "rebalance": spec["lead"]["kept_rebalance"],
            "qqq": spec["lead"]["kept_qqq"],
        }
        lead_payload["validated_reason"] = "materiality_bar_recorded_lead_book_not_switched"
        # A passing cell is reported. The weights stay the approved 50/50 book
        # until a kept_* reader is wired, which this run does only after a pass
        # is confirmed in the iteration note. See the decision note.
    write_signal(lead_payload, lead_dest, allowed_tickers=ALLOWED_TREND_TICKERS)
    write_signals_note(HERE / "signals.md", [("signals.json", "Trend-core (2% band, dip overlay on the SPY half)", lead_payload)])
    write_bundle(SIGNAL_DIR, lead_payload, fast_payloads)

    strategy_lines = []
    for row in signal_rows:
        weights = row["signal"]["weights"]
        text = ", ".join(f"{ticker} {weights[ticker]:.0%}" for ticker in sorted(weights))
        strategy_lines.append(f"{row['id']}: hold {text} for {session} ({'validated' if row['validated'] else 'not validated'}).")
    lead_weights = ", ".join(f"{ticker} {value:.0%}" for ticker, value in sorted(lead_payload["weights"].items()))
    headline = f"Lead book: hold {lead_weights} for {session} (approved, trend {'on' if lead_payload['diagnostics'].get('trend_on') else 'off'})."

    verdict = replication["verdict"] + " " + _verdict_text(
        exposure_rows, bil_stats, recent_block, reb_row, qqq_row, signal_rows, as_of,
    )
    recent_note = _recent_text(recent_block, as_of)
    caveats = [
        "Leveraged ETFs before the listing are a daily-reset model: leverage times the underlying, minus financing at the lagged T-bill plus 0.40%, minus the assumed expense ratio. Listed returns replace the model after the first real close-to-close bar. UGL is 2x gold. TMF is 3x TLT. Expense ratios are assumptions (TMF 1.03%, UGL 0.95%), not a prospectus scrape.",
        "Cash is the lagged 13-week bill yield, not the BIL NAV. The signal names that residual BIL.",
        "The exposure rows are fixed rules on the 2006-onward window. They are not a walk-forward selection. The 2.0x row is the live book and is not a new trial. The other rows are.",
        "Mean-reversion entries sit on one ordered list so the materiality bar has neighbors. RSI 5/10/15 is a real step. IBS and the 5-day band are different rules on that list, not a smooth scale.",
        "Deflated Sharpe uses every counting trial in the ledger, including this run. A high full-sample Sharpe can still have a low deflated Sharpe.",
        "The 50/50 blend rebalances every session. That is a research combination, not a cheaper trading rule.",
        "Bitcoin at 5 bps per side is cheaper than a retail coin fill. The 28 bp audit is the repo's Coinbase-like preset and is not a selection trial.",
        "Bill's RSI boost is checked on daily bars. A fill at the signal close is an optimistic stand-in for a 12:50 PT trade. The next open is the conservative fill, and it is the one the signal file uses. Neither fill is the intraday price he sees at 12:50.",
        "The published RSI boost is the named rule (RSI below 10, one third into UPRO, 10-session cap), not the walk-forward pick. The grid is in the ledger so the deflated Sharpe counts those looks.",
        "This is research on public prices. It is not a recommendation to buy leveraged ETFs.",
    ]
    results = {
        "as_of": as_of,
        "next_session": session,
        "expected_close": EXPECTED_CLOSE,
        "close_matches_expected": as_of == EXPECTED_CLOSE,
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "lead_oos": lead_stats,
        "recent": recent_block,
        "recent_note": recent_note,
        "exposure": exposure_rows,
        "rebalance": _change_block(reb_row, "Monthly reset versus the live quarterly book. Daily is the other neighbor."),
        "qqq": _change_block(qqq_row, "50/50 QQQ/TQQQ with the three-down dip on the QQQ half into QLD. Bands are 0%, 2%, and 4%."),
        "replication": replication,
        "strategies": signal_rows,
        "btc_note": btc_note,
        "verdict": verdict,
        "headline": headline,
        "signal_lines": [headline] + strategy_lines,
        "caveats": caveats,
        "lead_signal": {
            "date": lead_payload["date"],
            "weights": lead_payload["weights"],
            "trend_on": lead_payload["diagnostics"].get("trend_on"),
            "dip_on": lead_payload["diagnostics"].get("dip_on"),
            "reference_close": lead_payload["diagnostics"].get("reference_close"),
            "sma": lead_payload["diagnostics"].get("sma"),
        },
    }
    _write_json(OUT_DIR / "results.json", results)
    report = _load_report()
    (OUT_DIR / "RESULTS.md").write_text(report.render_report(results))
    _save_iterations(as_of, results, reb_row, qqq_row, signal_rows, n_trials, dispersion)
    ledger.save()
    print(json.dumps({"as_of": as_of, "session": session, "lines": results["signal_lines"], "kept_rebalance": kept_rebalance, "kept_qqq": kept_qqq}, default=_plain), flush=True)


def _stitch(paths: dict[str, pd.Series], picks: dict[str, str], index: pd.DatetimeIndex) -> pd.Series:
    pieces = []
    labels = list(picks.items())
    for i, (label, name) in enumerate(labels):
        start = pd.Timestamp(label)
        end = pd.Timestamp(labels[i + 1][0]) if i + 1 < len(labels) else None
        series = paths[name].dropna()
        if end is None:
            piece = series[series.index >= start]
        else:
            piece = series[(series.index >= start) & (series.index < end)]
        pieces.append(piece.reindex(index).dropna())
    if not pieces:
        return pd.Series(dtype=float)
    combined = pd.concat(pieces)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined.reindex(index)


def _public_oos(stats: dict) -> dict:
    return {
        "cagr": stats.get("cagr"),
        "sharpe": stats.get("sharpe"),
        "max_drawdown": stats.get("max_drawdown"),
        "worst_year": stats.get("worst_year"),
        "worst_year_return": stats.get("worst_year_return"),
        "start": stats.get("start"),
        "end": stats.get("end"),
        "n_days": stats.get("n_days"),
    }


def _spec_row(strategy_id: str, row: dict) -> dict:
    return {
        "id": strategy_id,
        "cell": row["last"],
        "validated": row["validated"],
        "validated_reason": "materiality_bar" if row["validated"] else "did_not_clear_materiality_bar",
        "oos": _public_oos(row["oos"]),
    }


def _strategy_result(strategy_id: str, row: dict, payload: dict) -> dict:
    weights = payload["weights"]
    text = ", ".join(f"{ticker} {value:.0%}" for ticker, value in sorted(weights.items()))
    levels = payload.get("levels") or {}
    return {
        "id": strategy_id,
        "cell": row["last"],
        "validated": row["validated"],
        "oos": row["oos"],
        "dsr": row["oos"].get("dsr"),
        "cost_10bps": row["stress"],
        "trades_per_year": row["hold"].get("trades_per_year"),
        "avg_holding_days": row["hold"].get("avg_holding_days"),
        "best_year_removed": row["removed"],
        "correlation": row["correlation"],
        "blend": row["blend"],
        "picks": row["picks"],
        "materiality_pass": row["result"]["pass"],
        "checks": row["result"]["checks"],
        "crypto_cost": row.get("crypto_cost"),
        "signal": {
            "date": payload["date"],
            "as_of": payload["as_of"],
            "weights": weights,
            "levels": levels,
            "validated": payload["validated"],
            "pick": (payload.get("diagnostics") or {}).get("pick") or (payload.get("diagnostics") or {}).get("vehicle"),
        },
        "signal_line": (
            f"For {payload['date']}, hold {text}. "
            f"Entry: {levels.get('entry')}. Exit: {levels.get('exit')}. Stop: {levels.get('stop')}."
        ),
    }


def _change_block(row: dict, note: str) -> dict:
    return {
        "decision": "kept" if row["validated"] else "rejected",
        "last_pick": row["last"],
        "note": note,
        "oos": {"current": row["result"]["oos"]["current"], "candidate": row["result"]["oos"]["candidate"]},
        "checks": row["result"]["checks"],
        "validated": row["validated"],
    }


def _boost_as_walk(walk: dict, replication: dict) -> dict:
    """Show the named rule in the strategy table. The walk-forward pick stays in the replication block."""
    row = dict(walk)
    row["last"] = replication["named_cell"]
    oos = dict(replication["named_next_open_oos"])
    oos["dsr"] = replication["vs_spy"]["dsr"]
    row["oos"] = oos
    row["stress"] = {}
    removed = replication["vs_spy"]["without_best_year"]
    row["removed"] = removed
    row["correlation"] = replication["correlation"]
    row["blend"] = replication["blend"]
    row["hold"] = {
        "trades_per_year": replication["windows"]["oos_2006"]["next_open"]["episodes"]["per_year"],
        "avg_holding_days": replication["windows"]["oos_2006"]["next_open"]["episodes"]["avg_days"],
    }
    return row


def _slice_from(returns: pd.Series, start: str) -> pd.Series:
    return returns.loc[pd.Timestamp(start):].dropna()


def _gap_stats(strategy: pd.Series, benchmark: pd.Series, n_trials: int, dispersion: float) -> dict:
    both = pd.concat([strategy.rename("book"), benchmark.rename("base")], axis=1).dropna()
    gap = both["book"] - both["base"]
    yearly = gap.groupby(gap.index.year).sum()
    year = int(yearly.idxmax())
    kept = both.loc[both.index.year != year]
    kept_gap = kept["book"] - kept["base"]
    dummy = pd.Series(0.0, index=kept_gap.index)
    return {
        "dsr": float(deflated_sharpe(gap, n_trials, dispersion, rf=None)),
        "improvement_sharpe": _pack(gap, dummy)["sharpe"],
        "best_year": year,
        "best_year_sum": float(yearly.loc[year]),
        "without_best_year": _pack(kept["book"], dummy),
        "without_best_year_benchmark": _pack(kept["base"], dummy.reindex(kept.index).fillna(0.0)),
        "without_best_year_improvement_sharpe": _pack(kept_gap, dummy.reindex(kept_gap.index).fillna(0.0))["sharpe"],
    }


def _window_row(returns: pd.Series, spy: pd.Series, flag: pd.Series, rf: pd.Series, start: str) -> dict:
    cut = _slice_from(returns, start)
    bench = spy.reindex(cut.index).dropna()
    cut = cut.reindex(bench.index).dropna()
    bench = bench.reindex(cut.index)
    stats = _pack(cut, rf, bench)
    episodes = [
        row for row in boost_episodes(flag.reindex(returns.index).fillna(False), returns, spy.reindex(returns.index))
        if row["start"] >= start
    ]
    years = len(cut) / 252.0
    return {"stats": stats, "episodes": episode_summary(episodes, years), "spy": _pack(bench, rf)}


def _replication(
    named, flags, next_paths, close_paths, turns,
    stand_next, stand_close, overlay_next, overlay_close,
    lead, spy, rf, oos_start, n_trials, dispersion, walk,
) -> dict:
    windows = {
        "oos_2006": str(pd.Timestamp(oos_start).date()),
        "since_2016": "2016-01-01",
        "since_2022": "2022-01-01",
    }
    flag = flags[named]
    table = {}
    for label, start in windows.items():
        table[label] = {
            "start": start,
            "next_open": _window_row(next_paths[named], spy, flag, rf, start),
            "close": _window_row(close_paths[named], spy, flag, rf, start),
            "standalone_next_open": _window_row(stand_next, spy, flag, rf, start),
            "standalone_close": _window_row(stand_close, spy, flag, rf, start),
            "overlay_next_open": _window_row(overlay_next, spy, flag, rf, start),
            "overlay_close": _window_row(overlay_close, spy, flag, rf, start),
        }
    named_oos = _slice_from(next_paths[named], windows["oos_2006"])
    spy_oos = spy.reindex(named_oos.index).dropna()
    named_oos = named_oos.reindex(spy_oos.index)
    lead_aligned = lead.reindex(named_oos.index).dropna()
    both = named_oos.reindex(lead_aligned.index).dropna()
    lead_aligned = lead_aligned.reindex(both.index)
    vs_spy = _gap_stats(named_oos, spy_oos.reindex(named_oos.index), n_trials, dispersion)
    vs_lead = _gap_stats(both, lead_aligned, n_trials, dispersion)
    blend = 0.5 * both + 0.5 * lead_aligned
    text = _replication_text(table, vs_spy, vs_lead, walk)
    return {
        "named_cell": named,
        "walk_forward_pick": walk["last"],
        "validated": walk["validated"],
        "named_next_open_oos": _pack(named_oos, rf, spy_oos.reindex(named_oos.index), turns[named].reindex(named_oos.index)),
        "windows": table,
        "vs_spy": vs_spy,
        "vs_lead": vs_lead,
        "correlation": float(both.corr(lead_aligned)) if len(both) > 5 else None,
        "blend": _pack(blend, rf),
        "verdict": text,
        "bill": {
            "since_2016_cagr": 0.195,
            "since_2016_spy": 0.154,
            "since_2016_maxdd": -0.45,
            "since_2016_spy_maxdd": -0.34,
            "since_2022_cagr": 0.170,
            "since_2022_spy": 0.115,
            "win_rate": 0.72,
            "boosts_per_year": 8.0,
            "avg_days": 3.0,
            "standalone_cagr": 0.115,
        },
    }


def _replication_text(table: dict, vs_spy: dict, vs_lead: dict, walk: dict) -> str:
    y2016 = table["since_2016"]["next_open"]
    y2016_close = table["since_2016"]["close"]
    y2022 = table["since_2022"]["next_open"]
    y2022_close = table["since_2022"]["close"]
    oos = table["oos_2006"]["next_open"]
    oos_close = table["oos_2006"]["close"]
    stand = table["since_2016"]["standalone_next_open"]
    stand_close = table["since_2016"]["standalone_close"]
    # A match has to be close on the compound growth he quoted, not just the same sign.
    close_gap = abs(y2016_close["stats"]["cagr"] - 0.195)
    open_gap = abs(y2016["stats"]["cagr"] - 0.195)
    spy_gap = abs(y2016["spy"]["cagr"] - 0.154)
    if min(close_gap, open_gap) <= 0.015 and spy_gap <= 0.015:
        opener = "Bill's since-2016 growth is close to one of the two daily fills."
    else:
        opener = "Bill's since-2016 numbers are not replicated on daily bars."
    return (
        f"{opener} He quotes 19.5% a year since 2016 against SPY at 15.4%, with a worst drop of -45% against SPY's -34%, "
        f"and 17.0% against 11.5% since 2022. He also quotes 72% winning boosts, about 8 a year, about 3 days each, "
        "and a standalone 3x version around 11-12% a year. "
        f"On these prices, since 2016, the next-open fill compounds at {_fmt_pct(y2016['stats']['cagr'])} "
        f"against close-to-close SPY at {_fmt_pct(y2016['spy']['cagr'])}, "
        f"max drawdown {_fmt_pct(y2016['stats']['max_drawdown'])} against {_fmt_pct(y2016['spy']['max_drawdown'])}. "
        f"The close fill, the optimistic stand-in for a 12:50 PT trade, compounds at {_fmt_pct(y2016_close['stats']['cagr'])} "
        f"with max drawdown {_fmt_pct(y2016_close['stats']['max_drawdown'])}. "
        f"Since 2022 the next-open fill is {_fmt_pct(y2022['stats']['cagr'])} against SPY {_fmt_pct(y2022['spy']['cagr'])}, "
        f"and the close fill is {_fmt_pct(y2022_close['stats']['cagr'])} against SPY {_fmt_pct(y2022_close['spy']['cagr'])}. "
        f"From 2006 the next-open fill is {_fmt_pct(oos['stats']['cagr'])} (max drawdown {_fmt_pct(oos['stats']['max_drawdown'])}, "
        f"worst year {oos['stats']['worst_year']} {_fmt_pct(oos['stats']['worst_year_return'])}) "
        f"and the close fill is {_fmt_pct(oos_close['stats']['cagr'])}. "
        f"Next-open boosts since 2016: win rate {_fmt_pct(y2016['episodes']['win_rate'])} "
        f"({_fmt_pct(y2016['episodes']['beat_spy_rate'])} beat SPY), "
        f"{y2016['episodes']['per_year']:.1f} a year, {y2016['episodes']['avg_days']:.1f} days on average, "
        f"average P&L {_fmt_pct(y2016['episodes']['avg_pnl'])}, worst {_fmt_pct(y2016['episodes']['worst_pnl'])}. "
        f"Close-fill boosts: win rate {_fmt_pct(y2016_close['episodes']['win_rate'])}, "
        f"{y2016_close['episodes']['per_year']:.1f} a year, {y2016_close['episodes']['avg_days']:.1f} days. "
        f"Standalone UPRO-or-cash since 2016 is {_fmt_pct(stand['stats']['cagr'])} next-open and "
        f"{_fmt_pct(stand_close['stats']['cagr'])} at the close. "
        f"The deflated Sharpe of the next-open book's daily gap over SPY, on the 2006-onward window and the full ledger, "
        f"is {vs_spy['dsr']:.3f}. Over the lead book it is {vs_lead['dsr']:.3f}. "
        f"Removing {vs_spy['best_year']}, the year the gap over SPY added the most, leaves a strategy compound growth of "
        f"{_fmt_pct(vs_spy['without_best_year']['cagr'])} against SPY {_fmt_pct(vs_spy['without_best_year_benchmark']['cagr'])}. "
        f"The walk-forward pick on the next-open grid is {walk['last']}; the published signal stays the named rule. "
        "The gap versus his quote is the 12:50 PT price versus a daily close, the fill (this close versus the next open), "
        "UPRO's financing and expense ratio in the stitched series, 5 bps per side, and the rule that an exit close does not re-enter."
    )


def _fmt_pct(value) -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _recent_text(recent: dict, as_of: str) -> str:
    if not recent.get("n_days"):
        return f"No sessions from 2026-10-26 through {as_of}."
    return (
        f"From {recent['start']} through {recent['end']} ({recent['n_days']} sessions in the next-open backtest), "
        f"the stitched lead book returned {_fmt_pct(recent['lead_return'])} and close-to-close SPY returned "
        f"{_fmt_pct(recent['spy_return'])}. The 200-day band was on for {_fmt_pct(recent['trend_on_share'])} of those "
        f"decision days and the three-down dip was on for {_fmt_pct(recent['dip_on_share'])}. "
        "If SPY's mark were $4,927 at the end of this window and both had started together, the book's mark would be "
        f"about ${recent.get('lead_if_spy_ended_at_4927', float('nan')):,.0f}. "
        "That is the backtest's gap, not a replay of the paper account's fills."
    )


def _verdict_text(exposure_rows, bil_stats, recent, reb_row, qqq_row, strategies, as_of) -> str:
    by_name = {row["name"]: row for row in exposure_rows}
    two = by_name["2.0x"]["oos"]
    one = by_name["1.0x"]["oos"]
    trail_1m = (by_name["2.0x"]["trail"].get("1m") or {}).get("trail_rate")
    trail_3m = (by_name["2.0x"]["trail"].get("3m") or {}).get("trail_rate")
    parts = [
        f"Through the {as_of} close, the live 2x book (50% SPY + 50% UPRO, 200-day 2% band, dip on the SPY half) "
        f"compounds at {_fmt_pct(two['cagr'])} out of sample with a Sharpe of {two['sharpe']:.2f}, "
        f"a worst drawdown of {_fmt_pct(two['max_drawdown'])}, and a worst year of {two['worst_year']} "
        f"({_fmt_pct(two['worst_year_return'])}). "
        f"It trails close-to-close SPY in {_fmt_pct(trail_1m)} of rolling 21-session windows and "
        f"{_fmt_pct(trail_3m)} of rolling 63-session windows. "
        f"The unlevered 1.0x mix, which is 100% SPY plus the same dip, compounds at {_fmt_pct(one['cagr'])}. "
        f"SPY/BIL, the band with no leverage and no dip, compounds at {_fmt_pct(bil_stats['cagr'])} "
        f"with a max drawdown of {_fmt_pct(bil_stats['max_drawdown'])}. "
        "The shortfall versus SPY since late October is what a 2x book does in a soft or choppy stretch: "
        "the same down days are larger, and the dip can add SSO on the SPY half while the market is already weak. "
        f"Moving the mix reset from quarterly to the walk-forward rebalance pick ({reb_row['last']}) "
        f"{'clears' if reb_row['validated'] else 'does not clear'} the materiality bar. "
        f"Swapping the vehicles for QQQ/TQQQ (walk-forward pick {qqq_row['last']}) "
        f"{'clears' if qqq_row['validated'] else 'does not clear'} it. "
        "The lead book is unchanged.",
    ]
    for row in strategies:
        oos = row["oos"]
        parts.append(
            f" {row['id']} ({row['cell']}) has an out-of-sample compound growth of {_fmt_pct(oos['cagr'])}, "
            f"Sharpe {oos['sharpe']:.2f}, worst drawdown {_fmt_pct(oos['max_drawdown'])}, "
            f"worst year {oos['worst_year']} ({_fmt_pct(oos['worst_year_return'])}). "
            f"It is {'validated' if row['validated'] else 'not validated'}."
        )
    return "".join(parts)


def _load_report():
    path = OUT_DIR / "report.py"
    spec = importlib.util.spec_from_file_location("faster_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iteration(as_of, title, hypothesis, kept, decision, note, extra) -> dict:
    return {
        "id": f"{as_of}-{extra['slug']}",
        "date": as_of,
        "title": title,
        "hypothesis": hypothesis,
        "status": "tested",
        "kept": kept,
        "decision": decision,
        "decision_note": note,
        **{key: value for key, value in extra.items() if key != "slug"},
    }


def _save_iterations(as_of, results, reb_row, qqq_row, strategies, n_trials, dispersion) -> None:
    gap_kept = bool(reb_row["validated"] or qqq_row["validated"])
    gap_note = results["verdict"]
    entries = [
        _iteration(
            as_of, "Exposure, monthly reset, and QQQ/TQQQ",
            HYPOTHESIS_GAP, gap_kept, "kept" if gap_kept else "rejected", gap_note,
            {
                "slug": "exposure-gap",
                "overlay": "exposure_gap",
                "variants": [
                    {
                        "name": row["name"],
                        "counts_for_deflation": row.get("counts_for_deflation", False),
                        "cagr": (row.get("oos") or {}).get("cagr"),
                        "sharpe": (row.get("oos") or {}).get("sharpe"),
                        "max_drawdown": (row.get("oos") or {}).get("max_drawdown"),
                        "years_selected": 0,
                    }
                    for row in results["exposure"]
                ],
                "oos": {
                    "current": reb_row["result"]["oos"]["current"],
                    "candidate": reb_row["result"]["oos"]["candidate"],
                },
                "oos_window": {
                    "start": results["lead_oos"].get("start"),
                    "end": results["lead_oos"].get("end"),
                    "n_days": results["lead_oos"].get("n_days"),
                },
                "materiality": reb_row["result"],
                "qqq_materiality": qqq_row["result"],
                "n_trials": n_trials,
                "sharpe_dispersion_daily": dispersion,
                "last_pick": reb_row["last"],
            },
        )
    ]
    for row in strategies:
        kept = False
        decision = "rejected"
        note = (
            f"{row['id']} walk-forward cell {row['cell']}: OOS CAGR {_fmt_pct(row['oos']['cagr'])}, "
            f"Sharpe {row['oos']['sharpe']:.3f}, max drawdown {_fmt_pct(row['oos']['max_drawdown'])}, "
            f"worst year {row['oos']['worst_year']} ({_fmt_pct(row['oos']['worst_year_return'])}), "
            f"deflated Sharpe {row['dsr']:.3f}. "
            f"Trades per year {row['trades_per_year']}. Average holding days {row['avg_holding_days']}. "
            f"At 10 bps per side, CAGR {_fmt_pct((row['cost_10bps'] or {}).get('cagr'))}. "
            f"Correlation with the lead book {row['correlation']}. "
            f"A daily 50/50 blend compounds at {_fmt_pct((row['blend'] or {}).get('cagr'))} "
            f"with max drawdown {_fmt_pct((row['blend'] or {}).get('max_drawdown'))}. "
            f"Best year removed ({(row['best_year_removed'] or {}).get('year')}): "
            f"CAGR {_fmt_pct((row['best_year_removed'] or {}).get('cagr'))}. "
            f"Materiality against the lead book: {'pass' if row['validated'] else 'fail'}. "
            "The trend book is not replaced. "
            + row["signal_line"]
        )
        entries.append(_iteration(
            as_of, row["id"], HYPOTHESIS_FAST, kept, decision, note,
            {
                "slug": row["id"],
                "overlay": row["id"],
                "variants": [],
                "oos": {"current": results["lead_oos"], "candidate": row["oos"]},
                "oos_window": {
                    "start": row["oos"].get("start"),
                    "end": row["oos"].get("end"),
                    "n_days": row["oos"].get("n_days"),
                },
                "materiality": {"checks": row["checks"], "pass": row["materiality_pass"]},
                "n_trials": n_trials,
                "last_pick": row["cell"],
                "walk_forward_picks": row["picks"],
            },
        ))
    path = HERE / "iterations.json"
    existing = json.loads(path.read_text())
    ids = {entry["id"] for entry in entries}
    existing = [row for row in existing if row.get("id") not in ids]
    existing.extend(entries)
    path.write_text(json.dumps(existing, indent=2, default=_plain) + "\n")
    results_path = HERE / "results.json"
    stored = json.loads(results_path.read_text())
    stored["iterations"] = existing
    results_path.write_text(json.dumps(stored, indent=2, default=_plain) + "\n")
    spec = importlib.util.spec_from_file_location("spy_upro_report_faster", HERE / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (HERE / "RESULTS.md").write_text(module.render_report(stored))


if __name__ == "__main__":
    main()
