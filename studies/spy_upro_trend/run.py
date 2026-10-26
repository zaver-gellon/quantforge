#!/usr/bin/env python3
"""SPY / leveraged-trend study.

Run from the repo root:

    python studies/spy_upro_trend/run.py

The protocol below is the selection rule. It is not revised after seeing this sample.
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

from quantforge.data import YahooSource, load_bars  # noqa: E402
from quantforge.data.base import DataError  # noqa: E402
from quantforge.engine.backtest import run_backtest, run_core_satellite, run_trend_core  # noqa: E402
from quantforge.engine.costs import CostModel  # noqa: E402
from quantforge.engine.leverage import (  # noqa: E402
    FINANCING_SPREAD,
    LEVERAGED_ETFS,
    stitch_ohlc,
    synthesize_leveraged,
    tracking_stats,
)
from quantforge.engine.schedule import rebalance_mask  # noqa: E402
from quantforge.eval.bootstrap import block_bootstrap  # noqa: E402
from quantforge.eval.ledger import TrialLedger  # noqa: E402
from quantforge.eval.metrics import slice_period, summarize  # noqa: E402
from quantforge.eval.sharpe import (  # noqa: E402
    deflated_sharpe,
    expected_max_z,
    expected_maximum_sharpe,
    probabilistic_sharpe,
)
from quantforge.eval.walkforward import walk_forward_select  # noqa: E402
from quantforge.signals.emit import write_signal, write_signals_note  # noqa: E402
from quantforge.strategies.credit import kept_hedge_stress  # noqa: E402
from quantforge.strategies.volscale import kept_sleeve_scale  # noqa: E402
from quantforge.strategies.dipbuy import DIP_RULES, dip_active  # noqa: E402
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE_START = "2000-01-03"
WF_FIRST_YEAR = 2006
MIN_OBS = 756
BASE_COST = 0.0005
BOOT_PATHS = 20_000
BOOT_BLOCK = 63
BOOT_SEED = 7
STUDY = "spy_upro_trend"

FILTERS = (
    (200, 0.0, "daily"),
    (200, 0.02, "daily"),
    (200, 0.0, "monthly"),
    (150, 0.0, "daily"),
    (100, 0.02, "daily"),
)
SUBPERIODS = (
    ("2000-2012", "2000-01-01", "2012-12-31"),
    ("2013-2026", "2013-01-01", "2026-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2000-2002", "2000-01-01", "2002-12-31"),
    ("2008-2009", "2008-01-01", "2009-12-31"),
    ("2020", "2020-01-01", "2020-12-31"),
)

PROTOCOL_MD = """
## Protocol

Precommitted before this sample was scored:

- Sample: first SPY session on or after 2000-01-03 through the last available session. The SMA may use earlier history. The T-bill input is the 13-week yield (`^IRX`), lagged one session, divided by 100 and by 252.
- Execution: the signal is fixed at the close and filled at the next open. Cash earns the lagged T-bill. SPY, SSO, and UPRO pay 5 bps one-way. Cash does not pay a spread in the base case.
- Leveraged sleeve: daily reset, `r_L = L * r - (L - 1) * (rf + 0.40% / 252) - ER / 252`, with expense ratios 0.89% (SSO) and 0.91% (UPRO). The synthetic open gaps by L times the underlying overnight move. Financing and the expense ratio sit in the rest of the close-to-close return.
- Grid, and every cell is a trial: leverage 2 and 3; sleeve weight 30, 40, 50, 60, and 70 percent; filters SMA 200 daily, SMA 200 daily with a 2% band, SMA 200 evaluated monthly, SMA 150 daily, and SMA 100 daily with a 2% band; mix reset monthly and quarterly. Plus a 100% sleeve for leverage 2 and 3 (the mix reset does nothing when the core weight is zero, so it is stored once). Plus unlevered sleeves of 50% and 100% on the same filters, monthly and quarterly.
- The desk's 50/50 UPRO mix is one of those cells: SMA 200, a 2% hysteresis band (enter when the close is above 1.02 times the average, exit only when it is below 0.98 times, hold the previous state in between), daily filter, quarterly mix. The desk chose that weight after seeing results. This study reports the cell as pre-specified. It does not crown it from the full-sample Sharpe.
- Walk-forward: on the first session of each year from 2006, pick the cell with the highest excess Sharpe versus T-bills using only returns strictly before that session, requiring 756 prior sessions. Ties break alphabetically. The stitched path is the out-of-sample record. It is compared with SPY over those same sessions, not with SPY from 2000.
- The live trend-core book is that 50/50 cell plus a dip overlay on the SPY half only. After three strictly down SPY closes, that half is SSO instead of SPY until the next open after the first up close or the fifth session, whichever comes first. The UPRO half is not resized by the dip. The other dip rules (two-day -2%, two-day -3%, three-day -3%, three-day -4%) are the same overlay on the same 50/50 book, and each rule is a trial.
- Buy-and-hold synthetic SSO and UPRO are trials. Buy-and-hold SPY is the benchmark and is not a trial. Cost and next-close reruns are recorded as audits and do not increment N.
- The walk-forward edge survives only if its CAGR exceeds buy-and-hold SPY over the same sessions (first decision date through the sample end), its information-ratio t-stat exceeds the expected maximum of N independent standard normals, and its deflated Sharpe ratio is at least 0.95. N and the Sharpe dispersion come from the persistent ledger.
- If that bar is cleared, the live book is the variant selected for the current year. If the path beats SPY but misses the bar, the edge does not survive; the flagship file still shows the rule's current weights and is marked not endorsed. If the path does not beat SPY, the recommendation is 100% SPY.
""".strip()

CAVEATS = [
    "Synthetic 3x/2x returns before the ETF listing are a model. The overlap table is the check. Real swap spreads, dividend treatment inside the fund, and creation/redemption can differ from `ER` plus 40 bps over T-bills.",
    "The multiple-testing bar treats trials as independent. These variants share one market, so the bar is conservative and a miss is informative. A pass would be more fragile than the formula suggests.",
    "The 50/50 weight was the desk's post-hoc choice. It is inside the grid so it is counted. It is not re-selected on the full sample.",
    "Block bootstrap redraws historical daily returns. It does not invent a new regime, and it ignores taxes. Trend and dip exits are mostly short-term gains in a taxable account.",
    "A consumer that rebalances to `weights` every session implements the daily-reset mix, not the quarterly drift. `trade` and `rebalance_mix` say when the lower-turnover book would actually trade. `signals.json` is the fixed trend-core book, including the dip overlay on the SPY half. The flagship file is written by the separate return-max study.",
    "This is research on public prices. It is not a recommendation to buy leveraged ETFs.",
]


def _load_report():
    path = Path(__file__).with_name("report.py")
    spec = importlib.util.spec_from_file_location("spy_upro_report", path)
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


def _clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.dropna(subset=["open", "close"]).copy()
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _yield_series(close: pd.Series) -> tuple[pd.Series, str]:
    series = close.dropna().astype(float)
    if series.empty:
        raise DataError("^IRX returned no yields")
    if float(series.median()) < 0.5:
        return series * 100.0, "decimal scaled by 100 to percent"
    if float(series.median()) > 30:
        raise DataError(f"^IRX median {float(series.median()):.2f} does not look like a percent yield")
    return series, "percent"


def _specs() -> list[TrendCoreSpec]:
    specs: list[TrendCoreSpec] = []
    for leverage in (2, 3):
        for sleeve in (0.3, 0.4, 0.5, 0.6, 0.7):
            core = round(1.0 - sleeve, 10)
            for window, band, freq in FILTERS:
                for rebalance in ("monthly", "quarterly"):
                    specs.append(TrendCoreSpec(
                        core_weight=core, sleeve_weight=sleeve, leverage=leverage,
                        window=window, band=band, freq=freq, rebalance=rebalance,
                    ))
        for window, band, freq in FILTERS:
            specs.append(TrendCoreSpec(
                core_weight=0.0, sleeve_weight=1.0, leverage=leverage,
                window=window, band=band, freq=freq, rebalance="quarterly",
            ))
    for sleeve in (0.5, 1.0):
        core = round(1.0 - sleeve, 10)
        for window, band, freq in FILTERS:
            for rebalance in ("monthly", "quarterly"):
                specs.append(TrendCoreSpec(
                    core_weight=core, sleeve_weight=sleeve, leverage=1,
                    window=window, band=band, freq=freq, rebalance=rebalance,
                ))
    names = [spec.name() for spec in specs]
    if len(names) != len(set(names)):
        raise RuntimeError("trend grid produced duplicate names")
    return specs


def _simulate(spec: TrendCoreSpec, bundle, cost: float, execution: str):
    state = bundle["trends"][(spec.window, spec.band, spec.freq)]
    mix = bundle["masks"][spec.rebalance]
    if spec.leverage == 1:
        invested = np.where(state.to_numpy(), spec.core_weight + spec.sleeve_weight, spec.core_weight)
        targets = pd.DataFrame({"SPY": invested}, index=bundle["cal"])
        reb = mix.to_numpy().astype(bool).copy()
        reb[0] = True
        if spec.rebalance == "daily":
            reb[:] = True
        else:
            reb[1:] = reb[1:] | (np.abs(np.diff(invested)) > 1e-12)
        model = CostModel(commission_bps={"SPY": cost * 10_000})
        return run_backtest(
            bundle["spy_open"].to_frame("SPY"),
            bundle["spy_close"].to_frame("SPY"),
            targets,
            pd.Series(reb, index=bundle["cal"]),
            model,
            bundle["rf"],
            execution=execution,
        )
    sleeve = bundle["synth"][spec.leverage]
    return run_core_satellite(
        bundle["spy_open"], bundle["spy_close"], sleeve["open"], sleeve["close"],
        state, mix, spec.core_weight, bundle["rf"],
        core_cost=cost, sleeve_cost=cost, cash_cost=0.0, execution=execution,
    )


def _window(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    out = series.reindex(index).astype(float)
    if out.isna().any():
        raise RuntimeError("return series is missing sessions inside the sample")
    return out


def _cagr(series: pd.Series) -> float:
    if len(series) < 20:
        return float("nan")
    equity = float((1.0 + series).prod())
    if equity <= 0:
        return float("nan")
    return equity ** (252.0 / len(series)) - 1.0


def _period_stats(series: pd.Series, benchmark: pd.Series) -> dict:
    out = {}
    for label, start, end in SUBPERIODS:
        left = slice_period(series, start, end)
        right = slice_period(benchmark, start, end)
        if len(left) < 20 or len(right) < 20:
            out[label] = {"cagr": None, "spy_cagr": None, "excess": None}
            continue
        cagr = _cagr(left)
        spy = _cagr(right.reindex(left.index).dropna())
        # Recompute on the intersection so the excess uses the same days.
        both = left.index.intersection(right.dropna().index)
        cagr = _cagr(left.loc[both])
        spy = _cagr(right.loc[both])
        out[label] = {"cagr": cagr, "spy_cagr": spy, "excess": cagr - spy}
    return out


def _score_factory(rf: pd.Series):
    def score(history: pd.Series) -> float:
        excess = history - rf.reindex(history.index).fillna(0.0)
        deviation = float(excess.std(ddof=1))
        if not math.isfinite(deviation) or deviation == 0.0:
            return -1e9
        return float(excess.mean() / deviation)
    return score


def _year_starts(index: pd.DatetimeIndex, first_year: int) -> list[pd.Timestamp]:
    starts = []
    last_year = int(index[-1].year)
    for year in range(first_year, last_year + 1):
        hits = index[index.year == year]
        if len(hits):
            starts.append(pd.Timestamp(hits[0]))
    return starts


def _attach_inference(metrics: dict, returns: pd.Series, rf: pd.Series, n_trials: int, dispersion: float) -> dict:
    metrics = dict(metrics)
    metrics["psr"] = probabilistic_sharpe(returns, rf=rf)
    metrics["dsr"] = deflated_sharpe(returns, n_trials, dispersion, rf=rf)
    return metrics


def _recommend(wf: dict, spy: dict, bar: float, live_name: str, live_spec: TrendCoreSpec) -> dict:
    beats = wf["cagr"] > spy["cagr"]
    significant = bool(beats and wf["ir_t"] > bar and wf["dsr"] >= 0.95)
    if significant:
        action = "trade_walk_forward"
        variant = live_name
        params = live_spec.params()
        edge = True
    elif beats:
        action = "unproven_walk_forward"
        variant = "BH_SPY"
        params = TrendCoreSpec(core_weight=1.0, sleeve_weight=0.0, leverage=1, band=0.0).params()
        edge = False
    else:
        action = "hold_spy"
        variant = "BH_SPY"
        params = TrendCoreSpec(core_weight=1.0, sleeve_weight=0.0, leverage=1, band=0.0).params()
        edge = False
    # The flagship file shows the rule's current weights when the path beat SPY,
    # including the unproven case. A failed path files 100% SPY.
    if beats:
        signal_params = live_spec.params()
        signal_variant = live_name
    else:
        signal_params = params
        signal_variant = "BH_SPY"
    return {
        "action": action,
        "edge_survives": edge,
        "variant": variant if action == "trade_walk_forward" else ("SPY" if not beats else live_name),
        "live_variant": live_name,
        "signal_variant": signal_variant,
        "signal_params": signal_params,
        "reason": action,
    }


def _validation_row(symbol: str, underlying: pd.DataFrame, actual: pd.DataFrame, irx_pct: pd.Series) -> dict:
    under = _clean_prices(underlying)
    rf = (irx_pct.reindex(under.index).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    meta = LEVERAGED_ETFS[symbol]
    synthetic = synthesize_leveraged(
        under["open"], under["close"], rf, meta["leverage"], meta["expense_ratio"], FINANCING_SPREAD,
    )
    stats = tracking_stats(synthetic["close"], actual["close"])
    stats["etf"] = symbol
    listed = _clean_prices(actual)
    stitched = stitch_ohlc(synthetic["open"], synthetic["close"], listed["open"], listed["close"])
    state = trend_state(under["close"], 200, 0.02, "daily")
    mix = pd.Series(False, index=under.index)
    pure_synth = run_core_satellite(
        under["open"], under["close"], synthetic["open"], synthetic["close"],
        state, mix, 0.0, rf, sleeve_cost=BASE_COST,
    )
    pure_live = run_core_satellite(
        under["open"], under["close"], stitched["open"], stitched["close"],
        state, mix, 0.0, rf, sleeve_cost=BASE_COST,
    )
    if stats.get("start"):
        start = stats["start"]
        stats["trend_synth_cagr"] = _cagr(pure_synth.returns.loc[start:].dropna())
        stats["trend_stitched_cagr"] = _cagr(pure_live.returns.loc[start:].dropna())
    else:
        stats["trend_synth_cagr"] = None
        stats["trend_stitched_cagr"] = None
    return stats


def _overlay(bundle, rule: str, index: pd.DatetimeIndex, sso: pd.DataFrame, upro: pd.DataFrame, cost: float):
    """50/50 trend book; the dip rule swaps only the SPY half into SSO."""
    dip = dip_active(bundle["spy_close"], rule).reindex(bundle["cal"]).fillna(False)
    trend = bundle["trends"][(200, 0.02, "daily")]
    result = run_trend_core(
        bundle["spy_open"], bundle["spy_close"],
        sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, bundle["masks"]["quarterly"], 0.5, 0.5, bundle["rf"],
        spy_cost=cost, sso_cost=cost, upro_cost=cost,
    )
    returns = _window(result.returns, index)
    turnover = result.turnover.reindex(index).fillna(0.0)
    exposure = result.exposure.reindex(index)
    return returns, turnover, exposure


def main() -> None:
    started = time.time()
    print("fetching Yahoo bars", flush=True)
    source = YahooSource()
    frames = load_bars(source, ["SPY", "SSO", "UPRO", "^IRX"], start="1993-01-01")
    optional = {}
    for symbol in ("QQQ", "QLD", "TQQQ"):
        try:
            optional.update(load_bars(source, [symbol], start="1993-01-01"))
        except DataError as exc:
            print(f"optional {symbol} unavailable: {exc}", flush=True)

    spy = _clean_prices(frames["SPY"])
    if spy.index[0] > pd.Timestamp("1999-01-01"):
        raise DataError(f"SPY history starts {spy.index[0].date()}, which is too late for a 200-day SMA before 2000")
    irx_pct, irx_unit = _yield_series(frames["^IRX"]["close"])
    cal = spy.index
    rf = (irx_pct.reindex(cal).ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    rf.index = cal
    bundle = {
        "cal": cal,
        "spy_open": spy["open"],
        "spy_close": spy["close"],
        "rf": rf,
        "trends": {(window, band, freq): trend_state(spy["close"], window, band, freq) for window, band, freq in FILTERS},
        "masks": {freq: rebalance_mask(cal, freq) for freq in ("daily", "monthly", "quarterly")},
        "synth": {},
    }
    for symbol, leverage in (("SSO", 2), ("UPRO", 3)):
        meta = LEVERAGED_ETFS[symbol]
        bundle["synth"][leverage] = synthesize_leveraged(
            spy["open"], spy["close"], rf, leverage, meta["expense_ratio"], FINANCING_SPREAD,
        )
    sample_index = spy["close"].pct_change().loc[SAMPLE_START:].dropna().index
    spy_r = _window(spy["close"].pct_change(), sample_index)
    print(f"sample {sample_index[0].date()} -> {sample_index[-1].date()} ({len(sample_index)} days)", flush=True)

    specs = _specs()
    by_name = {spec.name(): spec for spec in specs}
    desk = TrendCoreSpec()
    if desk.name() not in by_name:
        raise RuntimeError(f"desk spec {desk.name()} is not in the grid")
    returns: dict[str, pd.Series] = {}
    turnover: dict[str, pd.Series] = {}
    exposure: dict[str, pd.Series] = {}
    for i, spec in enumerate(specs, start=1):
        result = _simulate(spec, bundle, BASE_COST, "next_open")
        returns[spec.name()] = _window(result.returns, sample_index)
        turnover[spec.name()] = result.turnover.reindex(sample_index).fillna(0.0)
        exposure[spec.name()] = result.exposure.reindex(sample_index)
        if i % 25 == 0:
            print(f"simulated {i}/{len(specs)}", flush=True)
    print(f"grid done in {time.time() - started:.0f}s", flush=True)

    bh = {
        "BH_SPY": spy_r,
        "BH_SSO_synth": _window(bundle["synth"][2]["close"].pct_change(), sample_index),
        "BH_UPRO_synth": _window(bundle["synth"][3]["close"].pct_change(), sample_index),
    }
    decisions = _year_starts(sample_index, WF_FIRST_YEAR)
    walk = walk_forward_select(returns, decisions, _score_factory(rf), min_obs=MIN_OBS)
    pieces = []
    labels = list(walk.picks)
    for i, label in enumerate(labels):
        start = pd.Timestamp(label)
        stop = pd.Timestamp(labels[i + 1]) if i + 1 < len(labels) else None
        held = turnover[walk.picks[label]]
        if stop is None:
            pieces.append(held.loc[start:])
        else:
            pieces.append(held.loc[(held.index >= start) & (held.index < stop)])
    walk_turn = pd.concat(pieces)
    walk_turn = walk_turn[~walk_turn.index.duplicated(keep="first")].reindex(walk.returns.index).fillna(0.0)

    sso_px = bundle["synth"][2]
    upro_px = bundle["synth"][3]
    dip_returns = {}
    dip_turn = {}
    dip_exposure = {}
    for rule in DIP_RULES:
        series, traded, exposed = _overlay(bundle, rule, sample_index, sso_px, upro_px, BASE_COST)
        key = "dip_" + rule.replace("%", "pct").replace("<", "lt").replace("-", "")
        dip_returns[key] = series
        dip_turn[key] = traded
        dip_exposure[key] = exposed
    print(f"dips done in {time.time() - started:.0f}s", flush=True)

    metrics = {}
    for name, series in {**returns, **bh, "walk_forward": walk.returns, **dip_returns}.items():
        traded = turnover.get(name)
        if name == "walk_forward":
            traded = walk_turn
        if name in dip_turn:
            traded = dip_turn[name]
        if name in bh:
            traded = pd.Series(0.0, index=series.index)
        exposed = exposure.get(name)
        if name in dip_exposure:
            exposed = dip_exposure[name]
        metrics[name] = summarize(series, rf=rf.reindex(series.index), benchmark=spy_r.reindex(series.index), turnover=traded, exposure=exposed)

    ledger = TrialLedger(ROOT / "studies" / "trial_ledger.json")
    sample_end = str(sample_index[-1].date())
    for spec in specs:
        ledger.record(
            study=STUDY, family="trend_grid", name=spec.name(), params=spec.params(),
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[spec.name()],
            counts_for_deflation=True,
        )
    for name in ("BH_SSO_synth", "BH_UPRO_synth"):
        ledger.record(
            study=STUDY, family="buy_hold_levered", name=name,
            params={"instrument": name, "synthetic": True},
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[name],
            counts_for_deflation=True,
        )
    ledger.record(
        study=STUDY, family="benchmark", name="BH_SPY", params={"instrument": "SPY"},
        sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics["BH_SPY"],
        counts_for_deflation=False,
    )
    for key, rule in zip(dip_returns, DIP_RULES):
        ledger.record(
            study=STUDY, family="dip_buy", name=key,
            params={
                "rule": rule, "max_hold": 5, "entry": "next_open", "applies_to": "spy_half",
                "sleeve": "UPRO", "band": 0.02, "core_weight": 0.5,
            },
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[key],
            counts_for_deflation=True,
        )
    n_trials = ledger.n_trials()
    dispersion = ledger.sharpe_dispersion()
    for name, series in {**returns, **bh, "walk_forward": walk.returns, **dip_returns}.items():
        metrics[name] = _attach_inference(metrics[name], series, rf.reindex(series.index), n_trials, dispersion)
    for spec in specs:
        ledger.record(
            study=STUDY, family="trend_grid", name=spec.name(), params=spec.params(),
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[spec.name()],
        )
    for name in ("BH_SSO_synth", "BH_UPRO_synth"):
        ledger.record(
            study=STUDY, family="buy_hold_levered", name=name, params={"instrument": name, "synthetic": True},
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[name],
        )
    for key, rule in zip(dip_returns, DIP_RULES):
        ledger.record(
            study=STUDY, family="dip_buy", name=key,
            params={
                "rule": rule, "max_hold": 5, "entry": "next_open", "applies_to": "spy_half",
                "sleeve": "UPRO", "band": 0.02, "core_weight": 0.5,
            },
            sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics[key],
        )
    ledger.record(
        study=STUDY, family="benchmark", name="BH_SPY", params={"instrument": "SPY"},
        sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics["BH_SPY"],
        counts_for_deflation=False,
    )

    # Stitched desk mix: listed UPRO once it has a previous close, synthetic before that.
    upro_actual = _clean_prices(frames["UPRO"]).reindex(cal)
    stitched_upro = stitch_ohlc(
        bundle["synth"][3]["open"], bundle["synth"][3]["close"], upro_actual["open"], upro_actual["close"],
    )
    desk_state = bundle["trends"][(desk.window, desk.band, desk.freq)]
    desk_mix = bundle["masks"][desk.rebalance]
    stitched_run = run_core_satellite(
        bundle["spy_open"], bundle["spy_close"], stitched_upro["open"], stitched_upro["close"],
        desk_state, desk_mix, desk.core_weight, rf, core_cost=BASE_COST, sleeve_cost=BASE_COST,
    )
    sso_actual = _clean_prices(frames["SSO"]).reindex(cal)
    stitched_sso = stitch_ohlc(
        bundle["synth"][2]["open"], bundle["synth"][2]["close"], sso_actual["open"], sso_actual["close"],
    )
    core_r, core_turn, core_exp = _overlay(
        bundle, "3_down_closes", sample_index, stitched_sso, stitched_upro, BASE_COST,
    )
    metrics["trend_core"] = metrics["dip_3_down_closes"]
    metrics["trend_core_stitched"] = _attach_inference(
        summarize(
            core_r, rf=rf.reindex(core_r.index), benchmark=spy_r,
            turnover=core_turn, exposure=core_exp,
        ),
        core_r, rf.reindex(core_r.index), n_trials, dispersion,
    )
    ledger.record(
        study=STUDY, family="audit", name="trend_core_stitched",
        params={"base": "dip_3_down_closes", "prices": "synthetic_then_listed_SSO_and_UPRO"},
        sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics["trend_core_stitched"],
        counts_for_deflation=False,
    )
    stitched_r = _window(stitched_run.returns, sample_index)
    metrics["desk_50_50_stitched"] = _attach_inference(
        summarize(
            stitched_r, rf=rf.reindex(stitched_r.index), benchmark=spy_r,
            turnover=stitched_run.turnover.reindex(sample_index).fillna(0.0),
            exposure=stitched_run.exposure.reindex(sample_index),
        ),
        stitched_r, rf.reindex(stitched_r.index), n_trials, dispersion,
    )
    ledger.record(
        study=STUDY, family="audit", name="desk_50_50_stitched",
        params={"base": desk.name(), "prices": "synthetic_then_listed_UPRO"},
        sample_start=SAMPLE_START, sample_end=sample_end, metrics=metrics["desk_50_50_stitched"],
        counts_for_deflation=False,
    )

    validation = [
        _validation_row("SSO", frames["SPY"], frames["SSO"], irx_pct),
        _validation_row("UPRO", frames["SPY"], frames["UPRO"], irx_pct),
    ]
    if "QQQ" in optional and "QLD" in optional:
        validation.append(_validation_row("QLD", optional["QQQ"], optional["QLD"], irx_pct))
    if "QQQ" in optional and "TQQQ" in optional:
        validation.append(_validation_row("TQQQ", optional["QQQ"], optional["TQQQ"], irx_pct))
    for row in validation:
        print(
            f"validate {row['etf']}: corr {row.get('daily_corr')} gap {row.get('cagr_gap')}",
            flush=True,
        )

    live_label = list(walk.picks)[-1]
    live_name = walk.picks[live_label]
    live_spec = by_name[live_name]
    bar = expected_max_z(n_trials)
    spy_window = spy_r.reindex(walk.returns.index)
    metrics["BH_SPY_walk_window"] = _attach_inference(
        summarize(spy_window, rf=rf.reindex(spy_window.index)),
        spy_window,
        rf.reindex(spy_window.index),
        n_trials,
        dispersion,
    )
    ledger.record(
        study=STUDY, family="benchmark", name="BH_SPY_walk_window",
        params={"instrument": "SPY", "window": "walk_forward"},
        sample_start=str(spy_window.index[0].date()), sample_end=sample_end,
        metrics=metrics["BH_SPY_walk_window"], counts_for_deflation=False,
    )
    recommendation = _recommend(
        metrics["walk_forward"], metrics["BH_SPY_walk_window"], bar, live_name, live_spec,
    )

    headline_names = [
        ("BH_SPY", "BH_SPY"),
        ("BH_SPY_walk_window", "BH_SPY_walk_window"),
        ("BH_SSO_synth", "BH_SSO_synth"),
        ("BH_UPRO_synth", "BH_UPRO_synth"),
        ("pure_UPRO_sma200_band2", "core0_slv100_L3_sma200_band2_daily_quarterly"),
        ("pure_SSO_sma200_band2", "core0_slv100_L2_sma200_band2_daily_quarterly"),
        ("unlev_sma200_band2", "core0_slv100_L1_sma200_band2_daily_quarterly"),
        ("desk_50_50", desk.name()),
        ("desk_50_50_stitched", "desk_50_50_stitched"),
        ("trend_core", "trend_core"),
        ("trend_core_stitched", "trend_core_stitched"),
        ("walk_forward", "walk_forward"),
        ("dip_3_down_closes", "dip_3_down_closes"),
    ]
    headline = [(label, metrics[key]) for label, key in headline_names]

    sub_names = {
        "BH_SPY": bh["BH_SPY"],
        "desk_50_50": returns[desk.name()],
        "desk_50_50_stitched": stitched_r,
        "trend_core": dip_returns["dip_3_down_closes"],
        "trend_core_stitched": core_r,
        "walk_forward": walk.returns,
        "pure_UPRO_sma200_band2": returns["core0_slv100_L3_sma200_band2_daily_quarterly"],
        "dip_3_down_closes": dip_returns["dip_3_down_closes"],
    }
    subperiods = {name: _period_stats(series, spy_r) for name, series in sub_names.items()}
    spy_sub = [(label, _period_stats(spy_r, spy_r)[label]["cagr"]) for label, _, _ in SUBPERIODS]

    ranked = sorted(returns, key=lambda name: metrics[name]["sharpe"], reverse=True)[:8]
    full_sample_top = []
    for name in ranked:
        excess = _period_stats(returns[name], spy_r)["2013-2026"]["excess"]
        full_sample_top.append({
            "name": name,
            "cagr": metrics[name]["cagr"],
            "sharpe": metrics[name]["sharpe"],
            "max_drawdown": metrics[name]["max_drawdown"],
            "ir_t": metrics[name]["ir_t"],
            "excess_2013": excess,
        })

    print("bootstrap", flush=True)
    boot_series = {
        "BH_SPY": bh["BH_SPY"],
        "desk_50_50": returns[desk.name()],
        "trend_core": dip_returns["dip_3_down_closes"],
        "walk_forward": walk.returns,
        "pure_UPRO_sma200_band2": returns["core0_slv100_L3_sma200_band2_daily_quarterly"],
        "dip_3_down_closes": dip_returns["dip_3_down_closes"],
    }
    bootstrap = {}
    for name, series in boot_series.items():
        bootstrap[name] = {
            "1y": block_bootstrap(series, 252, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_r),
            "5y": block_bootstrap(series, 252 * 5, n_paths=BOOT_PATHS, block=BOOT_BLOCK, seed=BOOT_SEED, benchmark=spy_r),
        }

    print("cost and fill stress", flush=True)
    stress_specs = {
        "desk_50_50": desk,
        "pure_UPRO_sma200_band2": by_name["core0_slv100_L3_sma200_band2_daily_quarterly"],
        "live_walk_forward_cell": live_spec,
    }
    stress = {}
    for label, spec in stress_specs.items():
        row = {}
        for bps, fraction in ((0, 0.0), (3, 0.0003), (5, BASE_COST), (10, 0.001)):
            run = _simulate(spec, bundle, fraction, "next_open")
            row[f"cost_{bps}"] = _cagr(_window(run.returns, sample_index))
        close_fill = _simulate(spec, bundle, BASE_COST, "next_close")
        row["next_open"] = row["cost_5"]
        row["next_close"] = _cagr(_window(close_fill.returns, sample_index))
        stress[label] = row
        ledger.record(
            study=STUDY, family="audit", name=f"{label}_cost_stress",
            params={"spec": spec.name(), "costs_bps": [0, 3, 5, 10], "fills": ["next_open", "next_close"]},
            sample_start=SAMPLE_START, sample_end=sample_end,
            metrics={"cagr_by_cost_bps": {str(k): row[f"cost_{k}"] for k in (0, 3, 5, 10)}, "next_close_cagr": row["next_close"]},
            counts_for_deflation=False,
        )
    # Walk-forward path under the base decisions, re-simulated at 10 bps and at the next close.
    def _restitch(cost: float, execution: str) -> pd.Series:
        alt = {}
        for name in set(walk.picks.values()):
            run = _simulate(by_name[name], bundle, cost, execution)
            alt[name] = _window(run.returns, sample_index)
        pieces_ = []
        for i, label in enumerate(labels):
            start = pd.Timestamp(label)
            stop = pd.Timestamp(labels[i + 1]) if i + 1 < len(labels) else None
            held = alt[walk.picks[label]]
            pieces_.append(held.loc[start:] if stop is None else held.loc[(held.index >= start) & (held.index < stop)])
        return pd.concat(pieces_)
    wf_stress = _restitch(0.001, "next_open")
    wf_close = _restitch(BASE_COST, "next_close")
    stress["walk_forward"] = {
        "cost_0": None,
        "cost_3": None,
        "cost_5": metrics["walk_forward"]["cagr"],
        "cost_10": _cagr(wf_stress),
        "next_open": metrics["walk_forward"]["cagr"],
        "next_close": _cagr(wf_close),
    }

    frequency = sorted(
        ((name, sum(pick == name for pick in walk.picks.values())) for name in set(walk.picks.values())),
        key=lambda item: (-item[1], item[0]),
    )
    ledger.save()

    as_of = spy.index[-1]
    hedge_stress, hedge_rule, hedge_info = kept_hedge_stress(HERE / "iterations.json", spy.index, as_of=as_of)
    sleeve_scale, vol_info = kept_sleeve_scale(HERE / "iterations.json", spy["close"], as_of=as_of)
    desk_signal = build_trend_signal(
        desk, spy["close"], as_of=as_of, strategy_id="trend_core",
        hedge_on=hedge_stress, hedge_rule=hedge_rule, hedge_info=hedge_info,
        sleeve_scale=sleeve_scale, vol_info=vol_info,
    )
    write_signal(desk_signal, HERE / "signals.json")
    write_signals_note(
        HERE / "signals.md",
        [
            ("signals.json", "Trend-core (2% band, dip overlay on the SPY half)", desk_signal),
        ],
    )
    (HERE / "signals.md").write_text(
        (HERE / "signals.md").read_text()
        + "\n`signals_flagship.json` is written by `python studies/flagship/run.py` and is not this fixed book.\n"
    )

    dip_table = [(key, metrics[key]) for key in dip_returns]
    results = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "protocol_md": PROTOCOL_MD,
        "caveats": CAVEATS,
        "sample": {
            "start": str(sample_index[0].date()),
            "end": str(sample_index[-1].date()),
            "n_days": int(len(sample_index)),
            "years": len(sample_index) / 252.0,
        },
        "costs_one_way_bps": BASE_COST * 10_000,
        "financing_spread": FINANCING_SPREAD,
        "irx_unit": irx_unit,
        "n_trials": n_trials,
        "sharpe_dispersion_daily": dispersion,
        "sr0_daily": expected_maximum_sharpe(n_trials, dispersion),
        "multiple_testing_bar_t": bar,
        "validation": validation,
        "headline": headline,
        "walk_forward": {
            "picks": list(walk.picks.items()),
            "frequency": frequency,
            "live_variant": live_name,
            "live_decision": live_label,
        },
        "subperiod_labels": [label for label, _, _ in SUBPERIODS],
        "subperiods": subperiods,
        "spy_subperiods": spy_sub,
        "full_sample_top": full_sample_top,
        "bootstrap": bootstrap,
        "bootstrap_block": BOOT_BLOCK,
        "bootstrap_paths": BOOT_PATHS,
        "bootstrap_seed": BOOT_SEED,
        "stress": stress,
        "dip": dip_table,
        "recommendation": recommendation,
        "variants": {name: metrics[name] for name in list(returns) + list(bh) + ["walk_forward", "desk_50_50_stitched"] + list(dip_returns)},
    }
    iter_path = HERE / "iterations.json"
    if iter_path.exists():
        results["iterations"] = json.loads(iter_path.read_text())
    _write_json(HERE / "results.json", results)
    report = _load_report().render_report(results)
    (HERE / "RESULTS.md").write_text(report)
    print(
        f"edge_survives={recommendation['edge_survives']} action={recommendation['action']} "
        f"live={live_name} desk_cagr={metrics[desk.name()]['cagr']:.4f} "
        f"wf_cagr={metrics['walk_forward']['cagr']:.4f} spy_cagr={metrics['BH_SPY']['cagr']:.4f} "
        f"n={n_trials} bar={bar:.2f} ({time.time() - started:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
