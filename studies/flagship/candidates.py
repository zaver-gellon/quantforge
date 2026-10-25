"""Pre-specified return-max candidates. Nothing here looks at a future close."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantforge.engine.backtest import BacktestResult, run_backtest, run_trend_core
from quantforge.engine.costs import crypto_costs, equity_costs
from quantforge.engine.leverage import FINANCING_SPREAD, synthesize_leveraged
from quantforge.engine.overlay import half_kelly_exposure, scale_exposure, vol_target_exposure
from quantforge.engine.schedule import rebalance_mask
from quantforge.strategies.basket import (
    donchian_state,
    equal_active_weights,
    long_short_weights,
    momentum_weights,
    risk_parity_weights,
    trend_panel,
)
from quantforge.strategies.dipbuy import dip_active
from quantforge.strategies.trend import trend_state

# Listed products used when a candidate is leveraged. Names without an entry
# stay on the underlying and are labeled as synthetic in the signal note.
LISTED_LEVERAGE = {
    ("SPY", 2): "SSO",
    ("SPY", 3): "UPRO",
    ("QQQ", 2): "QLD",
    ("QQQ", 3): "TQQQ",
    ("IWM", 2): "UWM",
    ("IWM", 3): "TNA",
    ("EFA", 2): "EFO",
    ("EFA", 3): "DZK",
    ("EEM", 2): "EET",
    ("EEM", 3): "EDC",
    ("TLT", 2): "UBT",
    ("TLT", 3): "TMF",
    ("GLD", 2): "UGL",
}

ER = {
    ("SPY", 2): 0.0089,
    ("SPY", 3): 0.0091,
    ("QQQ", 2): 0.0095,
    ("QQQ", 3): 0.0084,
}


def expense_ratio(symbol: str, leverage: int) -> float:
    return ER.get((symbol, leverage), 0.0095)


@dataclass
class Candidate:
    name: str
    family: str
    params: dict
    returns: pd.Series
    weights: dict
    turnover_per_year: float | None = None
    eligible: bool = True
    counts: bool = True
    note: str = ""
    extras: dict = field(default_factory=dict)


def _last_weights(weights: pd.DataFrame, scale: float = 1.0) -> dict:
    if weights is None or weights.empty:
        return {}
    row = weights.iloc[-1]
    out = {}
    for ticker, value in row.items():
        scaled = float(value) * scale
        if abs(scaled) > 1e-6:
            out[str(ticker)] = round(scaled, 6)
    return out


def _run_weights(opens, closes, weights, rf, max_gross, allow_short=False, cost_bps=5.0, crypto=False):
    symbols = list(opens.columns)
    cost = crypto_costs(symbols) if crypto else equity_costs(symbols, commission_bps=cost_bps)
    frame_w = weights.reindex(index=opens.index, columns=symbols).fillna(0.0)
    gross = frame_w.abs().sum(axis=1)
    overflow = gross > max_gross
    if bool(overflow.any()):
        frame_w.loc[overflow] = frame_w.loc[overflow].div(gross[overflow] / max_gross, axis=0)
    reb = rebalance_mask(opens.index, "monthly")
    borrow = (FINANCING_SPREAD / 252.0) if max_gross > 1 or allow_short else 0.0
    return run_backtest(
        opens, closes, frame_w, reb, cost, rf.reindex(opens.index).fillna(0.0),
        max_gross=max(max_gross, 1.0), allow_short=allow_short, borrow_spread=borrow,
    )


def _candidate(name, family, params, result, weights, eligible=True, counts=True, note="", scale=1.0, extras=None) -> Candidate:
    years = max(result.returns.dropna().shape[0] / 252.0, 1e-9)
    return Candidate(
        name=name,
        family=family,
        params=params,
        returns=result.returns,
        weights=_last_weights(weights, scale),
        turnover_per_year=float(result.turnover.fillna(0.0).sum() / years),
        eligible=eligible,
        counts=counts,
        note=note,
        extras=extras or {},
    )


def trend_core_book(spy_open, spy_close, rf) -> Candidate:
    sso = synthesize_leveraged(spy_open, spy_close, rf, 2, expense_ratio("SPY", 2), FINANCING_SPREAD)
    upro = synthesize_leveraged(spy_open, spy_close, rf, 3, expense_ratio("SPY", 3), FINANCING_SPREAD)
    trend = trend_state(spy_close, 200, 0.02, "daily")
    dip = dip_active(spy_close, "3_down_closes")
    mix = rebalance_mask(spy_close.index, "quarterly")
    result = run_trend_core(
        spy_open, spy_close, sso["open"], sso["close"], upro["open"], upro["close"],
        trend, dip, mix, 0.5, 0.5, rf, spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005,
    )
    # Strategic weights, not the drifted book. The signal is the next session's target.
    on = bool(trend.iloc[-1])
    dipping = bool(dip.iloc[-1])
    weights = {"SSO" if dipping else "SPY": 0.5, "UPRO" if on else "BIL": 0.5}
    merged: dict[str, float] = {}
    for ticker, value in weights.items():
        merged[ticker] = round(merged.get(ticker, 0.0) + value, 6)
    frame = pd.DataFrame([merged])
    return _candidate(
        "trend_core", "trend_core",
        {"band": 0.02, "window": 200, "dip": "3_down_closes", "core": 0.5, "sleeve": "UPRO"},
        result, frame, note="SPY half dips to SSO; UPRO half follows the 2% band.",
    )


def cta_books(opens: pd.DataFrame, closes: pd.DataFrame, rf: pd.Series) -> list[Candidate]:
    out = []
    for window, band, leverage in (
        (100, 0.0, 1), (100, 0.02, 1), (200, 0.0, 1), (200, 0.02, 1),
        (100, 0.02, 2), (200, 0.02, 2), (200, 0.0, 2),
        (100, 0.02, 3), (200, 0.02, 3), (200, 0.0, 3),
    ):
        trend = trend_panel(closes, window, band, "monthly")
        if leverage == 1:
            px_o, px_c = opens, closes
            active = trend
        else:
            px_o, px_c, active = _levered_panel(opens, closes, rf, trend, leverage)
        weights = equal_active_weights(active, 1.0)
        result = _run_weights(px_o, px_c, weights, rf, max_gross=1.0)
        name = f"cta_L{leverage}_sma{window}_band{int(band * 100)}"
        out.append(_candidate(
            name, "cta_long_flat",
            {"window": window, "band": band, "leverage": leverage, "rebalance": "monthly", "direction": "long_flat"},
            result, weights,
            note="Equal weight in the trending listed or synthetic leveraged ETFs. Flat names are cash.",
        ))
    for leverage in (1, 2, 3):
        trend = trend_panel(closes, 200, 0.02, "monthly")
        weights = long_short_weights(trend, gross=float(leverage))
        result = _run_weights(opens, closes, weights, rf, max_gross=float(leverage), allow_short=True)
        out.append(_candidate(
            f"cta_ls_G{leverage}_sma200_band2", "cta_long_short",
            {"window": 200, "band": 0.02, "gross": leverage, "direction": "long_short", "rebalance": "monthly"},
            result, weights,
            note="Margin long/short, not a listed inverse ETF. Shorts pay the financing spread.",
        ))
    return out


def _levered_panel(opens, closes, rf, trend, leverage: int):
    px_o = {}
    px_c = {}
    active = {}
    for symbol in closes.columns:
        synth = synthesize_leveraged(
            opens[symbol], closes[symbol], rf.reindex(closes.index).fillna(0.0),
            leverage, expense_ratio(symbol, leverage), FINANCING_SPREAD,
        )
        ticker = LISTED_LEVERAGE.get((symbol, leverage), f"{symbol}{leverage}X")
        px_o[ticker] = synth["open"]
        px_c[ticker] = synth["close"]
        active[ticker] = trend[symbol]
    return pd.DataFrame(px_o), pd.DataFrame(px_c), pd.DataFrame(active)


def risk_parity_books(opens, closes, rf) -> list[Candidate]:
    out = []
    trend = trend_panel(closes, 200, 0.02, "monthly")
    for gate, gross in ((False, 1), (True, 1), (True, 2), (True, 3), (False, 2)):
        weights = risk_parity_weights(closes, 63, float(gross), trend if gate else None)
        result = _run_weights(opens, closes, weights, rf, max_gross=float(gross))
        name = f"rp_{'trend' if gate else 'always'}_G{gross}"
        out.append(_candidate(
            name, "risk_parity",
            {"vol_window": 63, "trend": gate, "gross": gross, "trend_window": 200, "band": 0.02},
            result, weights,
            note="Inverse-vol across the basket. Gross above 1 is financed margin, reset monthly.",
        ))
    return out


def momentum_books(opens, closes, rf, family: str, lookbacks=(126, 252), tops=(2, 3), grosses=(1, 2), eligible=True, note="") -> list[Candidate]:
    out = []
    for lookback in lookbacks:
        for top_n in tops:
            for gross in grosses:
                weights = momentum_weights(closes, lookback, 21, top_n, float(gross), absolute=True)
                result = _run_weights(opens, closes, weights, rf, max_gross=float(gross))
                name = f"{family}_top{top_n}_lb{lookback}_G{gross}"
                out.append(_candidate(
                    name, family,
                    {"lookback": lookback, "skip": 21, "top_n": top_n, "gross": gross, "absolute": True},
                    result, weights, eligible=eligible, note=note,
                ))
    return out


def crypto_books(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, rf: pd.Series) -> list[Candidate]:
    out = []
    aligned = {}
    for symbol, frame in frames.items():
        px = _crypto_on_sessions(frame, calendar)
        if px is not None and len(px) > 400:
            aligned[symbol] = px
    if "BTC-USD" not in aligned:
        return out
    specs = [("BTC-USD", "sma")]
    specs.append(("BTC-USD", "donchian"))
    if "ETH-USD" in aligned:
        specs.append(("ETH-USD", "sma"))
    for symbol, kind in specs:
        px = aligned[symbol]
        state = _crypto_state(px["close"], kind).reindex(px.index).fillna(False)
        weights = pd.DataFrame({symbol: state.astype(float)}, index=px.index)
        opens = px[["open"]].rename(columns={"open": symbol})
        closes = px[["close"]].rename(columns={"close": symbol})
        result = _run_weights(opens, closes, weights, rf.reindex(px.index).fillna(0.0), max_gross=1.0, crypto=True)
        name = f"crypto_{symbol.split('-')[0].lower()}_{kind}"
        out.append(_candidate(
            name, "crypto",
            {"symbol": symbol, "rule": kind, "vol_target": None},
            result, weights,
            note="Daily Coinbase mark on the NYSE calendar. Costs are 25 bps plus 3 bps of slippage.",
        ))
        if symbol == "BTC-USD":
            # Scale the untargeted sleeve. Reusing ``result.returns`` after the
            # first target would compound the overlay.
            base_returns = result.returns
            for target in (0.25, 0.40):
                exposure = vol_target_exposure(base_returns, target, window=63, cap=2.0)
                scaled = scale_exposure(
                    base_returns, exposure, rf.reindex(base_returns.index).fillna(0.0),
                    borrow_spread=FINANCING_SPREAD / 252.0, cost=0.0028,
                )
                scale = float(exposure.iloc[-1]) if len(exposure) else 1.0
                wrapped = BacktestResult(
                    returns=scaled,
                    turnover=result.turnover,
                    cost_drag=result.cost_drag,
                    exposure=result.exposure,
                )
                out.append(_candidate(
                    f"{name}_vol{int(target * 100)}", "crypto_vol_target",
                    {"symbol": symbol, "rule": kind, "vol_target": target, "cap": 2.0},
                    wrapped, weights, scale=scale,
                    note="Trailing 63-day vol target, capped at 2x, reset monthly. This scales the already-costed sleeve.",
                ))
    if "BTC-USD" in aligned and "ETH-USD" in aligned:
        both_o = pd.DataFrame({k: aligned[k]["open"] for k in ("BTC-USD", "ETH-USD")}).dropna()
        both_c = pd.DataFrame({k: aligned[k]["close"] for k in ("BTC-USD", "ETH-USD")}).reindex(both_o.index)
        state = pd.DataFrame({k: _crypto_state(both_c[k], "sma") for k in both_c.columns})
        weights = equal_active_weights(state, 1.0)
        result = _run_weights(both_o, both_c, weights, rf.reindex(both_o.index).fillna(0.0), max_gross=1.0, crypto=True)
        out.append(_candidate(
            "crypto_basket_sma", "crypto",
            {"symbols": ["BTC-USD", "ETH-USD"], "rule": "sma", "band": 0.02},
            result, weights,
            note="Equal weight in BTC and ETH while each is above its own 200-day average by the 2% band.",
        ))
    return out


def _crypto_state(close: pd.Series, kind: str) -> pd.Series:
    if kind == "sma":
        return trend_state(close, 200, 0.02, "daily")
    if kind == "donchian":
        return donchian_state(close, 55, 20)
    raise ValueError(kind)


def _crypto_on_sessions(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame | None:
    px = frame.dropna(subset=["close"]).copy()
    if px.empty:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(px.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    px.index = idx.normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    close = px["close"].reindex(calendar.union(px.index)).sort_index().ffill(limit=4).reindex(calendar)
    # The daily print is the mark. A position already on earns the change since the last session.
    out = pd.DataFrame({"open": close, "close": close}).dropna()
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    return out


def kelly_overlay(base: Candidate, rf: pd.Series) -> Candidate:
    exposure = half_kelly_exposure(base.returns, rf.reindex(base.returns.index).fillna(0.0), min_obs=252, cap=3.0)
    scaled = scale_exposure(
        base.returns, exposure, rf.reindex(base.returns.index).fillna(0.0),
        borrow_spread=FINANCING_SPREAD / 252.0, cost=0.0005,
    )
    scale = float(exposure.dropna().iloc[-1]) if exposure.notna().any() else 1.0
    weights = {ticker: round(value * scale, 6) for ticker, value in base.weights.items()}
    weights = {ticker: value for ticker, value in weights.items() if abs(value) > 1e-6}
    return Candidate(
        name=f"kelly_{base.name}",
        family="half_kelly",
        params={"base": base.name, "fraction": 0.5, "cap": 3.0, "min_obs": 252, "latest_exposure": scale},
        returns=scaled,
        weights=weights,
        turnover_per_year=base.turnover_per_year,
        eligible=base.eligible,
        counts=True,
        note="Expanding half-Kelly on the base book, capped at 3x, reset on month-ends. Estimated only on past returns.",
        extras={"exposure": scale},
    )
