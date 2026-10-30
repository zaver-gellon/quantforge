"""Core plus a trend-gated sleeve. Decisions use the close; fills are next open."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.engine.schedule import rebalance_mask
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal
from quantforge.strategies.dipbuy import dip_active

SLEEVE_SYMBOL = {1: "SPY", 2: "SSO", 3: "UPRO"}


@dataclass(frozen=True)
class TrendCoreSpec:
    """50/50 SPY + UPRO-when-trending is the default, not a fitted result.

    ``core_weight`` stays in ``core_symbol``. ``sleeve_weight`` is invested in
    the leveraged sleeve while the trend filter is on and in ``cash_symbol``
    (BIL) while it is off. ``freq`` is how often the filter is evaluated.
    ``rebalance`` is how often the mix is reset to those strategic weights.
    A trend flip between mix dates moves only the sleeve.
    """

    core_symbol: str = "SPY"
    core_weight: float = 0.5
    leverage: int = 3
    window: int = 200
    band: float = 0.02
    freq: str = "daily"
    rebalance: str = "quarterly"
    cash_symbol: str = "BIL"
    sleeve_weight: float | None = None

    def __post_init__(self):
        sleeve = self.sleeve_weight
        if sleeve is None:
            sleeve = round(1.0 - float(self.core_weight), 10)
        object.__setattr__(self, "sleeve_weight", float(sleeve))
        if self.leverage not in SLEEVE_SYMBOL:
            raise ValueError("leverage must be 1, 2, or 3")
        if self.core_weight < 0 or sleeve < 0 or self.core_weight + sleeve > 1 + 1e-8:
            raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
        if self.freq not in ("daily", "monthly"):
            raise ValueError("freq must be daily or monthly")
        if self.rebalance not in ("daily", "monthly", "quarterly"):
            raise ValueError("rebalance must be daily, monthly, or quarterly")
        if self.core_symbol not in ALLOWED_TREND_TICKERS or self.cash_symbol not in ALLOWED_TREND_TICKERS:
            raise ValueError("core and cash symbols must be SPY, UPRO, SSO, or BIL")

    @property
    def sleeve_symbol(self) -> str:
        return SLEEVE_SYMBOL[self.leverage]

    def name(self) -> str:
        core = int(round(self.core_weight * 100))
        sleeve = int(round(self.sleeve_weight * 100))
        band = int(round(self.band * 100))
        return (
            f"core{core}_slv{sleeve}_L{self.leverage}"
            f"_sma{self.window}_band{band}_{self.freq}_{self.rebalance}"
        )

    def params(self) -> dict:
        return {
            "core_symbol": self.core_symbol,
            "core_weight": self.core_weight,
            "sleeve_symbol": self.sleeve_symbol,
            "sleeve_weight": self.sleeve_weight,
            "leverage": self.leverage,
            "window": self.window,
            "band": self.band,
            "freq": self.freq,
            "rebalance": self.rebalance,
            "cash_symbol": self.cash_symbol,
        }


def trend_state(close: pd.Series, window: int, band: float = 0.0, freq: str = "daily") -> pd.Series:
    """True when the next session should be long the risky sleeve.

    The value at t uses the close at t and earlier bars only. With a band, the
    sleeve turns on only when price is above ``SMA * (1 + band)`` and turns off
    only when price is below ``SMA * (1 - band)``. Inside the band the previous
    state is kept, and the book stays off until the first real entry. Monthly
    evaluation updates only on month-end closes and carries the state forward.
    Before the SMA exists the state is off.
    """
    series = close.astype(float).sort_index()
    average = series.rolling(window, min_periods=window).mean()
    if freq == "daily":
        evaluate = np.ones(len(series), dtype=bool)
    elif freq == "monthly":
        evaluate = rebalance_mask(series.index, "monthly").to_numpy()
    else:
        raise ValueError("freq must be daily or monthly")
    prices = series.to_numpy(dtype=float)
    means = average.to_numpy(dtype=float)
    state = False
    out = np.zeros(len(series), dtype=bool)
    for i in range(len(series)):
        if evaluate[i] and np.isfinite(means[i]) and np.isfinite(prices[i]):
            upper = prices[i] > means[i] * (1.0 + band)
            lower = prices[i] < means[i] * (1.0 - band)
            if upper:
                state = True
            elif lower:
                state = False
        out[i] = state
    return pd.Series(out, index=series.index, name="trend_on")


def strategic_weights(
    spec: TrendCoreSpec,
    trend_on: bool,
    dip_on: bool = False,
    sleeve_fraction: float = 1.0,
) -> dict[str, float]:
    """Strategic weights. A dip swaps the core from SPY to SSO and leaves the sleeve budget alone.

    ``sleeve_fraction`` is the share of the sleeve budget kept in the leveraged
    fund while the trend is on. The rest of that budget is cash. The fraction
    is ignored when the trend is off.
    """
    raw: dict[str, float] = {}

    def add(symbol: str, weight: float) -> None:
        if weight > 0:
            raw[symbol] = raw.get(symbol, 0.0) + float(weight)

    fraction = min(1.0, max(0.0, float(sleeve_fraction))) if trend_on else 0.0
    add("SSO" if dip_on else spec.core_symbol, spec.core_weight)
    add(spec.sleeve_symbol, spec.sleeve_weight * fraction)
    add(spec.cash_symbol, spec.sleeve_weight * (1.0 - fraction))
    rounded = {symbol: round(weight, 6) for symbol, weight in raw.items() if round(weight, 6) > 0}
    total = sum(rounded.values())
    if total > 1.0:
        largest = max(rounded, key=rounded.get)
        rounded[largest] = round(rounded[largest] - (total - 1.0), 6)
        if rounded[largest] <= 0:
            del rounded[largest]
    return rounded


def build_trend_signal(
    spec: TrendCoreSpec,
    spy_close: pd.Series,
    as_of=None,
    strategy_id: str = "trend_core",
    generated_at: str | None = None,
    apply_dip: bool = True,
    dip_rule: str = "3_down_closes",
    dip_hold: int = 5,
    hedge_on: pd.Series | None = None,
    hedge_rule: str | None = None,
    hedge_info: dict | None = None,
    sleeve_scale: pd.Series | None = None,
    vol_info: dict | None = None,
    trend_override: pd.Series | None = None,
    vote_info: dict | None = None,
    dd_info: dict | None = None,
    idle_on: pd.Series | None = None,
    idle_ready: pd.Series | None = None,
    idle_symbol: str | None = None,
    idle_info: dict | None = None,
) -> dict:
    """Target weights for the next NYSE session after ``as_of``.

    The default book is half SPY and half UPRO while SPY is above its 200-day
    SMA by 2% (enter above 1.02x, exit below 0.98x, hold the previous state in
    between). The UPRO half is replaced by BIL when the trend is off. After
    three strictly down SPY closes, only the SPY half switches to SSO, for at
    most five sessions or until the first up close. The UPRO half is not resized.
    ``hedge_on`` true at the signal close moves that sleeve to BIL instead.
    ``sleeve_scale`` at the signal close is the fraction of the sleeve left in
    the leveraged fund when the sleeve is on. A missing scale leaves it at 1.
    ``trend_override`` replaces the single-average state. It is used only when
    a walk-forward vote cleared the materiality bar. ``dd_info`` marks a scale
    that came from a kept trailing-peak stop. ``idle_symbol`` is set only when
    a walk-forward diversifier cleared the materiality bar. While the equity
    trend is off and ``idle_ready`` is true, the whole book holds that asset
    if ``idle_on`` is true and BIL if it is false. Dates after ``as_of`` are
    ignored. A missing value on or before ``as_of`` is an error.
    """
    close = spy_close.dropna().astype(float).sort_index()
    if close.empty:
        raise ValueError("SPY close series is empty")
    if trend_override is None:
        state = trend_state(close, spec.window, spec.band, spec.freq)
    else:
        state = trend_override.reindex(close.index)
    if as_of is None:
        as_of_ts = close.index[-1]
    else:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_ts = as_of_ts.tz_convert("America/New_York").tz_localize(None)
        as_of_ts = as_of_ts.normalize()
    history = state.loc[:as_of_ts]
    if history.empty:
        raise ValueError("as_of is before the price history")
    as_of_ts = history.index[-1]
    history = state.loc[:as_of_ts]
    if history.isna().any():
        raise ValueError("trend override is missing sessions at or before as_of")
    history = history.astype(bool)
    visible = close.loc[:as_of_ts]
    dips = dip_active(visible, dip_rule, dip_hold) if apply_dip else pd.Series(False, index=visible.index)
    on = bool(history.iloc[-1])
    dip_on = bool(dips.iloc[-1]) if len(dips) else False
    previous = bool(history.iloc[-2]) if len(history) > 1 else None
    previous_dip = bool(dips.iloc[-2]) if len(dips) > 1 else False
    hedge_now = False
    hedge_prev = False
    if hedge_on is not None:
        hedge = hedge_on.reindex(visible.index).fillna(False).astype(bool)
        if len(hedge):
            hedge_now = bool(hedge.iloc[-1])
        if len(hedge) > 1:
            hedge_prev = bool(hedge.iloc[-2])
    idle_now = False
    idle_prev = False
    ready_now = False
    ready_prev = False
    if idle_symbol is not None:
        if idle_on is None or idle_ready is None:
            raise ValueError("idle symbol requires idle_on and idle_ready")
        aligned_idle = idle_on.reindex(visible.index)
        aligned_ready = idle_ready.reindex(visible.index)
        if aligned_idle.isna().any() or aligned_ready.isna().any():
            raise ValueError("idle state is missing sessions at or before as_of")
        aligned_idle = aligned_idle.astype(bool)
        aligned_ready = aligned_ready.astype(bool)
        idle_now = bool(aligned_idle.iloc[-1])
        ready_now = bool(aligned_ready.iloc[-1])
        if len(aligned_idle) > 1:
            idle_prev = bool(aligned_idle.iloc[-2])
            ready_prev = bool(aligned_ready.iloc[-2])
    sleeve_on = on and not hedge_now
    fraction = 1.0 if sleeve_on else 0.0
    scale_changed = False
    if sleeve_scale is not None:
        prev_fraction = 1.0 if (bool(previous) and not hedge_prev) else 0.0
        aligned_scale = sleeve_scale.reindex(visible.index).astype(float)
        if sleeve_on and len(aligned_scale) and np.isfinite(aligned_scale.iloc[-1]):
            fraction = min(1.0, max(0.0, float(aligned_scale.iloc[-1])))
        if len(aligned_scale) > 1 and bool(previous) and not hedge_prev and np.isfinite(aligned_scale.iloc[-2]):
            prev_fraction = min(1.0, max(0.0, float(aligned_scale.iloc[-2])))
        scale_changed = abs(fraction - prev_fraction) > 1e-8
    schedule = rebalance_mask(state.index, spec.rebalance)
    scheduled = bool(schedule.loc[as_of_ts])
    changed = previous is None or on != previous
    dip_changed = dip_on != previous_dip
    hedge_changed = hedge_now != hedge_prev
    scale_label = "drawdown_stop" if dd_info else "vol_scale"
    if spec.rebalance == "daily":
        trade = True
        reason = "trend_change" if changed else "daily_target"
    elif changed and dip_changed:
        trade = True
        reason = "trend_and_dip"
    elif changed and hedge_changed:
        trade = True
        reason = "trend_and_credit_hedge"
    elif changed and scale_changed:
        trade = True
        reason = f"trend_and_{scale_label}"
    elif changed and scheduled:
        trade = True
        reason = "trend_change_and_schedule"
    elif changed:
        trade = True
        reason = "trend_change"
    elif dip_changed and hedge_changed:
        trade = True
        reason = "dip_and_credit_hedge"
    elif dip_changed and scale_changed:
        trade = True
        reason = f"dip_and_{scale_label}"
    elif dip_changed:
        trade = True
        reason = "dip_overlay"
    elif hedge_changed and scale_changed:
        trade = True
        reason = f"credit_hedge_and_{scale_label}"
    elif hedge_changed:
        trade = True
        reason = "credit_hedge"
    elif scale_changed:
        trade = True
        reason = scale_label
    elif scheduled:
        trade = True
        reason = "scheduled_rebalance"
    else:
        trade = False
        reason = "none"

    def _book_mode(trend_flag: bool, hedge_flag: bool, ready_flag: bool, idle_flag: bool) -> str:
        if trend_flag and not hedge_flag:
            return "equity"
        # The diversifier replaces the book only after the equity trend is off.
        # A credit hedge, when one is kept, still only moves the sleeve.
        if not trend_flag and idle_symbol is not None and ready_flag and idle_flag:
            return "div"
        if not trend_flag and idle_symbol is not None and ready_flag and not idle_flag:
            return "cash"
        return "baseline"

    mode = _book_mode(on, hedge_now, ready_now, idle_now)
    prev_trend = False if previous is None else bool(previous)
    prev_mode = _book_mode(prev_trend, hedge_prev, ready_prev, idle_prev)
    if idle_symbol is not None and mode != prev_mode and not {mode, prev_mode} <= {"equity", "baseline"}:
        trade = True
        reason = "idle_diversifier"
    invested = float(spec.core_weight) + float(spec.sleeve_weight)
    if mode == "div":
        weights = {idle_symbol: round(invested, 6)}
    elif mode == "cash":
        weights = {spec.cash_symbol: round(invested, 6)}
    else:
        weights = strategic_weights(spec, sleeve_on, dip_on, fraction if sleeve_on else 1.0)
    average = close.rolling(spec.window, min_periods=spec.window).mean().loc[as_of_ts]
    payload = {
        "date": next_nyse_session(as_of_ts).strftime("%Y-%m-%d"),
        "weights": weights,
        "strategy": strategy_id,
        "strategy_name": spec.name(),
        "generated_at": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_date": pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        "trade": trade,
        "trade_reason": reason,
        "rebalance_mix": scheduled,
        "diagnostics": {
            "trend_on": on,
            "hedge_on": hedge_now,
            "hedge_rule": hedge_rule,
            "sleeve_on": sleeve_on,
            "dip_on": dip_on,
            "dip_rule": dip_rule if apply_dip else None,
            "reference_close": float(close.loc[as_of_ts]),
            "sma": None if not np.isfinite(average) else float(average),
            "sma_window": spec.window,
            "band": spec.band,
            "freq": spec.freq,
            "rebalance": spec.rebalance,
            "core_weight": spec.core_weight,
            "sleeve_weight": spec.sleeve_weight,
            "sleeve_symbol": spec.sleeve_symbol,
            "leverage": spec.leverage,
        },
    }
    if hedge_info:
        payload["diagnostics"].update(hedge_info)
    if vol_info:
        payload["diagnostics"].update(vol_info)
        payload["diagnostics"]["sleeve_scale"] = fraction
    if dd_info:
        payload["diagnostics"].update(dd_info)
        payload["diagnostics"]["sleeve_scale"] = fraction
    if vote_info:
        payload["diagnostics"].update(vote_info)
    if idle_info:
        payload["diagnostics"].update(idle_info)
    if idle_symbol is not None:
        payload["diagnostics"]["idle_symbol"] = idle_symbol
        payload["diagnostics"]["idle_on"] = bool(idle_now)
        payload["diagnostics"]["idle_ready"] = bool(ready_now)
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_symbol is not None:
        allowed.add(idle_symbol)
    validate_signal(payload, allowed_tickers=frozenset(allowed))
    return payload
