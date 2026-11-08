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


def _relocate_sleeve_cash(
    weights: dict[str, float],
    cash_symbol: str,
    sleeve_weight: float,
    symbol: str,
) -> dict[str, float]:
    """Move the sleeve budget from cash into ``symbol``. Core cash stays put."""
    budget = round(float(sleeve_weight), 6)
    if budget <= 0.0:
        return dict(weights)
    out = dict(weights)
    held = float(out.get(cash_symbol, 0.0))
    move = min(held, budget)
    if move <= 1e-12:
        raise ValueError("sleeve cash is not in the book to move into the diversifier")
    out[symbol] = round(float(out.get(symbol, 0.0)) + move, 6)
    left = round(held - move, 6)
    if left > 1e-12:
        out[cash_symbol] = left
    else:
        out.pop(cash_symbol, None)
    return out


def strategic_weights(
    spec: TrendCoreSpec,
    trend_on: bool,
    dip_on: bool = False,
    sleeve_fraction: float = 1.0,
    core_hold: float = 1.0,
    trend_is_on: bool | None = None,
    dip_boost: float | None = None,
    sleeve_unlevered: bool = False,
) -> dict[str, float]:
    """Strategic weights. A dip moves part of the core from SPY to SSO and leaves the sleeve budget alone.

    ``sleeve_fraction`` is the share of the sleeve budget kept in the leveraged
    fund while the trend is on. The rest of that budget is cash. The fraction
    is ignored when the trend is off. ``core_hold`` is the share of the core
    budget left in SPY or SSO while the equity trend is off. The rest of that
    budget is cash. It is ignored while the equity trend is on. ``trend_is_on``
    is that equity trend when a credit hedge has already turned ``trend_on``
    into the sleeve flag. When it is omitted, the sleeve flag is the equity trend.
    ``dip_boost`` is the share of the invested core held in SSO. When it is
    omitted, the whole invested core is SSO if ``dip_on`` and SPY otherwise.
    ``sleeve_unlevered`` puts the sleeve budget in the core symbol instead of
    the leveraged fund. It applies only while the equity trend is on, and the
    dip does not cover that stake.
    """
    raw: dict[str, float] = {}

    def add(symbol: str, weight: float) -> None:
        if weight > 0:
            raw[symbol] = raw.get(symbol, 0.0) + float(weight)

    equity_trend = bool(trend_on) if trend_is_on is None else bool(trend_is_on)
    hold = 1.0 if equity_trend else min(1.0, max(0.0, float(core_hold)))
    fraction = min(1.0, max(0.0, float(sleeve_fraction))) if trend_on else 0.0
    if dip_boost is None:
        boost = 1.0 if dip_on else 0.0
    else:
        boost = min(1.0, max(0.0, float(dip_boost)))
    add(spec.core_symbol, spec.core_weight * hold * (1.0 - boost))
    add("SSO", spec.core_weight * hold * boost)
    if bool(sleeve_unlevered) and equity_trend:
        add(spec.core_symbol, spec.sleeve_weight)
        add(spec.cash_symbol, spec.core_weight * (1.0 - hold))
    else:
        add(spec.sleeve_symbol, spec.sleeve_weight * fraction)
        add(spec.cash_symbol, spec.sleeve_weight * (1.0 - fraction) + spec.core_weight * (1.0 - hold))
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
    confirm_info: dict | None = None,
    dd_info: dict | None = None,
    idle_on: pd.Series | None = None,
    idle_ready: pd.Series | None = None,
    idle_symbol: str | None = None,
    idle_info: dict | None = None,
    core_hold: pd.Series | None = None,
    core_hold_info: dict | None = None,
    curve_on: pd.Series | None = None,
    curve_info: dict | None = None,
    dip_boost: pd.Series | None = None,
    dip_size_info: dict | None = None,
    drift_mix: pd.Series | None = None,
    drift_info: dict | None = None,
    sleeve_fast: pd.Series | None = None,
    sleeve_gear_info: dict | None = None,
    ema_info: dict | None = None,
    sleeve_cash_on: pd.Series | None = None,
    sleeve_cash_ready: pd.Series | None = None,
    sleeve_cash_symbol: str | None = None,
    sleeve_cash_info: dict | None = None,
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
    a walk-forward vote or confirmation rule cleared the materiality bar.
    ``confirm_info`` marks a kept N-day confirmation of the band. ``dd_info`` marks a scale
    that came from a kept trailing-peak stop.     ``idle_symbol`` is set only when
    a walk-forward diversifier cleared the materiality bar. While the equity
    trend is off and ``idle_ready`` is true, the whole book holds that asset
    if ``idle_on`` is true and BIL if it is false. ``core_hold`` is the share
    of the core budget left in equities while the equity trend is off, and only
    when a walk-forward core-hold rule cleared the materiality bar.
    ``curve_on`` true at the signal close parks the whole book in BIL. It is
    set only when a walk-forward yield-curve clock cleared the materiality bar.
    ``dip_boost`` is the share of the invested core held in SSO. It is set only
    when a walk-forward sized dip cleared the materiality bar. A missing series
    leaves the three-down-close overlay at a full SSO swap.
    ``drift_mix`` replaces the quarterly mix date when a walk-forward drift
    band cleared the materiality bar and the cell in force is not the calendar.
    True at the signal close resets both piles at the next open. Dates after
    ``as_of`` are ignored. A missing value on or before ``as_of`` is an error.
    ``sleeve_fast`` is set only when a walk-forward sleeve-gear rule cleared
    the materiality bar. False at the signal close, while the equity trend is
    on and no credit hedge has moved the sleeve, holds that sleeve in SPY
    instead of the leveraged fund. The dip does not cover that stake. True
    leaves the sleeve in the leveraged fund. The flag is ignored while the
    equity trend is off. ``ema_info`` records a kept exponential filter when
    ``trend_override`` is that filter. It does not change the weights.
    ``sleeve_cash_symbol`` is set only when a walk-forward sleeve diversifier
    cleared the materiality bar. While the equity trend is off, the filter is
    ready, and ``sleeve_cash_on`` is true, the sleeve budget is held in that
    asset instead of BIL. The core is not moved. The flag is ignored while the
    equity trend is on.
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
    if dip_boost is not None:
        aligned_boost = dip_boost.reindex(visible.index).astype(float)
        if aligned_boost.isna().any() or not np.isfinite(aligned_boost.to_numpy(dtype=float)).all():
            raise ValueError("dip boost is missing sessions at or before as_of")
        if (aligned_boost < -1e-12).any() or (aligned_boost > 1.0 + 1e-12).any():
            raise ValueError("dip boost must be in [0, 1]")
        boost_now = min(1.0, max(0.0, float(aligned_boost.iloc[-1])))
        boost_prev = min(1.0, max(0.0, float(aligned_boost.iloc[-2]))) if len(aligned_boost) > 1 else 0.0
    else:
        dips = dip_active(visible, dip_rule, dip_hold) if apply_dip else pd.Series(False, index=visible.index)
        dip_flag = bool(dips.iloc[-1]) if len(dips) else False
        prev_flag = bool(dips.iloc[-2]) if len(dips) > 1 else False
        boost_now = 1.0 if dip_flag else 0.0
        boost_prev = 1.0 if prev_flag else 0.0
    on = bool(history.iloc[-1])
    dip_on = boost_now > 1e-12
    previous = bool(history.iloc[-2]) if len(history) > 1 else None
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
    hold_now = 1.0
    hold_prev = 1.0
    if core_hold is not None:
        aligned_hold = core_hold.reindex(visible.index).astype(float)
        if aligned_hold.isna().any() or not np.isfinite(aligned_hold.to_numpy()).all():
            raise ValueError("core hold is missing sessions at or before as_of")
        if (aligned_hold < -1e-12).any() or (aligned_hold > 1.0 + 1e-12).any():
            raise ValueError("core hold must be in [0, 1]")
        hold_now = min(1.0, max(0.0, float(aligned_hold.iloc[-1])))
        if len(aligned_hold) > 1:
            hold_prev = min(1.0, max(0.0, float(aligned_hold.iloc[-2])))
    curve_now = False
    curve_prev = False
    if curve_on is not None:
        aligned_curve = curve_on.reindex(visible.index)
        if aligned_curve.isna().any():
            raise ValueError("yield-curve clock is missing sessions at or before as_of")
        aligned_curve = aligned_curve.astype(bool)
        curve_now = bool(aligned_curve.iloc[-1])
        if len(aligned_curve) > 1:
            curve_prev = bool(aligned_curve.iloc[-2])
    prev_on = False if previous is None else bool(previous)
    hold_changed = abs((1.0 if on else hold_now) - (1.0 if prev_on else hold_prev)) > 1e-8
    fast_now = True
    fast_prev = True
    if sleeve_fast is not None:
        aligned_fast = sleeve_fast.reindex(visible.index)
        if aligned_fast.isna().any():
            raise ValueError("sleeve gear is missing sessions at or before as_of")
        aligned_fast = aligned_fast.astype(bool)
        fast_now = bool(aligned_fast.iloc[-1])
        if len(aligned_fast) > 1:
            fast_prev = bool(aligned_fast.iloc[-2])
    unlevered_now = bool(sleeve_on and not fast_now)
    prev_sleeve_on = bool(prev_on and not hedge_prev)
    unlevered_prev = bool(prev_sleeve_on and not fast_prev)
    gear_changed = (
        unlevered_now != unlevered_prev
        and bool(on) == bool(prev_on)
        and bool(on)
        and not hedge_now
        and not hedge_prev
    )
    sleeve_cash_now = False
    sleeve_cash_prev = False
    if sleeve_cash_symbol is not None:
        if idle_symbol is not None:
            raise ValueError("a sleeve diversifier and a whole-book diversifier cannot both be kept")
        if sleeve_cash_on is None or sleeve_cash_ready is None:
            raise ValueError("sleeve diversifier requires its state and ready series")
        aligned_cash_on = sleeve_cash_on.reindex(visible.index)
        aligned_cash_ready = sleeve_cash_ready.reindex(visible.index)
        if aligned_cash_on.isna().any() or aligned_cash_ready.isna().any():
            raise ValueError("sleeve diversifier is missing sessions at or before as_of")
        aligned_cash_on = aligned_cash_on.astype(bool)
        aligned_cash_ready = aligned_cash_ready.astype(bool)
        sleeve_cash_now = (not on) and bool(aligned_cash_on.iloc[-1]) and bool(aligned_cash_ready.iloc[-1])
        if len(aligned_cash_on) > 1:
            sleeve_cash_prev = (
                (not prev_on) and bool(aligned_cash_on.iloc[-2]) and bool(aligned_cash_ready.iloc[-2])
            )
    sleeve_cash_changed = bool(
        sleeve_cash_symbol is not None
        and previous is not None
        and bool(on) == bool(previous)
        and sleeve_cash_now != sleeve_cash_prev
    )
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
    drift_replaced_calendar = False
    if drift_info is not None and not bool(drift_info.get("drift_calendar", True)):
        if drift_mix is None:
            raise ValueError("a drift band requires its mix series")
        aligned_mix = drift_mix.reindex(visible.index)
        if aligned_mix.isna().any():
            raise ValueError("drift mix is missing sessions at or before as_of")
        scheduled = bool(aligned_mix.astype(bool).iloc[-1])
        drift_replaced_calendar = True
    changed = previous is None or on != previous
    dip_changed = abs(boost_now - boost_prev) > 1e-8
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
    elif hold_changed:
        trade = True
        reason = "fast_reentry" if (core_hold_info or {}).get("reentry_rule") else "core_hold"
    elif gear_changed:
        trade = True
        reason = "sleeve_gear"
    elif sleeve_cash_changed:
        trade = True
        reason = "sleeve_diversifier"
    elif scheduled:
        trade = True
        reason = "drift_band" if drift_replaced_calendar else "scheduled_rebalance"
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
    if curve_now != curve_prev:
        trade = True
        reason = "curve_clock"
    elif curve_now:
        trade = False
        reason = "none"
    invested = float(spec.core_weight) + float(spec.sleeve_weight)
    if curve_now:
        weights = {spec.cash_symbol: round(invested, 6)}
    elif mode == "div":
        weights = {idle_symbol: round(invested, 6)}
    elif mode == "cash":
        weights = {spec.cash_symbol: round(invested, 6)}
    else:
        weights = strategic_weights(
            spec, sleeve_on, dip_on, fraction if sleeve_on else 1.0,
            core_hold=hold_now, trend_is_on=on, dip_boost=boost_now,
            sleeve_unlevered=unlevered_now,
        )
        if sleeve_cash_now:
            weights = _relocate_sleeve_cash(weights, spec.cash_symbol, spec.sleeve_weight, sleeve_cash_symbol)
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
    if confirm_info:
        payload["diagnostics"].update(confirm_info)
        if confirm_info.get("confirm_band") is not None:
            payload["diagnostics"]["band"] = float(confirm_info["confirm_band"])
    if core_hold_info:
        payload["diagnostics"].update(core_hold_info)
        payload["diagnostics"]["core_hold"] = hold_now
        payload["diagnostics"]["core_hold_applied"] = 1.0 if on else hold_now
    if curve_info or curve_on is not None:
        if curve_info:
            payload["diagnostics"].update(curve_info)
        payload["diagnostics"]["curve_on"] = bool(curve_now)
    if dip_size_info or dip_boost is not None:
        if dip_size_info:
            payload["diagnostics"].update(dip_size_info)
        payload["diagnostics"]["dip_boost"] = boost_now
    if drift_info:
        payload["diagnostics"].update(drift_info)
        if drift_replaced_calendar:
            payload["diagnostics"]["drift_mix"] = bool(scheduled)
    if sleeve_gear_info or sleeve_fast is not None:
        if sleeve_gear_info:
            payload["diagnostics"].update(sleeve_gear_info)
        payload["diagnostics"]["sleeve_fast_on"] = bool(fast_now)
        payload["diagnostics"]["sleeve_unlevered"] = bool(unlevered_now)
    if ema_info:
        payload["diagnostics"].update(ema_info)
    if sleeve_cash_info or sleeve_cash_symbol is not None:
        if sleeve_cash_info:
            payload["diagnostics"].update(sleeve_cash_info)
        payload["diagnostics"]["sleeve_cash_applied"] = bool(sleeve_cash_now)
    if idle_info:
        payload["diagnostics"].update(idle_info)
    if idle_symbol is not None:
        payload["diagnostics"]["idle_symbol"] = idle_symbol
        payload["diagnostics"]["idle_on"] = bool(idle_now)
        payload["diagnostics"]["idle_ready"] = bool(ready_now)
    allowed = set(ALLOWED_TREND_TICKERS)
    if idle_symbol is not None:
        allowed.add(idle_symbol)
    if sleeve_cash_symbol is not None:
        allowed.add(sleeve_cash_symbol)
    validate_signal(payload, allowed_tickers=frozenset(allowed))
    return payload
