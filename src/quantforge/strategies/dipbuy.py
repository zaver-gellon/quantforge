"""Dip overlay for the SPY half of the trend-core book.

After the rule (three strictly down closes, by default), that half is SSO
instead of SPY. The position is entered at the next open, and it is exited at
the next open after the first up close or after five trading sessions. The
UPRO half is not an input to this module. The same-day exit wins over a fresh
entry. ``dip_targets`` still reports the overlay as a 0/1 flag in the SSO column.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal

DIP_RULES = ("3_down_closes", "2d<-2%", "2d<-3%", "3d<-3%", "3d<-4%")


def rule_mask(close: pd.Series, rule: str) -> pd.Series:
    series = close.astype(float)
    if rule == "3_down_closes":
        down = series.diff() < 0
        mask = down & down.shift(1) & down.shift(2)
    elif rule == "2d<-2%":
        mask = series / series.shift(2) - 1.0 < -0.02
    elif rule == "2d<-3%":
        mask = series / series.shift(2) - 1.0 < -0.03
    elif rule == "3d<-3%":
        mask = series / series.shift(3) - 1.0 < -0.03
    elif rule == "3d<-4%":
        mask = series / series.shift(3) - 1.0 < -0.04
    else:
        raise ValueError(f"unknown dip rule {rule!r}")
    return mask.fillna(False)


def dip_targets(close: pd.Series, rule: str = "3_down_closes", max_hold: int = 5):
    """Weights decided at each close, and whether that close changes the position.

    Columns are ``SPY`` and ``SSO``. A weight at t is the book to hold on the
    next session.
    """
    if max_hold < 1:
        raise ValueError("max_hold must be positive")
    series = close.astype(float).sort_index()
    signal = rule_mask(series, rule).to_numpy(dtype=bool)
    prices = series.to_numpy(dtype=float)
    n = len(series)
    held = np.array(["SPY"] * n, dtype=object)
    decided = np.array(["SPY"] * n, dtype=object)
    in_trade = False
    hold_count = 0
    for t in range(n):
        if not in_trade:
            if signal[t]:
                in_trade = True
                hold_count = 0
                nxt = "SSO"
            else:
                nxt = "SPY"
        elif held[t] == "SSO":
            hold_count += 1
            up = t > 0 and np.isfinite(prices[t]) and np.isfinite(prices[t - 1]) and prices[t] > prices[t - 1]
            if (up and hold_count >= 1) or hold_count >= max_hold:
                in_trade = False
                nxt = "SPY"
            else:
                nxt = "SSO"
        else:
            nxt = "SPY"
            in_trade = False
        decided[t] = nxt
        if t + 1 < n:
            held[t + 1] = nxt
    weights = pd.DataFrame(0.0, index=series.index, columns=["SPY", "SSO"])
    weights["SPY"] = np.where(decided == "SPY", 1.0, 0.0)
    weights["SSO"] = np.where(decided == "SSO", 1.0, 0.0)
    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    changed[1:] = decided[1:] != decided[:-1]
    rebalance = pd.Series(changed, index=series.index, name="rebalance")
    return weights, rebalance


def dip_active(close: pd.Series, rule: str = "3_down_closes", max_hold: int = 5) -> pd.Series:
    """True when the SPY half should be SSO on the next session."""
    weights, _rebalance = dip_targets(close, rule, max_hold)
    return weights["SSO"].gt(0).rename("dip_on")


def build_dip_signal(
    spy_close: pd.Series,
    rule: str = "3_down_closes",
    max_hold: int = 5,
    as_of=None,
    strategy_id: str = "dip_buy",
    generated_at: str | None = None,
    core_weight: float = 0.5,
) -> dict:
    close = spy_close.dropna().astype(float).sort_index()
    if as_of is not None:
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/New_York").tz_localize(None)
        close = close.loc[: stamp.normalize()]
    if close.empty:
        raise ValueError("SPY close series is empty")
    if not 0 < core_weight <= 1:
        raise ValueError("core_weight must be in (0, 1]")
    weights, _rebalance = dip_targets(close, rule, max_hold)
    last = weights.iloc[-1]
    held = "SSO" if float(last["SSO"]) > 0 else "SPY"
    book = {held: float(core_weight)}
    as_of_ts = close.index[-1]
    payload = {
        "date": next_nyse_session(as_of_ts).strftime("%Y-%m-%d"),
        "weights": book,
        "strategy": strategy_id,
        "strategy_name": f"dip_{rule}_hold{max_hold}",
        "generated_at": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_date": pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        "trade": bool(_rebalance.iloc[-1]) if len(_rebalance) else True,
        "trade_reason": "dip_overlay",
        "diagnostics": {
            "rule": rule,
            "max_hold": max_hold,
            "reference_close": float(close.iloc[-1]),
            "position": "SSO" if book.get("SSO") else "SPY",
        },
    }
    validate_signal(payload, allowed_tickers=ALLOWED_TREND_TICKERS)
    return payload
