"""Cash instead of SSO while a dip hold is on and the 200-day band is off.

The live dip swaps the whole SPY half into SSO after three strictly down
closes, including while that band is off, and it leaves at the first up close
or after five sessions. The 2026-11-03 sized dip already tested blocking that
swap while the band is off, which leaves the core in SPY and ends the hold.
This rule does not rescore that block and it does not change the clock. The
hold is still the live clock. While the band is off and the hold is in the
scope below, a fixed fraction of the core pile earns T-bills and the rest
stays in SPY. While the band is on, the hold is the live SSO swap.

The cuts and the scopes were fixed before this sample was scored. Do not add
a cell after seeing the result. The live book is not a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.dipbuy import rule_mask
from quantforge.strategies.trend import trend_state

# One step on either axis is a neighbor. The live book is not on the axes.
SCOPES = ("born", "carry")
CUTS = (0.50, 1.0)
OFF_NAME = "cut_off"
MAX_HOLD = 5
RULE = "3_down_closes"
WINDOW = 200
BAND = 0.02


def cells() -> list[tuple[str, float]]:
    return [(scope, float(cut)) for scope in SCOPES for cut in CUTS]


def variant_name(scope: str, cut: float) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown off-trend dip scope {scope!r}")
    fraction = _cut_fraction(cut)
    points = int(round(fraction * 100))
    return f"zcut_{scope}_c{points:03d}"


def _cut_fraction(cut: float) -> float:
    for item in CUTS:
        if abs(float(cut) - float(item)) <= 1e-12:
            return float(item)
    raise ValueError(f"off-trend dip cut must be one of {CUTS}")


def offtrend_cut(
    close: pd.Series,
    trend: pd.Series,
    scope: str,
    cut: float,
    max_hold: int = MAX_HOLD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Core vehicle decided at each close, for the next session.

    Returns ``(invested, sso, clock)``. ``invested`` is the share of the core
    pile left in equities. ``sso`` is true when that share is SSO and false
    when it is SPY. ``clock`` is the live dip hold. A value uses the close at
    that session and earlier only.

    ``scope="carry"`` uses the cash vehicle on every off-trend session of a
    hold, including a hold that started while the band was on.
    ``scope="born"`` uses the cash vehicle only when the hold's entry close
    was already off-trend. A hold that started on-trend stays in SSO after
    the band turns off, which is the live book.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown off-trend dip scope {scope!r}")
    fraction = _cut_fraction(cut)
    if max_hold < 1:
        raise ValueError("max_hold must be positive")
    series = close.astype(float).sort_index()
    state = trend.reindex(series.index)
    if state.isna().any():
        raise ValueError("trend is missing sessions")
    state_np = state.astype(bool).to_numpy(dtype=bool)
    signal = rule_mask(series, RULE).to_numpy(dtype=bool)
    prices = series.to_numpy(dtype=float)
    n = len(series)
    held = np.zeros(n, dtype=bool)
    clock = np.zeros(n, dtype=bool)
    invested = np.ones(n, dtype=float)
    sso = np.zeros(n, dtype=bool)
    in_trade = False
    hold_count = 0
    born_off = False
    for t in range(n):
        if not in_trade:
            if signal[t]:
                in_trade = True
                hold_count = 0
                born_off = not bool(state_np[t])
                nxt = True
            else:
                nxt = False
        elif held[t]:
            hold_count += 1
            up = (
                t > 0
                and np.isfinite(prices[t])
                and np.isfinite(prices[t - 1])
                and prices[t] > prices[t - 1]
            )
            if (up and hold_count >= 1) or hold_count >= max_hold:
                in_trade = False
                nxt = False
                born_off = False
            else:
                nxt = True
        else:
            in_trade = False
            nxt = False
            born_off = False
        clock[t] = nxt
        if nxt and bool(state_np[t]):
            invested[t] = 1.0
            sso[t] = True
        elif nxt and not bool(state_np[t]) and (scope == "carry" or born_off):
            invested[t] = 1.0 - fraction
            sso[t] = False
        elif nxt and not bool(state_np[t]):
            invested[t] = 1.0
            sso[t] = True
        else:
            invested[t] = 1.0
            sso[t] = False
        if t + 1 < n:
            held[t + 1] = nxt
    index = series.index
    return (
        pd.Series(invested, index=index, name="core_invested"),
        pd.Series(sso, index=index, name="core_sso"),
        pd.Series(clock, index=index, name="dip_clock"),
    )


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "dip_cut"
    ]
    if not kept:
        return None
    return kept[-1]


def _params(entry: dict, name: str | None) -> tuple[str, float] | None:
    if name is None or name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        scope = row.get("scope")
        cut = row.get("cut")
        if scope is None or cut is None:
            raise DataError(f"kept off-trend dip cut {name} has no parameters")
        if scope not in SCOPES:
            raise DataError(f"kept off-trend dip scope {scope!r} is outside the grid")
        return scope, _cut_fraction(cut)
    raise DataError(f"kept off-trend dip cut has no parameters for {name}")


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def kept_dip_cut(iterations_path, close: pd.Series, as_of=None):
    """Core hold and SSO boost for a kept cut, or ``(None, None, None)``.

    The cell is the walk-forward pick in force on ``as_of``. The series use
    closes on or before ``as_of`` only, and the 200-day 2% band on those
    closes. A pick of the live book, and a rejected iteration, return
    ``(None, None, None)``. ``core_hold`` is the invested share. It is applied
    only while the band is off. ``dip_boost`` is 1 when the invested share is
    SSO and 0 when it is SPY.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the off-trend dip cut")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    chosen = _selection(entry, visible.index[-1])
    params = _params(entry, chosen)
    if params is None:
        return None, None, None
    scope, fraction = params
    trend = trend_state(visible, WINDOW, BAND, "daily")
    invested, sso, clock = offtrend_cut(visible, trend, scope, fraction)
    info = {
        "dip_cut_rule": variant_name(scope, fraction),
        "dip_cut_scope": scope,
        "dip_cut": float(fraction),
        "dip_clock_on": bool(clock.iloc[-1]),
        "dip_cut_on": bool(clock.iloc[-1] and not bool(trend.iloc[-1]) and float(invested.iloc[-1]) < 1.0 - 1e-12),
    }
    return invested.rename("core_hold"), sso.astype(float).rename("dip_boost"), info
