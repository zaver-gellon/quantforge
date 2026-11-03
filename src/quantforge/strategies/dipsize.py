"""Size the three-down-close overlay, and refuse it while the 200-day band is off.

The live overlay swaps the entire SPY half into SSO after three strictly down
closes, including while that band is off, and it leaves at the first up close
or after five sessions. The fractions and the gate below were fixed before
this sample was scored. Do not add a cell after seeing the result.

The ungated full swap is the live book. It is already counted. Every other
cell is a new trial. The entry, the first-up exit, and the five-session cap
are not searched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.dipbuy import rule_mask

# One step on either axis is a neighbor. ("any", 1.0) is the current book.
GATES = ("any", "trend_on")
BOOSTS = (0.25, 0.50, 0.75, 1.0)
LIVE_GATE = "any"
LIVE_BOOST = 1.0
MAX_HOLD = 5
RULE = "3_down_closes"


def cells() -> list[tuple[str, float]]:
    return [(gate, float(boost)) for gate in GATES for boost in BOOSTS]


def is_live_cell(gate: str, boost: float) -> bool:
    """The current book: the whole core pile is SSO, with no trend gate."""
    return gate == LIVE_GATE and abs(float(boost) - LIVE_BOOST) <= 1e-12


def variant_name(gate: str, boost: float) -> str:
    if gate not in GATES:
        raise ValueError(f"unknown dip gate {gate!r}")
    points = int(round(float(boost) * 100))
    tag = "any" if gate == "any" else "trend"
    return f"dipsize_{tag}_b{points:03d}"


def dip_boost(
    close: pd.Series,
    trend: pd.Series,
    gate: str,
    boost: float,
    max_hold: int = MAX_HOLD,
) -> pd.Series:
    """Share of the core pile that should be SSO on the next session.

    The entry is three strictly down closes. The exit is the first up close
    or ``max_hold`` sessions, and the exit wins over a fresh entry on the same
    close. ``gate="trend_on"`` also requires the trend to be on at the entry
    close, and it flattens a held dip on the close the trend turns off. That
    flatten does not re-enter on the same close. ``gate="any"`` ignores the
    trend. A boost of 1 with ``gate="any"`` matches ``dip_active``.
    """
    if gate not in GATES:
        raise ValueError(f"unknown dip gate {gate!r}")
    fraction = float(boost)
    if fraction < -1e-12 or fraction > 1.0 + 1e-12:
        raise ValueError("boost must be in [0, 1]")
    fraction = min(1.0, max(0.0, fraction))
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
    out = np.zeros(n, dtype=float)
    in_trade = False
    hold_count = 0
    for t in range(n):
        if not in_trade:
            allow = gate == "any" or bool(state_np[t])
            if bool(signal[t]) and allow:
                in_trade = True
                hold_count = 0
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
            trend_off = gate == "trend_on" and not bool(state_np[t])
            if trend_off or (up and hold_count >= 1) or hold_count >= max_hold:
                in_trade = False
                nxt = False
            else:
                nxt = True
        else:
            in_trade = False
            nxt = False
        out[t] = fraction if nxt else 0.0
        if t + 1 < n:
            held[t + 1] = nxt
    return pd.Series(out, index=series.index, name="dip_boost")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "dip_size"
    ]
    if not kept:
        return None
    return kept[-1]


def _params(entry: dict, name: str | None) -> tuple[str, float] | None:
    if name is None:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        gate = row.get("gate")
        boost = row.get("boost")
        if gate is None or boost is None:
            raise DataError(f"kept dip-size rule {name} has no parameters")
        if gate not in GATES:
            raise DataError(f"kept dip-size gate {gate!r} is outside the grid")
        fraction = float(boost)
        if not any(abs(fraction - item) <= 1e-12 for item in BOOSTS):
            raise DataError(f"kept dip-size boost {fraction} is outside the grid")
        return gate, fraction
    raise DataError(f"kept dip-size rule has no parameters for {name}")


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def kept_dip_size(iterations_path, close: pd.Series, trend: pd.Series, as_of=None):
    """SSO fraction for a kept sized dip, or ``(None, None)``.

    The fraction at each session is the full-history rule of the walk-forward
    cell selected for that session. Sessions after ``as_of`` are not used.
    Before the first pick, the fraction is the live overlay.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the sized dip")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    visible_trend = trend.reindex(visible.index)
    if visible_trend.isna().any():
        raise DataError("trend is missing sessions at or before as_of")
    picks = []
    for date, name in entry.get("walk_forward_picks") or []:
        decision = pd.Timestamp(date)
        if decision <= visible.index[-1]:
            picks.append((decision, name))
    picks.sort(key=lambda item: item[0])
    rules: dict[str, pd.Series] = {}
    values = np.zeros(len(visible), dtype=float)
    cursor = 0
    current = variant_name(LIVE_GATE, LIVE_BOOST)
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current = picks[cursor][1]
            cursor += 1
        if current not in rules:
            params = _params(entry, current)
            if params is None:
                raise DataError(f"kept dip-size pick {current} is not a cell")
            rules[current] = dip_boost(visible, visible_trend, params[0], params[1])
        values[i] = float(rules[current].iloc[i])
    series = pd.Series(values, index=visible.index, name="dip_boost")
    chosen = _selection(entry, visible.index[-1])
    if chosen is None:
        chosen = variant_name(LIVE_GATE, LIVE_BOOST)
    params = _params(entry, chosen)
    if params is None:
        raise DataError(f"kept dip-size pick {chosen} is not a cell")
    info = {
        "dip_size_rule": chosen,
        "dip_size_gate": params[0],
        "dip_size_boost": float(params[1]),
        "dip_size_live": bool(is_live_cell(params[0], params[1])),
    }
    return series, info
