"""Scale the UPRO sleeve's leverage by trailing realized volatility.

The 2026-10-26 volatility scale cut the sleeve's weight toward T-bills. This
grid does not. While the 200-day band is on, the sleeve stays in equities and
its leverage is ``clip(target_vol / realized_vol, 1, 3)``. That leverage is a
blend of SPY and UPRO: the UPRO share of the sleeve is ``(L - 1) / 2`` and the
rest is SPY. While the band is off, the sleeve is T-bills and this series is
ignored.

The lookbacks and targets were fixed before this sample was scored. Do not add
a cell after seeing the result. The current book, which keeps leverage at 3
whenever the band is on, is not a cell. It is already counted. Every leverage
cell is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on either axis is a neighbor. Targets are three times 12%, 16%, and
# 20% equity volatility, so the sleeve is still at 3x at or below that level.
LOOKBACKS = (21, 63, 126)
TARGET_VOLS = (0.36, 0.48, 0.60)
MIN_LEVERAGE = 1.0
MAX_LEVERAGE = 3.0
OFF_NAME = "vgear_off"


def cells() -> list[tuple[int, float]]:
    return [(int(lookback), float(target)) for lookback in LOOKBACKS for target in TARGET_VOLS]


def variant_name(lookback: int, target_vol: float) -> str:
    points = int(round(float(target_vol) * 100))
    return f"vgear_lb{int(lookback):03d}_tv{points:02d}"


def sleeve_leverage(close: pd.Series, lookback: int, target_vol: float) -> pd.Series:
    """Sleeve leverage known at the close. Missing windows stay at 3x.

    Realized volatility is the sample standard deviation of close-to-close
    simple returns over ``lookback`` sessions, annualized with ``sqrt(252)``.
    """
    if int(lookback) < 2:
        raise ValueError("lookback must be at least 2")
    target = float(target_vol)
    if not 0.0 < target <= MAX_LEVERAGE:
        raise ValueError("target_vol must be in (0, 3]")
    prices = close.astype(float).sort_index()
    returns = prices.pct_change()
    realized = returns.rolling(int(lookback), min_periods=int(lookback)).std(ddof=1) * np.sqrt(252.0)
    raw = target / realized
    leverage = raw.clip(lower=MIN_LEVERAGE, upper=MAX_LEVERAGE)
    leverage = leverage.where(realized > 0.0, MAX_LEVERAGE).fillna(MAX_LEVERAGE)
    leverage = leverage.clip(lower=MIN_LEVERAGE, upper=MAX_LEVERAGE)
    leverage.name = "sleeve_leverage"
    return leverage


def upro_share(leverage: pd.Series) -> pd.Series:
    """UPRO share of the sleeve. The rest is SPY. ``L = 1 + 2 * share``."""
    share = ((leverage.astype(float) - MIN_LEVERAGE) / (MAX_LEVERAGE - MIN_LEVERAGE)).clip(lower=0.0, upper=1.0)
    share = share.mask(share <= 1e-12, 0.0)
    share = share.mask(share >= 1.0 - 1e-12, 1.0)
    share.name = "upro_share"
    return share


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "vol_gear"
    ]
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _params_for(entry: dict, name: str) -> tuple[int, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        lookback = row.get("lookback")
        target = row.get("target_vol")
        if lookback is None or target is None:
            raise DataError(f"kept vol-gear rule {name} has no lookback or target")
        found = (int(lookback), float(target))
        if not any(found[0] == cell[0] and abs(found[1] - cell[1]) <= 1e-12 for cell in cells()):
            raise DataError(f"kept vol-gear parameters {found} are outside the grid")
        return found
    raise DataError(f"kept vol-gear rule has no parameters for {name}")


def kept_vol_gear(iterations_path, close: pd.Series, as_of=None):
    """UPRO share of the sleeve for a kept vol gear, or ``(None, None)``.

    The share is 1 when the walk-forward pick is the current book. The caller
    ignores the share while the 200-day band is off. Sessions after ``as_of``
    are not used.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the vol-gear rule")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    states = {
        variant_name(lookback, target): upro_share(sleeve_leverage(visible, lookback, target))
        for lookback, target in cells()
    }
    picks = []
    for date, name in entry.get("walk_forward_picks") or []:
        decision = pd.Timestamp(date)
        if decision <= visible.index[-1]:
            picks.append((decision, name))
    picks.sort(key=lambda item: item[0])
    values = np.ones(len(visible), dtype=float)
    cursor = 0
    current_name = OFF_NAME
    for i, ts in enumerate(visible.index):
        while cursor < len(picks) and picks[cursor][0] <= ts:
            current_name = picks[cursor][1]
            _params_for(entry, current_name)
            cursor += 1
        if current_name == OFF_NAME:
            values[i] = 1.0
        else:
            values[i] = float(states[current_name].loc[ts])
    series = pd.Series(values, index=visible.index, name="upro_share")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    share_now = None if params is None else float(states[variant_name(*params)].iloc[-1])
    leverage_now = None if share_now is None else float(MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * share_now)
    info = {
        "vol_gear_rule": chosen,
        "vol_gear_lookback": None if params is None else int(params[0]),
        "vol_gear_target": None if params is None else float(params[1]),
        "vol_gear_upro_share": share_now if share_now is not None else 1.0,
        "vol_gear_leverage": leverage_now if leverage_now is not None else MAX_LEVERAGE,
        "vol_gear_live": params is None,
    }
    return series, info
