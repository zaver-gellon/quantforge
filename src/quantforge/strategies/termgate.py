"""Cut the on-trend UPRO sleeve when the VIX curve is inverted.

The 3x sleeve stays fully invested until SPY closes through its 200-day band.
The one-month VIX divided by the three-month VIX is the options market's
price of near stress versus three-month stress. A calm curve is in contango
(the ratio is below 1). The gate uses a same-session pair of closes only.
A missing print does not invert the curve and does not cut the sleeve.
There is no forward fill. While the 200-day 2% band is on and the ratio is
at or above a threshold, the sleeve's UPRO share drops to a floor and the
rest of the sleeve is SPY. The core's three-down-close dip is a separate clock.

The thresholds and floors were fixed before this sample was scored. Do not
add a cell after seeing the result. The current book, which keeps the share
at 1 whenever the band is on, is not a cell. It is already counted. Every
term-structure cell is a new trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError
from quantforge.strategies.trend import trend_state

# One step on either axis is a neighbor. 1.00 is backwardation or a flat
# curve. 1.05 and 1.10 require a deeper inversion. A floor of 0.5 keeps the
# sleeve at 2x. A threshold below 1 would cut ordinary contango and is not
# in this test.
THRESHOLDS = (1.0, 1.05, 1.10)
FLOORS = (0.0, 0.5)
WINDOW = 200
BAND = 0.02
OFF_NAME = "term_off"
_GRID_TOL = 1e-8


def cells() -> list[tuple[float, float]]:
    return [(float(threshold), float(floor)) for threshold in THRESHOLDS for floor in FLOORS]


def variant_name(threshold: float, floor: float) -> str:
    points = int(round(float(threshold) * 100))
    floor_points = int(round(float(floor) * 100))
    return f"term_t{points:03d}_f{floor_points:02d}"


def _as_close(series: pd.Series) -> pd.Series:
    close = series.astype(float).copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index))
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert("America/New_York").tz_localize(None)
    close.index = close.index.normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def term_ratio(vix_close: pd.Series, vix3m_close: pd.Series, index: pd.Index) -> pd.Series:
    """Same-session VIX / VIX3M. A missing or non-positive print is missing.

    The value at t uses the two closes dated t and nothing later. A print on
    a session that is not in ``index`` is ignored. A gap is not filled from
    the previous session.
    """
    calendar = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    left = _as_close(vix_close).reindex(calendar)
    right = _as_close(vix3m_close).reindex(calendar)
    ratio = left / right
    bad = ~(left.gt(0) & right.gt(0) & np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(right.to_numpy(dtype=float)))
    ratio = ratio.mask(bad)
    ratio.index = pd.DatetimeIndex(index)
    ratio.name = "vix_term_ratio"
    return ratio


def term_share(
    ratio: pd.Series,
    trend_on: pd.Series,
    threshold: float,
    floor: float,
) -> pd.Series:
    """UPRO share of the sleeve known at the close.

    The value at t uses the ratio and the trend state through t only. It is
    ``floor`` when the trend is on and the ratio is finite and at least
    ``threshold``. Otherwise it is 1, including when the ratio is missing.
    The share is the weight for the next session. The caller ignores it while
    the sleeve is off.
    """
    level = float(threshold)
    if not np.isfinite(level) or level < 1.0:
        raise ValueError("threshold must be finite and at least 1")
    base = float(floor)
    if not np.isfinite(base) or base < 0.0 or base >= 1.0:
        raise ValueError("floor must be in [0, 1)")
    series = ratio.astype(float).sort_index()
    values = series.to_numpy(dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("term ratio must be finite or missing")
    trend = trend_on.reindex(series.index)
    if trend.isna().any():
        raise ValueError("trend state is missing sessions")
    stressed = series.ge(level).fillna(False).to_numpy(dtype=bool) & trend.astype(bool).to_numpy()
    decided = np.where(stressed, base, 1.0).astype(float)
    decided = np.where(decided <= 1e-12, 0.0, decided)
    decided = np.where(decided >= 1.0 - 1e-12, 1.0, decided)
    return pd.Series(decided, index=series.index, name="term_share")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "term_gate"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_term_gate(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _canonical(threshold: float, floor: float) -> tuple[float, float] | None:
    for cell_threshold, cell_floor in cells():
        if abs(float(threshold) - cell_threshold) <= _GRID_TOL and abs(float(floor) - cell_floor) <= _GRID_TOL:
            return (cell_threshold, cell_floor)
    return None


def _selection(entry: dict, as_of) -> str | None:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    return chosen


def _params_for(entry: dict, name: str) -> tuple[float, float] | None:
    if name == OFF_NAME:
        return None
    for row in entry.get("variants") or []:
        if row.get("name") != name:
            continue
        threshold = row.get("threshold")
        floor = row.get("floor")
        if threshold is None or floor is None:
            raise DataError(f"kept term-structure gate {name} has no threshold or floor")
        found = _canonical(float(threshold), float(floor))
        if found is None:
            raise DataError(f"kept term-structure parameters {(float(threshold), float(floor))} are outside the grid")
        return found
    raise DataError(f"kept term-structure gate has no parameters for {name}")


def kept_term_gate(
    iterations_path,
    close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    vix3m_close: pd.Series | None = None,
    as_of=None,
):
    """UPRO share of the sleeve for a kept term-structure gate, or ``(None, None)``.

    The share is 1 when the walk-forward pick is the current book. The caller
    ignores the share while the 200-day band is off. Sessions after ``as_of``
    are not used. The band is the live 200-day 2% state of ``close``. The
    ratio uses only VIX and VIX3M closes on or before ``as_of``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    if close is None or vix_close is None or vix3m_close is None:
        raise DataError("a kept term-structure gate needs SPY, VIX, and VIX3M closes")
    prices = close.dropna().astype(float).sort_index()
    if prices.empty:
        raise DataError("no closes for the term-structure gate")
    stamp = prices.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = prices.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    ratio = term_ratio(vix_close.loc[:stamp], vix3m_close.loc[:stamp], visible.index)
    trend = trend_state(visible, WINDOW, BAND, "daily")
    states = {
        variant_name(threshold, floor): term_share(ratio, trend, threshold, floor)
        for threshold, floor in cells()
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
    series = pd.Series(values, index=visible.index, name="term_share")
    chosen = _selection(entry, visible.index[-1]) or OFF_NAME
    params = _params_for(entry, chosen)
    share_now = 1.0 if params is None else float(states[variant_name(*params)].iloc[-1])
    last_ratio = float(ratio.iloc[-1]) if np.isfinite(ratio.iloc[-1]) else None
    info = {
        "term_rule": chosen,
        "term_threshold": None if params is None else float(params[0]),
        "term_floor": None if params is None else float(params[1]),
        "term_ratio": last_ratio,
        "term_share": share_now,
        "term_on": bool(share_now < 1.0 - 1e-8),
        "term_live": params is None,
    }
    return series, info
