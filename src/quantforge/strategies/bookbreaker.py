"""Book-level drawdown breaker for the UPRO sleeve.

The 2026-10-29 stop cuts the sleeve when SPY's price is a set distance below
its own trailing high. This rule does not rescore that price stop. It watches
the trend-core account. The running peak starts at the inception cash value
of 1 and ratchets up when the account makes a new high. The drawdown is the
account's equity divided by that peak, minus one. Equity is marked at the
close, and the decision fills at the next open.

The cuts and the two re-entry modes below were fixed before this sample was
scored. Do not add a cell after seeing the result. The half-cut heal is part
of the recover mode. It is not a third axis. The current book is not a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on one axis is a neighbor. The live book is not on either axis.
CUTS = (0.15, 0.20, 0.25)
MODES = ("confirm", "recover")
REENTRY_FRACTION = 0.5
OFF_NAME = "bk_off"
ONE_WAY_COST = 0.0005


def cells() -> list[tuple[float, str]]:
    """New trials. The unbraked book is not in this list."""
    return [(float(cut), mode) for cut in CUTS for mode in MODES]


def variant_name(cut: float, mode: str) -> str:
    width = float(cut)
    label = str(mode)
    if width not in CUTS or label not in MODES:
        raise ValueError(f"unknown book-breaker cell {(cut, mode)}")
    points = int(round(width * 100))
    return f"zbk_c{points:02d}_{label}"


def step_breaker(
    tripped: bool,
    seen_off: bool,
    drawdown: float,
    trend_on: bool,
    cut: float,
    mode: str,
    heal: float,
) -> tuple[bool, bool]:
    """State after this close. The result governs the next session.

    ``drawdown`` is equity / peak - 1 at this close. ``trend_on`` is the
    200-day state known at this close. ``heal`` is the recover-mode band,
    expressed as a positive fraction of equity (half the cut). confirm ignores
    ``heal``: once tripped, the trend must print an off close and then a later
    on close before the sleeve may hold UPRO again.
    """
    label = str(mode)
    if label not in MODES:
        raise ValueError(f"unknown re-entry mode {mode!r}")
    threshold = float(cut)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("cut must be in (0, 1)")
    healed = float(heal)
    if not np.isfinite(healed) or healed < 0.0 or healed >= threshold:
        raise ValueError("heal must lie in [0, cut)")
    loss = float(drawdown)
    if not np.isfinite(loss):
        raise ValueError("drawdown must be finite")
    on = bool(trend_on)
    if label == "recover":
        if (not tripped) and loss <= -threshold:
            return True, False
        if tripped and loss >= -healed and on:
            return False, False
        return bool(tripped), False
    if not tripped:
        if loss <= -threshold:
            return True, (not on)
        return False, False
    if not on:
        return True, True
    if seen_off:
        return False, False
    return True, bool(seen_off)


def book_drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown of wealth that starts at 1. A missing return leaves wealth unchanged.

    This is the same peak the breaker uses when the path is the account: the
    peak starts at 1 and updates after each finite return. It is not SPY's
    price and it is not a trailing window.
    """
    series = returns.astype(float)
    wealth = 1.0
    peak = 1.0
    out = np.empty(len(series), dtype=float)
    for i, value in enumerate(series.to_numpy(dtype=float)):
        if np.isfinite(value):
            wealth *= 1.0 + float(value)
        if wealth > peak:
            peak = wealth
        if peak <= 0.0 or not np.isfinite(wealth):
            raise ValueError("book equity is not a positive finite number")
        out[i] = wealth / peak - 1.0
    return pd.Series(out, index=series.index, name="book_drawdown")


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "book_breaker"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_book_breaker(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _selection(entry: dict, as_of) -> tuple[str | None, float | None, str | None]:
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == OFF_NAME:
        return None, None, None
    for row in entry.get("variants") or []:
        if row.get("name") != chosen:
            continue
        cut = row.get("cut")
        mode = row.get("reentry")
        if cut is None or mode is None:
            raise DataError(f"kept book breaker {chosen} has no cut and re-entry")
        width = float(cut)
        label = str(mode)
        if width not in CUTS or label not in MODES:
            raise DataError(f"kept book breaker {(width, label)} is outside the grid")
        return chosen, width, label
    raise DataError(f"kept book breaker has no parameters for {chosen}")


def kept_book_breaker(
    iterations_path,
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    cash_yield: pd.Series,
    as_of=None,
):
    """Sleeve scale for a kept book breaker, or ``(None, None)``.

    The cell is the walk-forward pick for ``as_of``. Prices after ``as_of``
    are dropped before the account is resimulated. The scale is 0 while the
    breaker is on and 1 otherwise. The 200-day gate still applies on top: a
    scale of 1 with the trend off holds the sleeve in T-bills. The live cell
    is not an override.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None
    closes = spy_close.dropna().astype(float).sort_index()
    if closes.empty:
        raise DataError("no SPY closes for the book breaker")
    stamp = closes.index[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible = closes.loc[:stamp]
    if visible.empty:
        raise DataError("as_of is before the price history")
    rule, cut, mode = _selection(entry, visible.index[-1])
    if rule is None or cut is None or mode is None:
        return None, None
    from quantforge.engine.backtest import run_book_breaker
    from quantforge.engine.schedule import rebalance_mask
    from quantforge.strategies.dipbuy import dip_active
    from quantforge.strategies.trend import trend_state

    idx = visible.index
    opens = spy_open.reindex(idx).astype(float)
    sso_o = sso_open.reindex(idx).astype(float)
    sso_c = sso_close.reindex(idx).astype(float)
    upro_o = upro_open.reindex(idx).astype(float)
    upro_c = upro_close.reindex(idx).astype(float)
    rf = cash_yield.reindex(idx).astype(float)
    if opens.isna().any() or sso_o.isna().any() or sso_c.isna().any() or upro_o.isna().any() or upro_c.isna().any():
        raise DataError("book breaker prices are missing sessions at or before as_of")
    if rf.isna().any():
        raise DataError("book breaker cash yield is missing sessions at or before as_of")
    trend = trend_state(visible, 200, 0.02, "daily")
    dip = dip_active(visible, "3_down_closes").reindex(idx).fillna(False)
    mix = rebalance_mask(idx, "quarterly")
    _result, breaker, _sleeve, drawdown = run_book_breaker(
        opens, visible, sso_o, sso_c, upro_o, upro_c,
        trend, dip, mix, cut, mode, 0.5, 0.5, rf,
        spy_cost=ONE_WAY_COST, sso_cost=ONE_WAY_COST, upro_cost=ONE_WAY_COST,
        reentry_fraction=REENTRY_FRACTION,
    )
    scale = (~breaker).astype(float)
    scale.name = "sleeve_scale"
    info = {
        "book_breaker_rule": rule,
        "book_breaker_cut": float(cut),
        "book_breaker_reentry": mode,
        "book_breaker_heal": None if mode != "recover" else float(cut) * REENTRY_FRACTION,
        "book_breaker_on": bool(breaker.iloc[-1]),
        "book_drawdown": float(drawdown.iloc[-1]),
    }
    return scale, info
