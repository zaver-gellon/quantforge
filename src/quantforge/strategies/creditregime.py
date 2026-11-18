"""Credit-regime gate for the UPRO sleeve.

The live sleeve stays on until SPY closes through its 200-day band. The
scored Baa hedge used a 20-session change in a FRED spread. This rule does
not rescore that change. It reads the relative total return of credit versus
intermediate Treasuries, using only closes dated on or before the signal.

HYG/IEF is the ratio once that ratio has a full simple average. Until then
the ratio is LQD/IEF, once that average exists. Before either average exists
the gate does not block the sleeve. When the active ratio changes, the state
is set from that close alone. After that, stress turns on below the lower
band and turns off only above the upper band. The windows and bands below
were fixed before this sample was scored. Do not add a cell after seeing
the result.

The book never holds HYG, LQD, or IEF. Stress moves only the leveraged
sleeve to T-bills. The value at t uses closes at t and earlier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantforge.data.base import DataError

# One step on one axis is a neighbor. The live book is not on either axis.
WINDOWS = (100, 150, 200)
BANDS = (0.0, 0.02)
OFF_NAME = "cr_off"
PRIMARY = "HYG"
FALLBACK = "LQD"
TREASURY = "IEF"


def cells() -> list[tuple[int, float]]:
    """New trials. The ungated book is not in this list."""
    return [(int(window), float(band)) for window in WINDOWS for band in BANDS]


def variant_name(window: int, band: float) -> str:
    lookback = int(window)
    width = float(band)
    if lookback not in WINDOWS or width not in BANDS:
        raise ValueError(f"unknown credit-regime cell {(window, band)}")
    points = int(round(width * 100))
    return f"zcr_w{lookback}_b{points:02d}"


def align_close(prices: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Last close dated on or before each session. A later print is not used."""
    clean = prices.dropna().astype(float).sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    clean.index = pd.DatetimeIndex(clean.index).normalize()
    calendar = pd.DatetimeIndex(pd.DatetimeIndex(sessions).normalize())
    combined = clean.index.union(calendar)
    filled = clean.reindex(combined).sort_index().ffill()
    return filled.reindex(calendar)


def credit_ratio(credit: pd.Series, treasury: pd.Series) -> pd.Series:
    """Credit close divided by the Treasury close. Non-positive Treasury is missing."""
    left = credit.astype(float)
    right = treasury.astype(float)
    ratio = left / right
    valid = np.isfinite(ratio.to_numpy(dtype=float)) & (right.to_numpy(dtype=float) > 0.0)
    out = ratio.where(valid)
    out.name = "credit_ratio"
    return out


def _ready_average(ratio: pd.Series, window: int) -> pd.Series:
    return ratio.astype(float).rolling(int(window), min_periods=int(window)).mean()


def regime_state(
    hyg: pd.Series,
    lqd: pd.Series,
    ief: pd.Series,
    sessions: pd.DatetimeIndex,
    window: int,
    band: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Healthy flag, stress flag, active pair, and the ratio's average.

    Healthy means the gate does not block the sleeve. Stress is the opposite.
    ``pair`` is ``HYG/IEF``, ``LQD/IEF``, or an empty string when neither
    average exists. The average is missing on those sessions.
    """
    lookback = int(window)
    if lookback < 2:
        raise ValueError("window must be at least 2")
    width = float(band)
    if width < 0.0 or width >= 1.0:
        raise ValueError("band must be in [0, 1)")
    calendar = pd.DatetimeIndex(pd.DatetimeIndex(sessions).normalize())
    if calendar.has_duplicates:
        raise ValueError("sessions must be unique")
    hyg_px = align_close(hyg, calendar)
    lqd_px = align_close(lqd, calendar)
    ief_px = align_close(ief, calendar)
    primary = credit_ratio(hyg_px, ief_px)
    fallback = credit_ratio(lqd_px, ief_px)
    primary_avg = _ready_average(primary, lookback)
    fallback_avg = _ready_average(fallback, lookback)
    use_primary = primary_avg.notna().to_numpy(dtype=bool)
    use_fallback = (~primary_avg.notna() & fallback_avg.notna()).to_numpy(dtype=bool)
    primary_px = primary.to_numpy(dtype=float)
    fallback_px = fallback.to_numpy(dtype=float)
    primary_mu = primary_avg.to_numpy(dtype=float)
    fallback_mu = fallback_avg.to_numpy(dtype=float)
    healthy = True
    previous = ""
    health = np.empty(len(calendar), dtype=bool)
    pairs = np.empty(len(calendar), dtype=object)
    averages = np.full(len(calendar), np.nan)
    ratios = np.full(len(calendar), np.nan)
    upper = 1.0 + width
    lower = 1.0 - width
    for i in range(len(calendar)):
        if use_primary[i]:
            pair = "HYG/IEF"
            price = primary_px[i]
            average = primary_mu[i]
        elif use_fallback[i]:
            pair = "LQD/IEF"
            price = fallback_px[i]
            average = fallback_mu[i]
        else:
            healthy = True
            previous = ""
            health[i] = True
            pairs[i] = ""
            continue
        above = bool(price > average * upper)
        below = bool(price < average * lower)
        if pair != previous:
            if above:
                healthy = True
            elif below:
                healthy = False
            else:
                healthy = True
            previous = pair
        elif healthy and below:
            healthy = False
        elif (not healthy) and above:
            healthy = True
        health[i] = healthy
        pairs[i] = pair
        averages[i] = average
        ratios[i] = price
    index = pd.DatetimeIndex(calendar)
    healthy_s = pd.Series(health, index=index, name="credit_healthy")
    stress = pd.Series(~health, index=index, name="credit_stress")
    pair_s = pd.Series(pairs, index=index, name="credit_pair")
    average_s = pd.Series(averages, index=index, name="credit_sma")
    ratio_s = pd.Series(ratios, index=index, name="credit_ratio")
    average_s.attrs["ratio"] = ratio_s
    return healthy_s, stress, pair_s, average_s


def pair_ready_dates(
    hyg: pd.Series,
    lqd: pd.Series,
    ief: pd.Series,
    sessions: pd.DatetimeIndex,
    windows: tuple[int, ...] = WINDOWS,
) -> dict[str, str | None]:
    """First session each average exists. Prices only; no return is scored."""
    calendar = pd.DatetimeIndex(pd.DatetimeIndex(sessions).normalize())
    hyg_px = align_close(hyg, calendar)
    lqd_px = align_close(lqd, calendar)
    ief_px = align_close(ief, calendar)
    primary = credit_ratio(hyg_px, ief_px)
    fallback = credit_ratio(lqd_px, ief_px)
    found: dict[str, str | None] = {}
    for window in windows:
        for label, ratio in ((PRIMARY, primary), (FALLBACK, fallback)):
            ready = _ready_average(ratio, int(window)).dropna()
            found[f"{label}_w{int(window)}"] = None if ready.empty else str(ready.index[0].date())
    return found


def _kept_entry(iterations_path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = [
        row for row in entries
        if row.get("kept") and row.get("status") == "tested" and row.get("overlay") == "credit_regime"
    ]
    if not kept:
        return None
    return kept[-1]


def has_kept_credit_regime(iterations_path) -> bool:
    return _kept_entry(iterations_path) is not None


def _selection(entry: dict, as_of) -> tuple[str | None, int | None, float | None]:
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
        window = row.get("window")
        band = row.get("band")
        if window is None or band is None:
            raise DataError(f"kept credit regime {chosen} has no window and band")
        lookback = int(window)
        width = float(band)
        if lookback not in WINDOWS or width not in BANDS:
            raise DataError(f"kept credit regime {(lookback, width)} is outside the grid")
        return chosen, lookback, width
    raise DataError(f"kept credit regime has no parameters for {chosen}")


def kept_credit_regime(
    iterations_path,
    sessions: pd.DatetimeIndex,
    hyg: pd.Series,
    lqd: pd.Series,
    ief: pd.Series,
    as_of=None,
):
    """Stress flag for a kept credit-regime gate, or ``(None, None, None)``.

    The cell is the walk-forward pick for ``as_of``. Closes after ``as_of``
    are dropped before the state is built. The live cell is not an override.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None, None
    calendar = pd.DatetimeIndex(sessions)
    if len(calendar) == 0:
        raise DataError("no sessions for the credit-regime gate")
    stamp = calendar[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    visible_sessions = calendar[calendar <= stamp]
    if len(visible_sessions) == 0:
        raise DataError("as_of is before the price history")
    rule, window, band = _selection(entry, visible_sessions[-1])
    if rule is None or window is None or band is None:
        return None, None, None
    hyg_v = hyg.loc[:stamp]
    lqd_v = lqd.loc[:stamp]
    ief_v = ief.loc[:stamp]
    _healthy, stress, pairs, averages = regime_state(
        hyg_v, lqd_v, ief_v, visible_sessions, window, band,
    )
    ratio = averages.attrs.get("ratio")
    last_pair = str(pairs.iloc[-1])
    last_ratio = None if ratio is None or not np.isfinite(ratio.iloc[-1]) else float(ratio.iloc[-1])
    last_average = None if not np.isfinite(averages.iloc[-1]) else float(averages.iloc[-1])
    info = {
        "hedge_rule": rule,
        "credit_rule": rule,
        "credit_window": int(window),
        "credit_band": float(band),
        "credit_pair": last_pair or None,
        "credit_ratio": last_ratio,
        "credit_sma": last_average,
        "credit_stress": bool(stress.iloc[-1]),
    }
    return stress, rule, info


def merge_hedge(hedge, regime):
    """Prefer a kept regime over the rejected Baa hedge. Both kept is an error."""
    hedge_stress, hedge_rule, hedge_info = hedge
    regime_stress, regime_rule, regime_info = regime
    if regime_stress is not None and hedge_stress is not None:
        raise RuntimeError("the iteration log kept both a Baa credit hedge and a credit-regime gate")
    if regime_stress is None:
        return hedge_stress, hedge_rule, hedge_info
    return regime_stress, regime_rule, regime_info
