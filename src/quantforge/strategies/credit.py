"""Credit-spread gate for the leveraged sleeve.

The spread at a session is the last print dated on or before that session.
The gate uses the change versus the value ``lookback`` sessions earlier.
A later print is never filled backward.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from quantforge.data.base import DataError


def parse_fred_csv(text: str) -> pd.Series:
    """Parse a FRED graph CSV. Non-numeric cells are missing observations."""
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 2:
        raise ValueError("FRED CSV did not have a date column and a value column")
    idx = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(idx))
    series = series[series.index.notna()].dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.index = pd.DatetimeIndex(series.index).normalize()
    series.index.name = "date"
    return series


def fetch_fred_csv(series_id: str, timeout: float = 60.0) -> str:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "quantforge/0.1 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:  # noqa: BLE001 — surface a data error to the caller
        raise DataError(f"FRED fetch failed for {series_id}: {exc}") from exc
    return raw.decode("utf-8")


def snapshot_path(iterations_path, series_id: str) -> Path:
    """FRED graph CSV saved next to the iteration log."""
    return Path(iterations_path).resolve().parent / f"{series_id}.csv"


def load_spread(series_id: str, iterations_path) -> pd.Series:
    """Load a spread from the study snapshot, fetching FRED only if it is missing.

    The snapshot is the file written beside ``iterations.json``. Signal
    generation uses that file so a FRED timeout cannot fail the command or
    silently turn the hedge off.
    """
    path = snapshot_path(iterations_path, series_id)
    if path.exists():
        return parse_fred_csv(path.read_text())
    return parse_fred_csv(fetch_fred_csv(series_id))


def align_spread(spread: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Last spread print dated on or before each session. Later prints are not used."""
    clean = spread.dropna().astype(float).sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    combined = clean.index.union(pd.DatetimeIndex(sessions))
    filled = clean.reindex(combined).sort_index().ffill()
    return filled.reindex(pd.DatetimeIndex(sessions))


def twenty_day_change(aligned: pd.Series, lookback: int = 20) -> pd.Series:
    """Change versus the value ``lookback`` sessions earlier. No lead."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    return aligned.astype(float) - aligned.astype(float).shift(lookback)


def stress_on(change: pd.Series, threshold: float) -> pd.Series:
    """True when the sleeve should be in T-bills on the next session.

    A missing change is not stress. The comparison is strict: a change equal
    to the threshold leaves the sleeve invested.
    """
    return change.gt(float(threshold)).fillna(False)


def gate_trend(trend: pd.Series, stress: pd.Series) -> pd.Series:
    """Turn the leveraged sleeve off when the spread says stress."""
    flag = stress.reindex(trend.index).fillna(False).astype(bool)
    return (trend.astype(bool) & ~flag).rename("trend_on")


def _kept_entry(iterations_path: Path) -> dict | None:
    path = Path(iterations_path)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    kept = []
    for row in entries:
        if not (row.get("kept") and row.get("status") == "tested"):
            continue
        # A vol-sleeve keep, or a re-check that only records a decision, is not a spread gate.
        if row.get("overlay") not in (None, "credit_hedge"):
            continue
        if not (row.get("data") or {}).get("series"):
            continue
        kept.append(row)
    if not kept:
        return None
    return kept[-1]


def _selection(entry: dict, as_of) -> tuple[str | None, float | None]:
    """Variant selected by the latest decision date on or before ``as_of``."""
    chosen = None
    stamp = pd.Timestamp(as_of).tz_localize(None).normalize()
    for date, name in entry.get("walk_forward_picks") or []:
        if pd.Timestamp(date) <= stamp:
            chosen = name
    if chosen is None or chosen == "hedge_off":
        return chosen, None
    for row in entry.get("variants") or []:
        if row.get("name") == chosen and row.get("threshold") is not None:
            return chosen, float(row["threshold"])
    raise DataError(f"kept hedge has no threshold for {chosen}")


def kept_hedge_stress(iterations_path, sessions: pd.DatetimeIndex, as_of=None):
    """Stress flag for a kept credit-spread hedge, or ``(None, None, None)``.

    The threshold is the walk-forward pick for ``as_of``. The spread is the
    snapshot saved next to the iteration log (the public FRED CSV of the series
    named in the entry). FRED is contacted only when that file is absent. The
    returned series is indexed like ``sessions`` and is safe to slice with ``as_of``.
    """
    entry = _kept_entry(iterations_path)
    if entry is None:
        return None, None, None
    sessions = pd.DatetimeIndex(sessions)
    if len(sessions) == 0:
        raise DataError("no sessions for the credit-spread hedge")
    stamp = sessions[-1] if as_of is None else pd.Timestamp(as_of)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = pd.Timestamp(stamp).normalize()
    rule, threshold = _selection(entry, stamp)
    if rule is None or threshold is None:
        return None, rule, None
    series_id = entry.get("data", {}).get("series")
    if not series_id:
        raise DataError("kept hedge does not name a spread series")
    lookback = int(entry.get("data", {}).get("lookback") or 20)
    spread = load_spread(series_id, iterations_path)
    # Drop prints after the signal date before aligning, so a later revision
    # of the file cannot leak into an earlier as_of.
    spread = spread.loc[:stamp]
    aligned = align_spread(spread, sessions)
    change = twenty_day_change(aligned, lookback)
    stress = stress_on(change, threshold)
    visible = change.loc[:stamp].dropna()
    prints = spread.loc[:stamp]
    info = {
        "hedge_rule": rule,
        "hedge_threshold": threshold,
        "hedge_series": series_id,
        "hedge_lookback": lookback,
        "spread_change": None if visible.empty else float(visible.iloc[-1]),
        "spread_date": None if prints.empty else str(pd.Timestamp(prints.index[-1]).date()),
        "spread_value": None if prints.empty else float(prints.iloc[-1]),
    }
    return stress, rule, info
