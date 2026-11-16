"""Short-term mean reversion on SPY, held in a liquid S&P ETF.

The grid below is fixed. Do not add a cell after seeing a sample.

Enter at the close, while SPY is above its 200-day average, when one of the
precommitted entry rules is true. Hold SPY, SSO, or UPRO. Exit at the close
when SPY is back above its 5-day average, when the hold reaches N sessions,
or when SPY closes at or below its 200-day average. An exit close does not
re-enter. The decision at t is the position for the next session. Idle weight
is cash (BIL in the signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered so one step is a neighbor. RSI thresholds are a real step.
# IBS and the 5-day band are different rules that sit on the same list
# because the materiality bar needs an ordered axis. They are not a smooth scale.
ENTRIES = ("rsi2_05", "rsi2_10", "rsi2_15", "ibs_0.2", "band5")
HOLDS = (3, 5, 10)
VEHICLES = ("SPY", "SSO", "UPRO")
SMA_REGIME = 200
SMA_EXIT = 5
BAND_WINDOW = 5
BAND_WIDTH = 1.0


def cells() -> list[tuple[str, int, str]]:
    return [(entry, int(hold), vehicle) for entry in ENTRIES for hold in HOLDS for vehicle in VEHICLES]


def cell_name(entry: str, hold: int, vehicle: str) -> str:
    if entry not in ENTRIES:
        raise ValueError(f"unknown mean-reversion entry {entry!r}")
    if int(hold) not in HOLDS:
        raise ValueError(f"unknown mean-reversion hold {hold}")
    if vehicle not in VEHICLES:
        raise ValueError(f"unknown mean-reversion vehicle {vehicle!r}")
    return f"{entry}_n{int(hold):02d}_{vehicle}"


def parse_cell(name: str) -> tuple[str, int, str]:
    for entry, hold, vehicle in cells():
        if cell_name(entry, hold, vehicle) == name:
            return entry, hold, vehicle
    raise ValueError(f"unknown mean-reversion cell {name!r}")


def wilder_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Wilder RSI. The seed is the simple average of the first ``period`` changes.

    After the seed, ``avg_t = ((period - 1) * avg_{t-1} + x_t) / period``.
    The value at t uses the close at t and earlier closes only. A stretch with
    no losses is 100. A flat stretch is 50.
    """
    if period < 1:
        raise ValueError("period must be positive")
    series = close.astype(float).sort_index()
    delta = series.diff()
    gain = delta.clip(lower=0.0).to_numpy(dtype=float)
    loss = (-delta).clip(lower=0.0).to_numpy(dtype=float)
    n = len(series)
    out = np.full(n, np.nan)
    if n <= period:
        return pd.Series(out, index=series.index, name="rsi")
    seed = slice(1, period + 1)
    if not np.isfinite(gain[seed]).all() or not np.isfinite(loss[seed]).all():
        return pd.Series(out, index=series.index, name="rsi")
    avg_gain = float(np.mean(gain[seed]))
    avg_loss = float(np.mean(loss[seed]))

    def _value(up: float, down: float) -> float:
        if down == 0.0 and up == 0.0:
            return 50.0
        if down == 0.0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + up / down)

    out[period] = _value(avg_gain, avg_loss)
    width = float(period)
    for i in range(period + 1, n):
        if not np.isfinite(gain[i]) or not np.isfinite(loss[i]):
            avg_gain = np.nan
            avg_loss = np.nan
            continue
        if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
            continue
        avg_gain = ((width - 1.0) * avg_gain + gain[i]) / width
        avg_loss = ((width - 1.0) * avg_loss + loss[i]) / width
        out[i] = _value(avg_gain, avg_loss)
    return pd.Series(out, index=series.index, name="rsi")


def internal_bar_strength(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(close - low) / (high - low). A flat bar is 0.5. Uses that bar only."""
    span = high.astype(float) - low.astype(float)
    out = (close.astype(float) - low.astype(float)) / span.where(span > 0.0)
    return out.fillna(0.5)


def lower_band(close: pd.Series, window: int = BAND_WINDOW, width: float = BAND_WIDTH) -> pd.Series:
    """Simple average minus ``width`` population standard deviations of the close."""
    series = close.astype(float)
    average = series.rolling(window, min_periods=window).mean()
    deviation = series.rolling(window, min_periods=window).std(ddof=0)
    return average - float(width) * deviation


def entry_mask(close: pd.Series, high: pd.Series, low: pd.Series, rule: str) -> pd.Series:
    """True when the entry rule is true at this close. Missing inputs stay False."""
    if rule == "rsi2_05":
        mask = wilder_rsi(close, 2) < 5.0
    elif rule == "rsi2_10":
        mask = wilder_rsi(close, 2) < 10.0
    elif rule == "rsi2_15":
        mask = wilder_rsi(close, 2) < 15.0
    elif rule == "ibs_0.2":
        mask = internal_bar_strength(high, low, close) < 0.2
    elif rule == "band5":
        band = lower_band(close)
        mask = close.astype(float) < band
    else:
        raise ValueError(f"unknown mean-reversion entry {rule!r}")
    return mask.fillna(False)


def regime_on(close: pd.Series, window: int = SMA_REGIME) -> pd.Series:
    """True when the close is strictly above its simple average. No band."""
    series = close.astype(float)
    average = series.rolling(window, min_periods=window).mean()
    return (series > average).fillna(False)


def target_flag(
    close: pd.Series,
    entry: pd.Series,
    regime: pd.Series,
    max_hold: int,
    exit_window: int = SMA_EXIT,
) -> pd.Series:
    """True at t means the vehicle should be held on the next session.

    Exit when the close is above the ``exit_window`` average, when the hold
    reaches ``max_hold`` sessions, or when the regime is off. The exit close
    is not a new entry.
    """
    cap = int(max_hold)
    if cap < 1:
        raise ValueError("max_hold must be positive")
    series = close.astype(float).sort_index()
    entries = entry.reindex(series.index).fillna(False).astype(bool).to_numpy()
    allowed = regime.reindex(series.index).fillna(False).astype(bool).to_numpy()
    average = series.rolling(exit_window, min_periods=exit_window).mean().to_numpy(dtype=float)
    prices = series.to_numpy(dtype=float)
    n = len(series)
    out = np.zeros(n, dtype=bool)
    in_trade = False
    hold_count = 0
    for t in range(n):
        if not in_trade:
            nxt = bool(allowed[t] and entries[t])
            if nxt:
                in_trade = True
                hold_count = 0
        else:
            hold_count += 1
            above = np.isfinite(average[t]) and np.isfinite(prices[t]) and prices[t] > average[t]
            leave = (not allowed[t]) or (above and hold_count >= 1) or hold_count >= cap
            if leave:
                in_trade = False
                nxt = False
            else:
                nxt = True
        out[t] = nxt
    return pd.Series(out, index=series.index, name="hold_next")


def episode_stats(active: pd.Series, periods_per_year: int = 252) -> dict:
    """Entries and holding length of a next-session position flag.

    ``active[t]`` True means the risk position is on during session t+1.
    Holding days count those sessions, not the decision bar.
    """
    flag = pd.Series(active).fillna(False).astype(bool)
    values = flag.to_numpy()
    previous = np.concatenate([[False], values[:-1]])
    entries = int((values & ~previous).sum())
    held = previous  # on during this session if the prior close said so
    # The last decision has not been held yet; do not count it as a completed day.
    lengths: list[int] = []
    run = 0
    for bit in held:
        if bit:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    years = len(flag) / float(periods_per_year) if periods_per_year else float("nan")
    return {
        "n_trades": entries,
        "trades_per_year": float(entries / years) if years else float("nan"),
        "avg_holding_days": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": int(len(lengths)),
    }
