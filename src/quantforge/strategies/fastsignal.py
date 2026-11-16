"""Live target weights for the faster books. Decisions use the close at ``as_of`` only."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantforge.calendar import next_nyse_session
from quantforge.strategies.boost import book_frame, boost_flag, named_cell, parse_cell as parse_boost
from quantforge.strategies.btctrend import parse_cell as parse_btc
from quantforge.strategies.btctrend import trend_on as btc_trend_on
from quantforge.strategies.meanrev import (
    SMA_EXIT,
    SMA_REGIME,
    cell_name as meanrev_name,
    entry_mask,
    lower_band,
    parse_cell as parse_meanrev,
    regime_on,
    target_flag,
    wilder_rsi,
    internal_bar_strength,
)
from quantforge.strategies.nasdaqvol import (
    TREND_WINDOW,
    cell_name as nasdaq_name,
    parse_cell as parse_nasdaq,
    realized_vol,
    tqqq_weight,
)
from quantforge.strategies.rotation import (
    BONDS,
    CASH,
    GOLD,
    cash_trailing_return,
    cell_name as rotation_name,
    parse_cell as parse_rotation,
    trailing_return,
    weekly_pick,
)
from quantforge.strategies.trend import trend_state


def book_weights(risky: dict[str, float], cash: str = "BIL") -> dict[str, float]:
    """Positive weights. Anything left after the risky sleeves is ``cash``.

    A zero risky weight is omitted. The cash name is omitted when the book is
    fully invested. Weights are rounded to six decimals and the largest name
    absorbs the rounding residual so the sum is at most 1.
    """
    raw = {}
    for symbol, value in risky.items():
        weight = float(value)
        if not np.isfinite(weight) or weight < -1e-12:
            raise ValueError(f"weight for {symbol} must be finite and non-negative")
        if weight > 1e-12:
            raw[str(symbol)] = weight
    total = float(sum(raw.values()))
    if total > 1.0 + 1e-8:
        raise ValueError(f"risky weights sum to {total:.6f}")
    residual = 1.0 - total
    if residual > 1e-8:
        raw[cash] = raw.get(cash, 0.0) + residual
    if not raw:
        raw[cash] = 1.0
    rounded = {symbol: round(weight, 6) for symbol, weight in raw.items() if weight > 1e-12}
    drift = round(sum(rounded.values()) - 1.0, 6)
    if drift > 0:
        largest = max(rounded, key=rounded.get)
        rounded[largest] = round(rounded[largest] - drift, 6)
        if rounded[largest] <= 0:
            del rounded[largest]
    return rounded


def _as_of(index: pd.DatetimeIndex, as_of) -> pd.Timestamp:
    idx = pd.DatetimeIndex(index)
    if as_of is None:
        return pd.Timestamp(idx[-1])
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = stamp.normalize()
    history = idx[idx <= stamp]
    if len(history) == 0:
        raise ValueError("as_of is before the price history")
    return pd.Timestamp(history[-1])


def _shell(strategy_id: str, as_of_ts: pd.Timestamp, weights: dict, levels: dict, diagnostics: dict) -> dict:
    return {
        "strategy": strategy_id,
        "strategy_id": strategy_id,
        "date": next_nyse_session(as_of_ts).strftime("%Y-%m-%d"),
        "as_of": pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        "signal_date": pd.Timestamp(as_of_ts).strftime("%Y-%m-%d"),
        "weights": weights,
        "levels": levels,
        "validated": False,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "diagnostics": diagnostics,
    }


def meanrev_signal(spy: pd.DataFrame, cell: str, as_of=None) -> dict:
    """SPY frame needs open, high, low, close. ``cell`` is a grid name."""
    entry, hold, vehicle = parse_meanrev(cell)
    close = spy["close"].dropna().astype(float).sort_index()
    high = spy["high"].reindex(close.index).astype(float)
    low = spy["low"].reindex(close.index).astype(float)
    as_of_ts = _as_of(close.index, as_of)
    visible = close.loc[:as_of_ts]
    high_v = high.loc[:as_of_ts]
    low_v = low.loc[:as_of_ts]
    regime = regime_on(visible, SMA_REGIME)
    entries = entry_mask(visible, high_v, low_v, entry)
    flag = target_flag(visible, entries, regime, hold)
    holding = bool(flag.iloc[-1])
    weights = book_weights({vehicle: 1.0} if holding else {})
    sma200 = float(visible.rolling(SMA_REGIME, min_periods=SMA_REGIME).mean().iloc[-1])
    sma5 = float(visible.rolling(SMA_EXIT, min_periods=SMA_EXIT).mean().iloc[-1])
    rsi = wilder_rsi(visible, 2)
    ibs = internal_bar_strength(high_v, low_v, visible)
    band = lower_band(visible)
    last_close = float(visible.iloc[-1])
    if entry.startswith("rsi2"):
        entry_level = float(entry.split("_")[1])
        entry_value = None if not np.isfinite(rsi.iloc[-1]) else float(rsi.iloc[-1])
        entry_text = f"RSI(2) < {entry_level:.0f}"
    elif entry == "ibs_0.2":
        entry_level = 0.2
        entry_value = float(ibs.iloc[-1])
        entry_text = "IBS < 0.2"
    else:
        entry_level = None if not np.isfinite(band.iloc[-1]) else float(band.iloc[-1])
        entry_value = last_close
        entry_text = "close below the 5-day lower band (1 sigma)"
    levels = {
        "entry": entry_text,
        "entry_level": entry_level,
        "entry_value": entry_value,
        "exit": f"SPY close above its {SMA_EXIT}-day average",
        "exit_level": sma5,
        "stop": (
            f"flat after {hold} sessions, or when SPY closes at or below its "
            f"{SMA_REGIME}-day average"
        ),
        "stop_level": sma200,
        "reference_close": last_close,
    }
    diagnostics = {
        "cell": meanrev_name(entry, hold, vehicle),
        "entry": entry,
        "max_hold": hold,
        "vehicle": vehicle,
        "regime_on": bool(regime.iloc[-1]),
        "in_position": holding,
        "sma200": sma200,
        "sma5": sma5,
        "rsi2": None if not np.isfinite(rsi.iloc[-1]) else float(rsi.iloc[-1]),
        "ibs": float(ibs.iloc[-1]),
        "lower_band": None if not np.isfinite(band.iloc[-1]) else float(band.iloc[-1]),
        "rule": (
            "Enter at the close while SPY is above its 200-day average. "
            "Exit at the next open after SPY closes above its 5-day average, "
            f"after {hold} sessions, or after a close at or below the 200-day average. "
            "Cash is BIL."
        ),
    }
    return _shell("meanrev", as_of_ts, weights, levels, diagnostics)


def rotation_signal(closes: pd.DataFrame, rf_daily: pd.Series, cell: str, as_of=None) -> dict:
    """``closes`` columns are UPRO, TQQQ, TMF, and UGL. ``rf_daily`` is the lagged bill yield."""
    equity, lookback = parse_rotation(cell)
    frame = closes.astype(float).sort_index()
    as_of_ts = _as_of(frame.index, as_of)
    visible = frame.loc[:as_of_ts]
    rf = rf_daily.reindex(visible.index).astype(float).fillna(0.0)
    scores = {
        equity: trailing_return(visible[equity], lookback),
        BONDS: trailing_return(visible[BONDS], lookback),
        GOLD: trailing_return(visible[GOLD], lookback),
    }
    hurdle = cash_trailing_return(rf, lookback)
    pick = weekly_pick(scores, hurdle, visible.index)
    choice = str(pick.iloc[-1])
    weights = book_weights({} if choice == CASH else {choice: 1.0})
    score_now = {
        equity: _finite(scores[equity].iloc[-1]),
        BONDS: _finite(scores[BONDS].iloc[-1]),
        GOLD: _finite(scores[GOLD].iloc[-1]),
        CASH: _finite(hurdle.iloc[-1]),
    }
    levels = {
        "entry": (
            f"week-ending close, top trailing {lookback}-session return among "
            f"{equity}, {BONDS}, and {GOLD}, and only if it beats cash"
        ),
        "entry_level": score_now.get(choice),
        "entry_value": score_now,
        "exit": "the next week-ending close, when the ranking is run again",
        "exit_level": None,
        "stop": "cash when the best risky sleeve does not beat the T-bill return over the same window",
        "stop_level": score_now[CASH],
        "reference_close": {name: _finite(visible[name].iloc[-1]) for name in (equity, BONDS, GOLD)},
    }
    diagnostics = {
        "cell": rotation_name(equity, lookback),
        "equity": equity,
        "lookback": lookback,
        "pick": choice,
        "scores": score_now,
        "rule": (
            "On the last NYSE session of the week, hold the sleeve with the highest "
            f"{lookback}-session total return among {equity}, TMF, and UGL, if that "
            "return is above the compounded T-bill return. Otherwise hold BIL. "
            "Fill the next open. UGL is 2x gold."
        ),
    }
    return _shell("rotation", as_of_ts, weights, levels, diagnostics)


def nasdaq_signal(qqq_close: pd.Series, tqqq_close: pd.Series, cell: str, as_of=None) -> dict:
    band, target = parse_nasdaq(cell)
    qqq = qqq_close.dropna().astype(float).sort_index()
    tqqq = tqqq_close.reindex(qqq.index).astype(float)
    as_of_ts = _as_of(qqq.index, as_of)
    qqq_v = qqq.loc[:as_of_ts]
    tqqq_v = tqqq.loc[:as_of_ts]
    weight = tqqq_weight(qqq_v, tqqq_v, band, target)
    held = float(weight.iloc[-1])
    weights = book_weights({"TQQQ": held} if held > 1e-12 else {})
    state = trend_state(qqq_v, TREND_WINDOW, band, "daily")
    average = qqq_v.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean()
    sma = float(average.iloc[-1])
    vol = realized_vol(tqqq_v)
    vol_now = _finite(vol.iloc[-1])
    last = float(qqq_v.iloc[-1])
    upper = sma * (1.0 + band)
    lower = sma * (1.0 - band)
    levels = {
        "entry": f"QQQ close above {upper:.4f} (200-day average times {1.0 + band:.2f})",
        "entry_level": upper,
        "entry_value": last,
        "exit": f"QQQ close below {lower:.4f} (200-day average times {1.0 - band:.2f})",
        "exit_level": lower,
        "stop": "no price stop; the position is resized every session to the vol target and capped at 100% TQQQ",
        "stop_level": None,
        "reference_close": last,
        "vol_target": target,
        "realized_vol": vol_now,
    }
    diagnostics = {
        "cell": nasdaq_name(band, target),
        "band": band,
        "vol_target": target,
        "trend_on": bool(state.iloc[-1]),
        "tqqq_weight": held,
        "realized_vol": vol_now,
        "sma200": sma,
        "rule": (
            "Hold TQQQ only while the 200-day band is on. Size it to "
            f"{target:.0%} / 20-day realized vol of TQQQ, capped at 100%. The rest is BIL. "
            "TQQQ before its listing is a daily-reset 3x QQQ with financing and the expense ratio."
        ),
    }
    return _shell("nasdaq-vol", as_of_ts, weights, levels, diagnostics)


def boost_signal(spy_close: pd.Series, cell: str | None = None, as_of=None) -> dict:
    """Named SPY/UPRO boost, or another grid cell. The book is for the next open."""
    threshold, fraction, hold = parse_boost(named_cell() if cell is None else cell)
    close = spy_close.dropna().astype(float).sort_index()
    as_of_ts = _as_of(close.index, as_of)
    visible = close.loc[:as_of_ts]
    flag = boost_flag(visible, threshold, hold)
    weights = book_weights(book_frame(flag, fraction).iloc[-1].to_dict())
    rsi = wilder_rsi(visible, 2)
    sma200 = float(visible.rolling(200, min_periods=200).mean().iloc[-1])
    sma5 = float(visible.rolling(5, min_periods=5).mean().iloc[-1])
    rsi_now = _finite(rsi.iloc[-1])
    held_sessions = _completed(flag)
    levels = {
        "entry": f"Wilder RSI(2) < {threshold:.0f} and SPY above its 200-day average",
        "entry_level": threshold,
        "entry_value": rsi_now,
        "exit": f"RSI(2) > 70, or SPY close above its 5-day average ({sma5:.4f}), or {hold} sessions",
        "exit_level": sma5,
        "exit_rsi": 70.0,
        "stop": "none",
        "stop_level": None,
        "reference_close": float(visible.iloc[-1]),
        "rsi2": rsi_now,
        "boost_on": bool(flag.iloc[-1]),
        "sma200": sma200,
        "sessions_held": held_sessions,
    }
    diagnostics = {
        "cell": cell_name_boost(threshold, fraction, hold),
        "fill": "next_open",
        "fraction": fraction,
        "max_hold": hold,
        "threshold": threshold,
        "boost_on": bool(flag.iloc[-1]),
        "rsi2": rsi_now,
        "sessions_held": held_sessions,
        "rule": (
            "Hold SPY. After the close, if Wilder RSI(2) is below the threshold and SPY "
            "is above its 200-day average, move that fraction from SPY into UPRO at the next open. "
            "Return to SPY when RSI(2) is above 70, SPY closes above its 5-day average, or the hold expires. "
            "No stop. This daily signal is not the 12:50 PT check."
        ),
    }
    return _shell("spy-rsi-boost", as_of_ts, weights, levels, diagnostics)


def _completed(flag: pd.Series) -> int:
    """Boosted sessions already finished as of this close.

    A true flag at t-1 means session t was held. The count is the trailing run
    of those completed sessions.
    """
    values = flag.fillna(False).astype(bool).to_numpy()
    if len(values) < 2:
        return 0
    run = 0
    for bit in values[:-1]:
        run = run + 1 if bit else 0
    return int(run)


def cell_name_boost(threshold: float, fraction: float, hold: int) -> str:
    from quantforge.strategies.boost import cell_name
    return cell_name(threshold, fraction, hold)


def btc_signal(btc_close: pd.Series, cell: str, as_of=None) -> dict:
    fast, slow = parse_btc(cell)
    close = btc_close.dropna().astype(float).sort_index()
    as_of_ts = _as_of(close.index, as_of)
    visible = close.loc[:as_of_ts]
    flag = btc_trend_on(visible, fast, slow)
    holding = bool(flag.iloc[-1])
    weights = book_weights({"BTC-USD": 1.0} if holding else {})
    fast_avg = float(visible.rolling(fast, min_periods=fast).mean().iloc[-1])
    slow_avg = float(visible.rolling(slow, min_periods=slow).mean().iloc[-1])
    last = float(visible.iloc[-1])
    levels = {
        "entry": f"BTC {fast}-day average above its {slow}-day average",
        "entry_level": slow_avg,
        "entry_value": fast_avg,
        "exit": f"BTC {fast}-day average at or below its {slow}-day average",
        "exit_level": slow_avg,
        "stop": "no price stop; the alternative is BIL",
        "stop_level": None,
        "reference_close": last,
    }
    diagnostics = {
        "cell": cell,
        "fast": fast,
        "slow": slow,
        "trend_on": holding,
        "sma_fast": fast_avg,
        "sma_slow": slow_avg,
        "rule": (
            f"Hold BTC-USD when its {fast}-day average is above its {slow}-day average. "
            "Otherwise hold BIL. The decision is the NYSE close; fill the next NYSE open."
        ),
    }
    return _shell("btc-trend", as_of_ts, weights, levels, diagnostics)


def _finite(value) -> float | None:
    number = float(value)
    if not np.isfinite(number):
        return None
    return number
