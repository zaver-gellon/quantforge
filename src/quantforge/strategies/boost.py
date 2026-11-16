"""SPY plus an RSI(2) sleeve in UPRO.

The named rule holds 100% SPY. When Wilder RSI(2) is below 10 and SPY is
above its 200-day average, one third of the book moves from SPY into UPRO.
It goes back to 100% SPY when RSI(2) is above 70, when SPY closes above its
5-day average, or after 10 sessions. There is no stop. An exit close does
not re-enter. The grid below is fixed.

The flag at t is the book for the interval after close t. A next-open fill
and a same-close fill use the same flag. They do not use the same price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.strategies.meanrev import wilder_rsi

THRESHOLDS = (5.0, 10.0, 15.0)
FRACTIONS = (0.25, 1.0 / 3.0, 0.5)
HOLDS = (5, 10)
EXIT_RSI = 70.0
REGIME_WINDOW = 200
EXIT_SMA = 5
NAMED = (10.0, 1.0 / 3.0, 10)


def cells() -> list[tuple[float, float, int]]:
    return [(float(threshold), float(fraction), int(hold)) for threshold in THRESHOLDS for fraction in FRACTIONS for hold in HOLDS]


def cell_name(threshold: float, fraction: float, hold: int) -> str:
    pair = _match(threshold, fraction, hold)
    pct = int(round(pair[1] * 100))
    return f"rsi{int(pair[0]):02d}_f{pct:02d}_n{int(pair[2]):02d}"


def _match(threshold: float, fraction: float, hold: int) -> tuple[float, float, int]:
    for item in cells():
        if (
            abs(item[0] - float(threshold)) < 1e-9
            and abs(item[1] - float(fraction)) < 1e-9
            and item[2] == int(hold)
        ):
            return item
    raise ValueError(f"unknown boost cell {(threshold, fraction, hold)}")


def parse_cell(name: str) -> tuple[float, float, int]:
    for threshold, fraction, hold in cells():
        if cell_name(threshold, fraction, hold) == name:
            return threshold, fraction, hold
    raise ValueError(f"unknown boost cell {name!r}")


def named_cell() -> str:
    return cell_name(*NAMED)


def hold_flag(entry: pd.Series, leave: pd.Series, max_hold: int) -> pd.Series:
    """True at t means the boost is on after this close.

    ``leave`` is tested on closes after the entry close. The exit close does
    not open a new boost. ``max_hold`` counts sessions the boost is on.
    """
    cap = int(max_hold)
    if cap < 1:
        raise ValueError("max_hold must be positive")
    index = entry.index
    entries = entry.reindex(index).fillna(False).astype(bool).to_numpy()
    leaves = leave.reindex(index).fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(index), dtype=bool)
    in_trade = False
    held = 0
    for t in range(len(index)):
        if not in_trade:
            nxt = bool(entries[t])
            if nxt:
                in_trade = True
                held = 0
        else:
            held += 1
            if bool(leaves[t]) or held >= cap:
                in_trade = False
                nxt = False
            else:
                nxt = True
        out[t] = nxt
    return pd.Series(out, index=index, name="boost_on")


def boost_flag(
    close: pd.Series,
    threshold: float = 10.0,
    max_hold: int = 10,
    exit_rsi: float = EXIT_RSI,
    regime_window: int = REGIME_WINDOW,
    exit_sma: int = EXIT_SMA,
) -> pd.Series:
    """Boost flag from the close. Above the 200-day average is required to enter."""
    series = close.astype(float).sort_index()
    rsi = wilder_rsi(series, 2)
    slow = series.rolling(int(regime_window), min_periods=int(regime_window)).mean()
    fast = series.rolling(int(exit_sma), min_periods=int(exit_sma)).mean()
    entry = (rsi < float(threshold)) & (series > slow)
    leave = (rsi > float(exit_rsi)) | (series > fast)
    return hold_flag(entry.fillna(False), leave.fillna(False), max_hold)


def book_frame(flag: pd.Series, fraction: float) -> pd.DataFrame:
    """SPY and UPRO weights. The residual, if any, is cash."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("boost fraction must be in (0, 1]")
    on = flag.fillna(False).astype(bool)
    spy = pd.Series(np.where(on, 1.0 - float(fraction), 1.0), index=flag.index, dtype=float)
    upro = pd.Series(np.where(on, float(fraction), 0.0), index=flag.index, dtype=float)
    return pd.DataFrame({"SPY": spy, "UPRO": upro})


def cash_frame(flag: pd.Series) -> pd.DataFrame:
    """100% UPRO while the boost is on, otherwise cash. The standalone 3x rule."""
    on = flag.fillna(False).astype(bool)
    upro = pd.Series(np.where(on, 1.0, 0.0), index=flag.index, dtype=float)
    return pd.DataFrame({"UPRO": upro})


def overlay_frame(
    trend_on: pd.Series,
    dip_on: pd.Series,
    boost_on: pd.Series,
    fraction: float,
    core: float = 0.5,
    sleeve: float = 0.5,
) -> pd.DataFrame:
    """Lead-book strategic weights with the boost applied to the SPY core.

    ``fraction`` of the SPY core moves into UPRO. The existing UPRO sleeve is
    not resized. The dip keeps the core in SSO and the boost does not stack
    on that stake. A trend-off sleeve stays in cash.
    """
    index = trend_on.index
    trend = trend_on.reindex(index).fillna(False).astype(bool).to_numpy()
    dip = dip_on.reindex(index).fillna(False).astype(bool).to_numpy()
    boost = boost_on.reindex(index).fillna(False).astype(bool).to_numpy()
    spy = np.zeros(len(index))
    sso = np.zeros(len(index))
    upro = np.zeros(len(index))
    core_w = float(core)
    sleeve_w = float(sleeve)
    take = float(fraction)
    for i in range(len(index)):
        if dip[i]:
            sso[i] = core_w
            upro[i] = sleeve_w if trend[i] else 0.0
            continue
        spy_w = core_w
        upro_w = sleeve_w if trend[i] else 0.0
        if boost[i]:
            move = min(take * core_w, spy_w)
            spy_w -= move
            upro_w += move
        spy[i] = spy_w
        upro[i] = upro_w
    return pd.DataFrame({"SPY": spy, "SSO": sso, "UPRO": upro}, index=index)


def close_to_close_book(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    rf: pd.Series,
    cost: float,
) -> pd.Series:
    """Weights at close t earn the close-to-close return from t to t+1.

    ``cost`` is the one-way fraction charged on each risky dollar traded.
    Cash is not charged. The cost of a trade at t is subtracted from the
    return that starts at t.
    """
    assets = [column for column in weights.columns if column in closes.columns]
    if not assets:
        raise ValueError("weights and closes share no assets")
    frame = weights[assets].astype(float).sort_index()
    asset_ret = closes[assets].reindex(frame.index).astype(float).pct_change()
    previous = frame.shift(1)
    cash = 1.0 - previous.sum(axis=1)
    yield_ = rf.reindex(frame.index).astype(float).fillna(0.0)
    gross = (previous * asset_ret).sum(axis=1) + cash * yield_
    drag = frame.diff().abs().sum(axis=1) * float(cost)
    net = gross - drag.shift(1)
    net.name = "return"
    return net.dropna()


def boost_episodes(flag: pd.Series, portfolio: pd.Series, benchmark: pd.Series) -> list[dict]:
    """One row per boost. The return at t is inside the boost when ``flag`` was true at t-1."""
    held = flag.astype(bool).reindex(portfolio.index).shift(1).fillna(False)
    bench = benchmark.reindex(portfolio.index).astype(float)
    port = portfolio.astype(float)
    episodes: list[dict] = []
    start = None
    port_rets: list[float] = []
    bench_rets: list[float] = []
    dates: list = []

    def _close(end_loc: int) -> None:
        if start is None or not port_rets:
            return
        pnl = float(np.prod(1.0 + np.asarray(port_rets)) - 1.0)
        spy = float(np.prod(1.0 + np.asarray(bench_rets)) - 1.0)
        episodes.append({
            "start": str(pd.Timestamp(dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
            "days": int(len(port_rets)),
            "pnl": pnl,
            "spy_pnl": spy,
            "win": bool(pnl > 0.0),
            "beat_spy": bool(pnl > spy),
        })

    for loc, (stamp, is_on) in enumerate(held.items()):
        if bool(is_on) and np.isfinite(port.loc[stamp]) and np.isfinite(bench.loc[stamp]):
            if start is None:
                start = loc
            port_rets.append(float(port.loc[stamp]))
            bench_rets.append(float(bench.loc[stamp]))
            dates.append(stamp)
        elif start is not None:
            _close(loc)
            start = None
            port_rets = []
            bench_rets = []
            dates = []
    if start is not None:
        _close(len(held))
    return episodes


def episode_summary(episodes: list[dict], years: float) -> dict:
    if not episodes:
        return {
            "n": 0,
            "win_rate": None,
            "beat_spy_rate": None,
            "avg_pnl": None,
            "worst_pnl": None,
            "avg_days": None,
            "per_year": 0.0 if years else None,
        }
    pnl = np.asarray([row["pnl"] for row in episodes], dtype=float)
    days = np.asarray([row["days"] for row in episodes], dtype=float)
    return {
        "n": int(len(episodes)),
        "win_rate": float(np.mean([row["win"] for row in episodes])),
        "beat_spy_rate": float(np.mean([row["beat_spy"] for row in episodes])),
        "avg_pnl": float(pnl.mean()),
        "worst_pnl": float(pnl.min()),
        "avg_days": float(days.mean()),
        "per_year": float(len(episodes) / years) if years else None,
    }
