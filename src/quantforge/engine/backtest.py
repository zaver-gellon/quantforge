"""Long-only backtests. A decision at the close of bar t trades on bar t+1.

``next_open`` (the default) fills at the next open: the overnight gap into
that session is earned by the previous holdings, and the new holdings earn the
open-to-close move. ``next_close`` is a robustness fill: the whole close-to-close
move is earned by the previous holdings, and the new holdings start afterwards.

Cash earns ``cash_yield`` over the overnight into the session. The yield at
date t must already be known at the previous close (callers lag T-bill prints).
Uninvested weight is cash. Weights are long-only and sum to at most 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantforge.engine.costs import CostModel


@dataclass
class BacktestResult:
    returns: pd.Series
    turnover: pd.Series
    cost_drag: pd.Series
    exposure: pd.Series


def _align_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_index()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def run_backtest(
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    targets: pd.DataFrame,
    rebalance: pd.Series,
    cost: CostModel,
    cash_yield: pd.Series,
    execution: str = "next_open",
    max_gross: float = 1.0,
    allow_short: bool = False,
    borrow_spread: float = 0.0,
    short_borrow: float = 0.0,
    impact: pd.DataFrame | None = None,
) -> BacktestResult:
    """Trade ``targets`` on the sessions after ``rebalance`` is True.

    ``targets.loc[t]`` are portfolio weights decided at the close of t.
    They are filled on the next session. Between fills, holdings drift and
    cash earns the yield. ``max_gross`` is the cap on the sum of absolute
    weights. Above 1, the residual cash is negative and, when
    ``borrow_spread`` is set, that debit pays the cash yield plus the spread.
    ``borrow_spread`` and ``short_borrow`` are daily rates. Short borrow is
    charged on the absolute short notional carried from the previous close.
    ``impact`` is an extra one-way cost fraction by decision date and asset,
    known at that close (a market-impact model, not a second signal).
    Short weights require ``allow_short``.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")
    open_px = _align_frame(open_px)
    close_px = _align_frame(close_px).reindex(open_px.index)
    assets = list(open_px.columns)
    if list(close_px.columns) != assets:
        close_px = close_px.reindex(columns=assets)
    targets = targets.copy()
    targets.index = pd.DatetimeIndex(pd.to_datetime(targets.index))
    unknown = [col for col in targets.columns if col not in assets and float(targets[col].abs().sum()) > 0]
    if unknown:
        raise ValueError(f"targets include assets without prices: {unknown}")
    target_w = targets.reindex(index=open_px.index, columns=assets).fillna(0.0).to_numpy(dtype=float)
    if not allow_short and np.any(target_w < -1e-10):
        raise ValueError("short weights are not supported")
    if np.any(np.abs(target_w).sum(axis=1) > max_gross + 1e-6):
        raise ValueError(f"target weights exceed max gross {max_gross}")
    reb = rebalance.reindex(open_px.index).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(open_px.index).astype(float).fillna(0.0).to_numpy(dtype=float)
    opens = open_px.to_numpy(dtype=float)
    closes = close_px.to_numpy(dtype=float)
    if np.any(~np.isfinite(opens)) or np.any(~np.isfinite(closes)) or np.any(opens <= 0) or np.any(closes <= 0):
        raise ValueError("open and close prices must be finite and positive")
    asset_costs = np.array([cost.one_way_fraction(asset) for asset in assets], dtype=float)
    cash_cost = cost.one_way_fraction("CASH")
    impact_arr = None
    if impact is not None:
        impact_arr = impact.reindex(index=open_px.index, columns=assets).fillna(0.0).to_numpy(dtype=float)
        impact_arr = np.maximum(impact_arr, 0.0)
    n, k = opens.shape
    held = np.zeros(k)
    cash = 1.0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = float(held.sum() + cash)
        if short_borrow > 0.0 and np.any(held < 0.0):
            fee = float((-np.minimum(held, 0.0)).sum()) * float(short_borrow)
            cash -= fee
            drag[i] += fee / prev if prev else 0.0
        rate = rf[i] + (borrow_spread if cash < -1e-12 else 0.0)
        trade_costs = asset_costs
        if impact_arr is not None:
            trade_costs = asset_costs + impact_arr[i - 1]
        if execution == "next_open":
            held *= opens[i] / closes[i - 1]
            cash *= 1.0 + rate
            if reb[i - 1]:
                held, cash, traded, cost_drag = _trade_to(held, cash, target_w[i - 1], trade_costs, cash_cost)
                turn[i] = traded
                drag[i] += cost_drag
            held *= closes[i] / opens[i]
        else:
            held *= closes[i] / closes[i - 1]
            cash *= 1.0 + rate
            if reb[i - 1]:
                held, cash, traded, cost_drag = _trade_to(held, cash, target_w[i - 1], trade_costs, cash_cost)
                turn[i] = traded
                drag[i] += cost_drag
        new = float(held.sum() + cash)
        rets[i] = new / prev - 1.0
        exposure[i] = float(held.sum() / new) if new else 0.0
    index = open_px.index
    return BacktestResult(
        returns=pd.Series(rets, index=index, name="return"),
        turnover=pd.Series(turn, index=index, name="turnover"),
        cost_drag=pd.Series(drag, index=index, name="cost_drag"),
        exposure=pd.Series(exposure, index=index, name="exposure"),
    )


def run_core_satellite(
    core_open: pd.Series,
    core_close: pd.Series,
    sleeve_open: pd.Series,
    sleeve_close: pd.Series,
    trend_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    cash_yield: pd.Series,
    core_cost: float = 0.0,
    sleeve_cost: float = 0.0,
    cash_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """SPY-style core plus a leveraged sleeve that is either invested or in cash.

    ``trend_on[t]`` True means the sleeve should be in the leveraged asset for
    the next session. ``mix_rebalance[t]`` True resets the book to
    ``core_weight`` / ``1 - core_weight`` on the next session. A trend flip
    that is not a mix day moves only the sleeve, leaving the drifted core in
    place. Costs are one-way fractions of the dollars traded in that pile.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if not 0.0 <= core_weight <= 1.0:
        raise ValueError("core_weight must be in [0, 1]")
    idx = core_close.index
    co = core_open.reindex(idx).to_numpy(dtype=float)
    cc = core_close.reindex(idx).to_numpy(dtype=float)
    so = sleeve_open.reindex(idx).to_numpy(dtype=float)
    sc = sleeve_close.reindex(idx).to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(co))
        or np.any(~np.isfinite(cc))
        or np.any(~np.isfinite(so))
        or np.any(~np.isfinite(sc))
        or np.any(co <= 0)
        or np.any(cc <= 0)
        or np.any(so <= 0)
        or np.any(sc <= 0)
    ):
        raise ValueError("core and sleeve prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    lev = 0.0
    cash = 1.0
    in_lev = False
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + lev + cash
        if execution == "next_open":
            core *= co[i] / cc[i - 1]
            if lev != 0.0:
                lev *= so[i] / sc[i - 1]
            cash *= 1.0 + rf[i]
            core, lev, cash, in_lev, turn[i], drag[i] = _maybe_rebalance(
                i, core, lev, cash, in_lev, trend[i - 1], mix[i - 1],
                core_weight, core_cost, sleeve_cost, cash_cost,
            )
            core *= cc[i] / co[i]
            if in_lev and lev != 0.0:
                lev *= sc[i] / so[i]
        else:
            core *= cc[i] / cc[i - 1]
            if lev != 0.0:
                lev *= sc[i] / sc[i - 1]
            cash *= 1.0 + rf[i]
            core, lev, cash, in_lev, turn[i], drag[i] = _maybe_rebalance(
                i, core, lev, cash, in_lev, trend[i - 1], mix[i - 1],
                core_weight, core_cost, sleeve_cost, cash_cost,
            )
        new = core + lev + cash
        rets[i] = new / prev - 1.0
        exposure[i] = (core + lev) / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def _maybe_rebalance(
    i: int,
    core: float,
    lev: float,
    cash: float,
    in_lev: bool,
    want_lev: bool,
    do_mix: bool,
    core_weight: float,
    core_cost: float,
    sleeve_cost: float,
    cash_cost: float,
):
    if i != 1 and not do_mix and want_lev == in_lev:
        return core, lev, cash, in_lev, 0.0, 0.0
    equity = core + lev + cash
    if i == 1 or do_mix:
        tgt_core = core_weight * equity
        sleeve_nav = (1.0 - core_weight) * equity
    else:
        tgt_core = core
        sleeve_nav = lev + cash
    tgt_lev = sleeve_nav if want_lev else 0.0
    tgt_cash = 0.0 if want_lev else sleeve_nav
    traded_core = abs(tgt_core - core)
    traded_lev = abs(tgt_lev - lev)
    traded_cash = abs(tgt_cash - cash)
    cost = traded_core * core_cost + traded_lev * sleeve_cost + traded_cash * cash_cost
    scale = (equity - cost) / equity if equity else 1.0
    turnover = (traded_core + traded_lev + traded_cash) / 2.0 / equity if equity else 0.0
    return (
        tgt_core * scale,
        tgt_lev * scale,
        tgt_cash * scale,
        bool(want_lev),
        turnover,
        cost / equity if equity else 0.0,
    )


def _trade_to(held, cash, target_w, asset_costs, cash_cost):
    equity = float(held.sum() + cash)
    if equity <= 0:
        raise ValueError("equity is non-positive")
    desired = target_w * equity
    desired_cash = (1.0 - float(target_w.sum())) * equity
    traded = np.abs(desired - held)
    cash_traded = abs(desired_cash - cash)
    cost = float(traded @ asset_costs + cash_traded * cash_cost)
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = float(traded.sum() + cash_traded) / 2.0 / equity
    return desired * scale, desired_cash * scale, turnover, cost / equity


def run_trend_core(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """50/50 trend book with the dip overlay on the core only.

    ``trend_on[t]`` puts the sleeve in UPRO for the next session; otherwise the
    sleeve is cash. ``dip_on[t]`` swaps the core from SPY to SSO for the next
    session and does not resize the sleeve. ``mix_rebalance[t]`` resets both
    piles to ``core_weight`` / ``sleeve_weight``. A dip flip leaves the UPRO
    pile's dollars where they were, apart from its share of any trading cost.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + sleeve + resid
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_upro, turn[i], drag[i] = _allocate_trend(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_upro,
                bool(dip[i - 1]), bool(trend[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_upro, turn[i], drag[i] = _allocate_trend(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_upro,
                bool(dip[i - 1]), bool(trend[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_drift_band(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_on: pd.Series,
    band: float,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> tuple[BacktestResult, pd.Series]:
    """Trend-core book that resets the mix only when UPRO is overweight.

    There is no calendar reset. After the close, if the sleeve is in UPRO and
    its weight is strictly above ``sleeve_weight + band``, the next open resets
    both piles to ``core_weight`` / ``sleeve_weight``. A sleeve in T-bills is
    never trimmed, and a sleeve that has fallen below the target is not topped
    up. A trend flip still swaps the sleeve instrument without resizing the
    piles. The first session is the initial allocation. The returned series is
    true on the close that calls for a reset at the next open. A 1e-12
    tolerance keeps a reset from retriggering on floating-point dust.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    width = float(band)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("drift band must be finite and non-negative")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    pending = False
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    mix_next = np.zeros(n, dtype=bool)
    ceiling = float(sleeve_weight) + width
    for i in range(1, n):
        prev = core + sleeve + resid
        do_mix = bool(i == 1 or pending)
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_upro, turn[i], drag[i] = _allocate_trend(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_upro,
                bool(dip[i - 1]), bool(trend[i - 1]), do_mix,
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_upro, turn[i], drag[i] = _allocate_trend(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_upro,
                bool(dip[i - 1]), bool(trend[i - 1]), do_mix,
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
        pending = bool(sleeve_upro and new > 0.0 and sleeve / new > ceiling + 1e-12)
        mix_next[i] = pending
    result = BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )
    return result, pd.Series(mix_next, index=idx, name="drift_mix")


def run_dip_size(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_boost: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book whose dip holds a fraction of the core in SSO.

    ``dip_boost[t]`` is the share of the core pile held in SSO on the next
    session. The rest of that pile stays in SPY. A boost of 0 or 1 is the
    boolean overlay. A value in between splits the pile and does not rebalance
    the split back to that fraction until the boost changes or the mix resets.
    The sleeve is unchanged. A boost of 1 on the same dates as ``dip_on``
    matches ``run_trend_core``.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    raw_boost = dip_boost.reindex(idx).astype(float)
    if raw_boost.isna().any() or not np.isfinite(raw_boost.to_numpy(dtype=float)).all():
        raise ValueError("dip boost is missing sessions")
    if (raw_boost < -1e-12).any() or (raw_boost > 1.0 + 1e-12).any():
        raise ValueError("dip boost must be in [0, 1]")
    boost_np = raw_boost.clip(0.0, 1.0).to_numpy(dtype=float)
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core_spy = 0.0
    core_sso = 0.0
    sleeve = 0.0
    resid = 1.0
    held_boost = 0.0
    sleeve_upro = False
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core_spy + core_sso + sleeve + resid
        want_boost = float(boost_np[i - 1])
        want_trend = bool(trend[i - 1])
        do_mix = bool(i == 1 or mix[i - 1])
        if execution == "next_open":
            core_spy = _mark(core_spy, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], False)
            core_sso = _mark(core_sso, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], True)
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core_spy, core_sso, sleeve, resid, held_boost, sleeve_upro, turn[i], drag[i] = _allocate_dip(
                core_spy + core_sso + sleeve + resid,
                core_spy, core_sso, sleeve, resid, held_boost, sleeve_upro,
                want_boost, want_trend, do_mix,
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core_spy = _mark(core_spy, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], False)
            core_sso = _mark(core_sso, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], True)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core_spy = _mark(core_spy, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], False)
            core_sso = _mark(core_sso, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], True)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core_spy, core_sso, sleeve, resid, held_boost, sleeve_upro, turn[i], drag[i] = _allocate_dip(
                core_spy + core_sso + sleeve + resid,
                core_spy, core_sso, sleeve, resid, held_boost, sleeve_upro,
                want_boost, want_trend, do_mix,
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core_spy + core_sso + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core_spy + core_sso + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_core_hold(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    off_core_fraction: float,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can park part of the core in T-bills when the trend is off.

    ``off_core_fraction`` is the share of the core pile left in SPY (or SSO while
    the dip is on) on the next session when ``trend_on`` is false. The rest of
    that pile earns the cash yield. A fraction of 1 leaves the core fully
    invested in both states, which is the unscaled trend-core book. While the
    trend is on the fraction is ignored and the core pile stays fully invested.
    A trend flip resizes only that invested share and moves the sleeve. It does
    not reset drifted pile sizes. A mix reset still does. Cash pays no spread.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    fraction = float(off_core_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("off_core_fraction must be in [0, 1]")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    core_cash = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    held_frac = 1.0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + core_cash + sleeve + resid
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            core_cash *= 1.0 + rf[i]
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, core_cash, sleeve, resid, core_sso, sleeve_upro, held_frac, turn[i], drag[i] = _allocate_core_hold(
                core + core_cash + sleeve + resid, core, core_cash, sleeve, resid,
                core_sso, sleeve_upro, held_frac,
                bool(dip[i - 1]), bool(trend[i - 1]), fraction, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            core_cash *= 1.0 + rf[i]
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, core_cash, sleeve, resid, core_sso, sleeve_upro, held_frac, turn[i], drag[i] = _allocate_core_hold(
                core + core_cash + sleeve + resid, core, core_cash, sleeve, resid,
                core_sso, sleeve_upro, held_frac,
                bool(dip[i - 1]), bool(trend[i - 1]), fraction, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + core_cash + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_fast_reentry(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    fast_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can put the core back to work while the slow band is off.

    ``trend_on[t]`` is the sleeve filter, known at the close of t and filled at
    the next open. While it is on, the book is the ordinary trend core and
    ``fast_on`` is ignored. While it is off, the sleeve stays in T-bills and the
    core is invested only when ``fast_on[t]`` is true. Otherwise that pile earns
    the cash yield. The dip applies only to the invested core. A flip resizes
    the invested share of the core and moves the sleeve. It does not reset
    drifted pile sizes. A mix reset still does. Cash pays no spread.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    fast = fast_on.reindex(idx)
    if fast.isna().any():
        raise ValueError("fast_on is missing sessions")
    fast = fast.astype(bool).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    core_cash = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    held_frac = 1.0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + core_cash + sleeve + resid
        fraction = 1.0 if fast[i - 1] else 0.0
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            core_cash *= 1.0 + rf[i]
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, core_cash, sleeve, resid, core_sso, sleeve_upro, held_frac, turn[i], drag[i] = _allocate_core_hold(
                core + core_cash + sleeve + resid, core, core_cash, sleeve, resid,
                core_sso, sleeve_upro, held_frac,
                bool(dip[i - 1]), bool(trend[i - 1]), fraction, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            core_cash *= 1.0 + rf[i]
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, core_cash, sleeve, resid, core_sso, sleeve_upro, held_frac, turn[i], drag[i] = _allocate_core_hold(
                core + core_cash + sleeve + resid, core, core_cash, sleeve, resid,
                core_sso, sleeve_upro, held_frac,
                bool(dip[i - 1]), bool(trend[i - 1]), fraction, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + core_cash + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_sleeve_gear(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    fast_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can hold the sleeve in SPY instead of UPRO.

    ``trend_on[t]`` is the sleeve's slow filter, known at the close of t and
    filled at the next open. While it is off, the sleeve is in T-bills and
    ``fast_on`` is ignored. While it is on and ``fast_on[t]`` is true, the
    sleeve is UPRO. While it is on and ``fast_on[t]`` is false, the sleeve is
    SPY. The dip applies only to the core pile. The sleeve's SPY stake is not
    dipped and is not merged with the core, so a mix reset still restores the
    two piles. A gear change swaps the sleeve instrument and does not by itself
    resize drifted piles. Cash pays no spread.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    fast = fast_on.reindex(idx)
    if fast.isna().any():
        raise ValueError("fast_on is missing sessions")
    fast = fast.astype(bool).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    # 0 cash, 1 SPY, 2 UPRO. The slow trend off forces cash.
    gear = np.where(trend, np.where(fast, 2, 1), 0).astype(int)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_gear = 0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + sleeve + resid
        want = int(gear[i - 1])
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            sleeve = _mark_gear(
                sleeve, prices["spy_o"][i], prices["spy_c"][i - 1],
                prices["upro_o"][i], prices["upro_c"][i - 1], sleeve_gear, 1.0 + rf[i],
            )
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_gear, turn[i], drag[i] = _allocate_sleeve_gear(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_gear,
                bool(dip[i - 1]), want, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            sleeve = _mark_gear(
                sleeve, prices["spy_c"][i], prices["spy_o"][i],
                prices["upro_c"][i], prices["upro_o"][i], sleeve_gear, 1.0,
            )
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            sleeve = _mark_gear(
                sleeve, prices["spy_c"][i], prices["spy_c"][i - 1],
                prices["upro_c"][i], prices["upro_c"][i - 1], sleeve_gear, 1.0 + rf[i],
            )
            resid *= 1.0 + rf[i]
            core, sleeve, resid, core_sso, sleeve_gear, turn[i], drag[i] = _allocate_sleeve_gear(
                core + sleeve + resid, core, sleeve, resid, core_sso, sleeve_gear,
                bool(dip[i - 1]), want, bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if sleeve_gear in (1, 2) else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_curve_clock(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    cash_on: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can park the whole book in T-bills.

    ``cash_on[t]`` is known at the close of t and is filled at the next open.
    While it is true, the core and the sleeve are both sold and earn the cash
    yield. The dip and the trend are ignored until the clock is off. Coming
    back out of cash resets both piles to ``core_weight`` / ``sleeve_weight``
    under the trend and dip state at that close. A cash flag that is false on
    every session is the unscaled trend-core book. Cash pays no spread.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    clock = cash_on.reindex(idx)
    if clock.isna().any():
        raise ValueError("cash_on is missing sessions")
    clock = clock.astype(bool).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    in_cash = False
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + sleeve + resid
        want_cash = bool(clock[i - 1])
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_upro:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            else:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
        equity = core + sleeve + resid
        if want_cash:
            core, sleeve, resid, core_sso, sleeve_upro, in_cash, turn[i], drag[i] = _allocate_flat_cash(
                equity, core, sleeve, resid, core_sso, sleeve_upro, in_cash,
                spy_cost, sso_cost, upro_cost,
            )
        else:
            core, sleeve, resid, core_sso, sleeve_upro, turn[i], drag[i] = _allocate_trend(
                equity, core, sleeve, resid, core_sso, sleeve_upro,
                bool(dip[i - 1]), bool(trend[i - 1]), bool(i == 1 or mix[i - 1] or in_cash),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            in_cash = False
            if execution == "next_open":
                core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
                if sleeve_upro:
                    sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        new = core + sleeve + resid
        rets[i] = new / prev - 1.0
        invested = 0.0 if in_cash else core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def _allocate_flat_cash(
    equity: float,
    core: float,
    sleeve: float,
    resid: float,
    core_sso: bool,
    sleeve_upro: bool,
    already_cash: bool,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    """Sell the core and the sleeve. A book that is already cash does not trade."""
    if equity <= 0:
        raise ValueError("equity is non-positive")
    if already_cash and core == 0.0 and sleeve == 0.0:
        return 0.0, 0.0, equity, False, False, True, 0.0, 0.0
    cost = 0.0
    traded = 0.0
    if core > 0.0:
        cost += core * (sso_cost if core_sso else spy_cost)
        traded += core
    if sleeve > 0.0 and sleeve_upro:
        cost += sleeve * upro_cost
        traded += sleeve
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    return (
        0.0,
        0.0,
        equity - cost,
        False,
        False,
        True,
        traded / 2.0 / equity,
        cost / equity,
    )


def _allocate_core_hold(
    equity: float,
    core: float,
    core_cash: float,
    sleeve: float,
    resid: float,
    core_sso: bool,
    sleeve_upro: bool,
    held_frac: float,
    want_dip: bool,
    want_trend: bool,
    off_core_fraction: float,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    if equity <= 0:
        raise ValueError("equity is non-positive")
    desired = 1.0 if want_trend else float(off_core_fraction)
    desired = min(1.0, max(0.0, desired))
    frac_same = abs(desired - held_frac) <= 1e-12
    if not do_mix and want_dip == core_sso and want_trend == sleeve_upro and frac_same:
        return core, core_cash, sleeve, resid, core_sso, sleeve_upro, held_frac, 0.0, 0.0
    if do_mix:
        core_budget = core_weight * equity
        sleeve_budget = sleeve_weight * equity
    else:
        core_budget = core + core_cash
        sleeve_budget = sleeve
    tgt_core = core_budget * desired
    tgt_core_cash = core_budget * (1.0 - desired)
    tgt_sleeve = sleeve_budget
    tgt_resid = equity - core_budget - sleeve_budget
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta
    if want_trend != sleeve_upro:
        if sleeve_upro:
            cost += sleeve * upro_cost
            traded += sleeve
        if want_trend:
            cost += tgt_sleeve * upro_cost
            traded += tgt_sleeve
    elif want_trend:
        delta = abs(tgt_sleeve - sleeve)
        cost += delta * upro_cost
        traded += delta
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    shrink = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_core * shrink,
        tgt_core_cash * shrink,
        tgt_sleeve * shrink,
        tgt_resid * shrink,
        bool(want_dip),
        bool(want_trend),
        desired,
        turnover,
        cost / equity,
    )


def run_vol_sleeve(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    sleeve_scale: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book with a time-varying invested fraction of the UPRO sleeve.

    ``sleeve_scale[t]`` is the fraction of the sleeve budget held in UPRO on the
    next session while the trend is on. The rest of that budget earns the cash
    yield. A trend-off session holds none of the sleeve in UPRO. The core and
    the dip overlay are unchanged. A scale of 1 on every session is the
    unscaled trend-core book. The fraction is applied every session it changes;
    there is no extra deadband.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    scale = sleeve_scale.reindex(idx).astype(float).to_numpy(dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale < -1e-12) or np.any(scale > 1 + 1e-12):
        raise ValueError("sleeve_scale must be finite and in [0, 1] on every session")
    scale = np.clip(scale, 0.0, 1.0)
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    upro = 0.0
    sleeve_cash = 0.0
    resid = 1.0
    core_sso = False
    held_frac = 0.0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + upro + sleeve_cash + resid
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if upro != 0.0:
                upro *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            sleeve_cash *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, upro, sleeve_cash, resid, core_sso, held_frac, turn[i], drag[i] = _allocate_scaled(
                core + upro + sleeve_cash + resid, core, upro, sleeve_cash, resid,
                core_sso, held_frac, bool(dip[i - 1]), bool(trend[i - 1]),
                float(scale[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if upro != 0.0:
                upro *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if upro != 0.0:
                upro *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            sleeve_cash *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, upro, sleeve_cash, resid, core_sso, held_frac, turn[i], drag[i] = _allocate_scaled(
                core + upro + sleeve_cash + resid, core, upro, sleeve_cash, resid,
                core_sso, held_frac, bool(dip[i - 1]), bool(trend[i - 1]),
                float(scale[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
            )
        new = core + upro + sleeve_cash + resid
        rets[i] = new / prev - 1.0
        exposure[i] = (core + upro) / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def _allocate_scaled(
    equity: float,
    core: float,
    upro: float,
    sleeve_cash: float,
    resid: float,
    core_sso: bool,
    held_frac: float,
    want_dip: bool,
    want_trend: bool,
    scale: float,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    if equity <= 0:
        raise ValueError("equity is non-positive")
    desired = float(scale) if want_trend else 0.0
    desired = min(1.0, max(0.0, desired))
    frac_same = abs(desired - held_frac) <= 1e-12
    if not do_mix and want_dip == core_sso and frac_same:
        return core, upro, sleeve_cash, resid, core_sso, held_frac, 0.0, 0.0
    if do_mix:
        tgt_core = core_weight * equity
        budget = sleeve_weight * equity
    else:
        tgt_core = core
        budget = upro + sleeve_cash
    tgt_upro = budget * desired
    tgt_sleeve_cash = budget * (1.0 - desired)
    tgt_resid = equity - tgt_core - budget
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta
    upro_delta = abs(tgt_upro - upro)
    cost += upro_delta * upro_cost
    traded += upro_delta
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    shrink = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_core * shrink,
        tgt_upro * shrink,
        tgt_sleeve_cash * shrink,
        tgt_resid * shrink,
        bool(want_dip),
        desired,
        turnover,
        cost / equity,
    )


def run_vol_gear(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    trend_on: pd.Series,
    upro_share: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book whose on-trend sleeve is a blend of SPY and UPRO.

    ``upro_share[t]`` is the fraction of the sleeve budget held in UPRO on the
    next session while the trend is on. The rest of that budget is SPY, not
    cash. A trend-off session holds the sleeve in cash and ignores the share.
    The dip applies only to the core pile. The sleeve's SPY stake is not
    dipped and is not merged with the core, so a mix reset still restores the
    two piles. A share of 1 on every session is the unscaled trend-core book.
    The blend is applied every session the share changes. Cash pays no spread.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    share = upro_share.reindex(idx).astype(float).to_numpy(dtype=float)
    if np.any(~np.isfinite(share)) or np.any(share < -1e-12) or np.any(share > 1.0 + 1e-12):
        raise ValueError("upro_share must be finite and in [0, 1] on every session")
    share = np.clip(share, 0.0, 1.0)
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve_spy = 0.0
    sleeve_upro = 0.0
    sleeve_cash = 0.0
    resid = 1.0
    core_sso = False
    in_equity = False
    held_share = 0.0
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)
    for i in range(1, n):
        prev = core + sleeve_spy + sleeve_upro + sleeve_cash + resid
        want_equity = bool(trend[i - 1])
        want_share = float(share[i - 1]) if want_equity else 0.0
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_spy != 0.0:
                sleeve_spy *= prices["spy_o"][i] / prices["spy_c"][i - 1]
            if sleeve_upro != 0.0:
                sleeve_upro *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            sleeve_cash *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve_spy, sleeve_upro, sleeve_cash, resid, core_sso, in_equity, held_share, turn[i], drag[i] = (
                _allocate_vol_gear(
                    core + sleeve_spy + sleeve_upro + sleeve_cash + resid,
                    core, sleeve_spy, sleeve_upro, sleeve_cash, resid,
                    core_sso, in_equity, held_share,
                    bool(dip[i - 1]), want_equity, want_share, bool(i == 1 or mix[i - 1]),
                    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
                )
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if sleeve_spy != 0.0:
                sleeve_spy *= prices["spy_c"][i] / prices["spy_o"][i]
            if sleeve_upro != 0.0:
                sleeve_upro *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if sleeve_spy != 0.0:
                sleeve_spy *= prices["spy_c"][i] / prices["spy_c"][i - 1]
            if sleeve_upro != 0.0:
                sleeve_upro *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            sleeve_cash *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve_spy, sleeve_upro, sleeve_cash, resid, core_sso, in_equity, held_share, turn[i], drag[i] = (
                _allocate_vol_gear(
                    core + sleeve_spy + sleeve_upro + sleeve_cash + resid,
                    core, sleeve_spy, sleeve_upro, sleeve_cash, resid,
                    core_sso, in_equity, held_share,
                    bool(dip[i - 1]), want_equity, want_share, bool(i == 1 or mix[i - 1]),
                    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
                )
            )
        new = core + sleeve_spy + sleeve_upro + sleeve_cash + resid
        rets[i] = new / prev - 1.0
        exposure[i] = (core + sleeve_spy + sleeve_upro) / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def _allocate_vol_gear(
    equity: float,
    core: float,
    sleeve_spy: float,
    sleeve_upro: float,
    sleeve_cash: float,
    resid: float,
    core_sso: bool,
    in_equity: bool,
    held_share: float,
    want_dip: bool,
    want_equity: bool,
    want_share: float,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    """Split the sleeve between SPY and UPRO, or park it in cash when the trend is off."""
    if equity <= 0:
        raise ValueError("equity is non-positive")
    if want_equity:
        if want_share >= 1.0 - 1e-12:
            want_share = 1.0
        elif want_share <= 1e-12:
            want_share = 0.0
    else:
        want_share = 0.0
    share_same = (not want_equity) or abs(want_share - held_share) <= 1e-12
    if not do_mix and want_dip == core_sso and want_equity == in_equity and share_same:
        return (
            core, sleeve_spy, sleeve_upro, sleeve_cash, resid,
            core_sso, in_equity, held_share, 0.0, 0.0,
        )
    if do_mix:
        tgt_core = core_weight * equity
        budget = sleeve_weight * equity
    else:
        tgt_core = core
        budget = sleeve_spy + sleeve_upro + sleeve_cash
    if want_equity:
        tgt_upro = budget * want_share
        tgt_spy = budget * (1.0 - want_share)
        tgt_cash = 0.0
    else:
        tgt_upro = 0.0
        tgt_spy = 0.0
        tgt_cash = budget
    tgt_resid = equity - tgt_core - budget
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta
    spy_delta = abs(tgt_spy - sleeve_spy)
    upro_delta = abs(tgt_upro - sleeve_upro)
    cost += spy_delta * spy_cost + upro_delta * upro_cost
    traded += spy_delta + upro_delta
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    shrink = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_core * shrink,
        tgt_spy * shrink,
        tgt_upro * shrink,
        tgt_cash * shrink,
        tgt_resid * shrink,
        bool(want_dip),
        bool(want_equity),
        float(want_share),
        turnover,
        cost / equity,
    )


def run_idle_diversifier(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    div_open: pd.Series,
    div_close: pd.Series,
    trend_on: pd.Series,
    div_on: pd.Series,
    div_ready: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    div_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can leave equities for one diversifier.

    While ``trend_on`` is true the book is the trend-core book, dip included.
    While it is false and ``div_ready`` is false, it is still that book, with
    the sleeve in cash, so the years before the diversifier's average exists
    are not a separate timing rule. While the equity trend is false and the
    diversifier's filter is ready, the whole invested book holds the
    diversifier if ``div_on`` is true and T-bills otherwise. A decision at the
    close of t is filled at the next open. Diversifier prices may be missing
    before the fund lists; they are read only when the book holds that fund.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    div_o = div_open.reindex(idx).to_numpy(dtype=float)
    div_c = div_close.reindex(idx).to_numpy(dtype=float)
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    diversify = div_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    ready = div_ready.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    div_pos = 0.0
    resid = 1.0
    core_sso = False
    sleeve_upro = False
    book = "baseline"
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)

    def desired_at(j: int) -> str:
        if trend[j]:
            return "equity"
        if ready[j] and diversify[j]:
            return "div"
        if ready[j]:
            return "cash"
        return "baseline"

    def _require_div(j: int, *extra: int) -> None:
        for k in (j, *extra):
            if k < 0 or not np.isfinite(div_o[k]) or not np.isfinite(div_c[k]) or div_o[k] <= 0 or div_c[k] <= 0:
                raise ValueError(f"diversifier prices are missing on {idx[k].date()}")

    for i in range(1, n):
        prev = core + sleeve + div_pos + resid
        desired = desired_at(i - 1)
        if execution == "next_open":
            if book in ("equity", "baseline"):
                core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
                if sleeve_upro and sleeve != 0.0:
                    sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
                elif sleeve != 0.0:
                    sleeve *= 1.0 + rf[i]
            elif book == "div" and div_pos != 0.0:
                _require_div(i, i - 1)
                div_pos *= div_o[i] / div_c[i - 1]
            resid *= 1.0 + rf[i]
            core, sleeve, div_pos, resid, core_sso, sleeve_upro, book, turn[i], drag[i] = _step_idle(
                core + sleeve + div_pos + resid, core, sleeve, div_pos, resid,
                core_sso, sleeve_upro, book, desired, bool(dip[i - 1]),
                bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
            )
            if book == "div" and div_pos != 0.0:
                _require_div(i)
                div_pos *= div_c[i] / div_o[i]
            elif book in ("equity", "baseline"):
                core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
                if sleeve_upro and sleeve != 0.0:
                    sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
        else:
            if book in ("equity", "baseline"):
                core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
                if sleeve_upro and sleeve != 0.0:
                    sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
                elif sleeve != 0.0:
                    sleeve *= 1.0 + rf[i]
            elif book == "div" and div_pos != 0.0:
                _require_div(i, i - 1)
                div_pos *= div_c[i] / div_c[i - 1]
            resid *= 1.0 + rf[i]
            core, sleeve, div_pos, resid, core_sso, sleeve_upro, book, turn[i], drag[i] = _step_idle(
                core + sleeve + div_pos + resid, core, sleeve, div_pos, resid,
                core_sso, sleeve_upro, book, desired, bool(dip[i - 1]),
                bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
            )
        new = core + sleeve + div_pos + resid
        rets[i] = new / prev - 1.0
        if book == "div":
            invested = div_pos
        elif book == "cash":
            invested = 0.0
        else:
            invested = core + (sleeve if sleeve_upro else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def run_sleeve_diversifier(
    spy_open: pd.Series,
    spy_close: pd.Series,
    sso_open: pd.Series,
    sso_close: pd.Series,
    upro_open: pd.Series,
    upro_close: pd.Series,
    div_open: pd.Series,
    div_close: pd.Series,
    trend_on: pd.Series,
    div_on: pd.Series,
    div_ready: pd.Series,
    dip_on: pd.Series,
    mix_rebalance: pd.Series,
    core_weight: float,
    sleeve_weight: float,
    cash_yield: pd.Series,
    spy_cost: float = 0.0,
    sso_cost: float = 0.0,
    upro_cost: float = 0.0,
    div_cost: float = 0.0,
    execution: str = "next_open",
) -> BacktestResult:
    """Trend-core book that can park only the off-trend sleeve in one diversifier.

    While ``trend_on`` is true the book is the trend-core book, dip included,
    and the diversifier is not held. While the trend is false and the
    diversifier filter is not ready, or is ready and off, the sleeve stays in
    T-bills, so the book is again that trend core. While the trend is false,
    the filter is ready, and ``div_on`` is true, only the sleeve budget holds
    the diversifier. The core stays in SPY, or in SSO when the dip is on.
    A decision at the close of t is filled at the next open. Diversifier prices
    may be missing before the fund lists; they are read only when the sleeve
    holds that fund.
    """
    if execution not in ("next_open", "next_close"):
        raise ValueError(f"unknown execution {execution!r}")
    if core_weight < 0 or sleeve_weight < 0 or core_weight + sleeve_weight > 1 + 1e-8:
        raise ValueError("core and sleeve weights must be non-negative and sum to at most 1")
    idx = spy_close.index
    prices = {
        "spy_o": spy_open.reindex(idx).to_numpy(dtype=float),
        "spy_c": spy_close.reindex(idx).to_numpy(dtype=float),
        "sso_o": sso_open.reindex(idx).to_numpy(dtype=float),
        "sso_c": sso_close.reindex(idx).to_numpy(dtype=float),
        "upro_o": upro_open.reindex(idx).to_numpy(dtype=float),
        "upro_c": upro_close.reindex(idx).to_numpy(dtype=float),
    }
    stacked = np.column_stack(list(prices.values()))
    if np.any(~np.isfinite(stacked)) or np.any(stacked <= 0):
        raise ValueError("SPY, SSO, and UPRO prices must be finite and positive")
    div_o = div_open.reindex(idx).to_numpy(dtype=float)
    div_c = div_close.reindex(idx).to_numpy(dtype=float)
    trend = trend_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    diversify = div_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    ready = div_ready.reindex(idx).fillna(False).to_numpy(dtype=bool)
    dip = dip_on.reindex(idx).fillna(False).to_numpy(dtype=bool)
    mix = mix_rebalance.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rf = cash_yield.reindex(idx).astype(float).fillna(0.0).to_numpy(dtype=float)
    n = len(idx)
    core = 0.0
    sleeve = 0.0
    div_pos = 0.0
    resid = 1.0
    core_sso = False
    mode = "cash"
    rets = np.full(n, np.nan)
    turn = np.zeros(n)
    drag = np.zeros(n)
    exposure = np.full(n, np.nan)

    def desired_mode(j: int) -> str:
        if trend[j]:
            return "upro"
        if ready[j] and diversify[j]:
            return "div"
        return "cash"

    def _require_div(j: int, *extra: int) -> None:
        for k in (j, *extra):
            if k < 0 or not np.isfinite(div_o[k]) or not np.isfinite(div_c[k]) or div_o[k] <= 0 or div_c[k] <= 0:
                raise ValueError(f"diversifier prices are missing on {idx[k].date()}")

    for i in range(1, n):
        prev = core + sleeve + div_pos + resid
        want = desired_mode(i - 1)
        if execution == "next_open":
            core = _mark(core, prices["sso_o"][i], prices["sso_c"][i - 1], prices["spy_o"][i], prices["spy_c"][i - 1], core_sso)
            if mode == "upro" and sleeve != 0.0:
                sleeve *= prices["upro_o"][i] / prices["upro_c"][i - 1]
            elif mode == "div" and div_pos != 0.0:
                _require_div(i, i - 1)
                div_pos *= div_o[i] / div_c[i - 1]
            elif mode == "cash" and sleeve != 0.0:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, div_pos, resid, core_sso, mode, turn[i], drag[i] = _step_sleeve_cash(
                core + sleeve + div_pos + resid, core, sleeve, div_pos, resid,
                core_sso, mode, want, bool(dip[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
            )
            core = _mark(core, prices["sso_c"][i], prices["sso_o"][i], prices["spy_c"][i], prices["spy_o"][i], core_sso)
            if mode == "upro" and sleeve != 0.0:
                sleeve *= prices["upro_c"][i] / prices["upro_o"][i]
            elif mode == "div" and div_pos != 0.0:
                _require_div(i)
                div_pos *= div_c[i] / div_o[i]
        else:
            core = _mark(core, prices["sso_c"][i], prices["sso_c"][i - 1], prices["spy_c"][i], prices["spy_c"][i - 1], core_sso)
            if mode == "upro" and sleeve != 0.0:
                sleeve *= prices["upro_c"][i] / prices["upro_c"][i - 1]
            elif mode == "div" and div_pos != 0.0:
                _require_div(i, i - 1)
                div_pos *= div_c[i] / div_c[i - 1]
            elif mode == "cash" and sleeve != 0.0:
                sleeve *= 1.0 + rf[i]
            resid *= 1.0 + rf[i]
            core, sleeve, div_pos, resid, core_sso, mode, turn[i], drag[i] = _step_sleeve_cash(
                core + sleeve + div_pos + resid, core, sleeve, div_pos, resid,
                core_sso, mode, want, bool(dip[i - 1]), bool(i == 1 or mix[i - 1]),
                core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
            )
        new = core + sleeve + div_pos + resid
        rets[i] = new / prev - 1.0
        invested = core + (sleeve if mode == "upro" else 0.0) + (div_pos if mode == "div" else 0.0)
        exposure[i] = invested / new if new else 0.0
    return BacktestResult(
        returns=pd.Series(rets, index=idx, name="return"),
        turnover=pd.Series(turn, index=idx, name="turnover"),
        cost_drag=pd.Series(drag, index=idx, name="cost_drag"),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
    )


def _step_sleeve_cash(
    equity, core, sleeve, div_pos, resid,
    core_sso, mode, want, want_dip, do_mix,
    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
):
    """Delegate cash/UPRO sleeves to the trend-core allocator so that path matches."""
    if mode != "div" and want != "div":
        core, sleeve, resid, core_sso, sleeve_upro, traded, cost = _allocate_trend(
            equity, core, sleeve, resid, core_sso, mode == "upro",
            bool(want_dip), want == "upro", do_mix,
            core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
        )
        return core, sleeve, 0.0, resid, core_sso, ("upro" if sleeve_upro else "cash"), traded, cost
    return _allocate_sleeve_cash(
        equity, core, sleeve, div_pos, resid, core_sso, mode,
        bool(want_dip), want, do_mix,
        core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
    )


def _allocate_sleeve_cash(
    equity, core, sleeve, div_pos, resid, core_sso, mode, want_dip, want_mode, do_mix,
    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
):
    """Same pile rules as the trend core, with the sleeve in cash, the diversifier, or UPRO.

    Cash pays no spread. The core pile is never moved into the diversifier.
    Switching instruments pays the leaving leg and the arriving leg.
    """
    if equity <= 0:
        raise ValueError("equity is non-positive")
    if want_mode not in ("cash", "upro", "div") or mode not in ("cash", "upro", "div"):
        raise ValueError(f"unknown sleeve mode {mode!r} -> {want_mode!r}")
    if mode == "div" and sleeve != 0.0:
        raise ValueError("diversifier sleeve and equity sleeve are both open")
    if mode != "div" and div_pos != 0.0:
        raise ValueError("diversifier position is open while the sleeve is not in it")
    sleeve_dollars = div_pos if mode == "div" else sleeve
    if not do_mix and want_dip == core_sso and want_mode == mode:
        return core, sleeve, div_pos, resid, core_sso, mode, 0.0, 0.0
    if do_mix:
        tgt_core = core_weight * equity
        tgt_sleeve = sleeve_weight * equity
    else:
        tgt_core = core
        tgt_sleeve = sleeve_dollars
    tgt_resid = equity - tgt_core - tgt_sleeve
    if tgt_resid < -1e-6:
        raise ValueError("sleeve targets exceed equity")
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta

    def _leg(which: str, dollars: float) -> tuple[float, float]:
        if which == "upro":
            return dollars * upro_cost, dollars
        if which == "div":
            return dollars * div_cost, dollars
        return 0.0, 0.0

    if want_mode != mode:
        left_cost, left_traded = _leg(mode, sleeve_dollars)
        right_cost, right_traded = _leg(want_mode, tgt_sleeve)
        cost += left_cost + right_cost
        traded += left_traded + right_traded
    else:
        moved_cost, moved = _leg(want_mode, abs(tgt_sleeve - sleeve_dollars))
        cost += moved_cost
        traded += moved
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    tgt_core *= scale
    tgt_sleeve *= scale
    tgt_resid = max(0.0, tgt_resid) * scale
    if want_mode == "div":
        return tgt_core, 0.0, tgt_sleeve, tgt_resid, bool(want_dip), "div", turnover, cost / equity
    if want_mode == "upro":
        return tgt_core, tgt_sleeve, 0.0, tgt_resid, bool(want_dip), "upro", turnover, cost / equity
    return tgt_core, tgt_sleeve, 0.0, tgt_resid, bool(want_dip), "cash", turnover, cost / equity


def _step_idle(
    equity, core, sleeve, div_pos, resid,
    core_sso, sleeve_upro, book, desired, want_dip, do_mix,
    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
):
    """One rebalance. Core-book states delegate to the trend-core allocator."""
    if desired in ("equity", "baseline") and book in ("equity", "baseline"):
        core, sleeve, resid, core_sso, sleeve_upro, traded, cost = _allocate_trend(
            equity, core, sleeve, resid, core_sso, sleeve_upro,
            bool(want_dip), desired == "equity", do_mix,
            core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost,
        )
        return core, sleeve, 0.0, resid, core_sso, sleeve_upro, desired, traded, cost
    if desired == book and book in ("div", "cash"):
        return core, sleeve, div_pos, resid, core_sso, sleeve_upro, book, 0.0, 0.0
    core, sleeve, div_pos, resid, core_sso, sleeve_upro, traded, cost = _switch_idle(
        equity, core, sleeve, div_pos, resid, core_sso, sleeve_upro, desired, bool(want_dip),
        core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
    )
    return core, sleeve, div_pos, resid, core_sso, sleeve_upro, desired, traded, cost


def _switch_idle(
    equity, core, sleeve, div_pos, resid, core_sso, sleeve_upro, desired, want_dip,
    core_weight, sleeve_weight, spy_cost, sso_cost, upro_cost, div_cost,
):
    """Sell whatever is held and buy the target book. Cash pays no spread."""
    if equity <= 0:
        raise ValueError("equity is non-positive")
    cost = 0.0
    traded = 0.0
    if core > 0.0:
        cost += core * (sso_cost if core_sso else spy_cost)
        traded += core
    if sleeve_upro and sleeve > 0.0:
        cost += sleeve * upro_cost
        traded += sleeve
    if div_pos > 0.0:
        cost += div_pos * div_cost
        traded += div_pos
    hold_div = desired == "div"
    hold_equity = desired in ("equity", "baseline")
    new_sso = bool(want_dip) if hold_equity else False
    want_upro = desired == "equity"
    if hold_equity:
        tgt_core = core_weight * equity
        tgt_sleeve = sleeve_weight * equity
        tgt_div = 0.0
    elif hold_div:
        tgt_core = 0.0
        tgt_sleeve = 0.0
        tgt_div = (core_weight + sleeve_weight) * equity
    else:
        tgt_core = 0.0
        tgt_sleeve = 0.0
        tgt_div = 0.0
    if tgt_core > 0.0:
        cost += tgt_core * (sso_cost if new_sso else spy_cost)
        traded += tgt_core
    if want_upro and tgt_sleeve > 0.0:
        cost += tgt_sleeve * upro_cost
        traded += tgt_sleeve
    if tgt_div > 0.0:
        cost += tgt_div * div_cost
        traded += tgt_div
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    tgt_resid = equity - tgt_core - tgt_sleeve - tgt_div
    if tgt_resid < -1e-6:
        raise ValueError("idle targets exceed equity")
    return (
        tgt_core * scale,
        tgt_sleeve * scale,
        tgt_div * scale,
        max(0.0, tgt_resid) * scale,
        new_sso,
        want_upro,
        turnover,
        cost / equity,
    )


def _allocate_dip(
    equity: float,
    core_spy: float,
    core_sso: float,
    sleeve: float,
    resid: float,
    held_boost: float,
    sleeve_upro: bool,
    want_boost: float,
    want_trend: bool,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    """Split the core pile between SPY and SSO. A 0/1 boost matches ``_allocate_trend``."""
    if equity <= 0:
        raise ValueError("equity is non-positive")
    boost_same = abs(float(want_boost) - float(held_boost)) <= 1e-12
    if not do_mix and boost_same and want_trend == sleeve_upro:
        return core_spy, core_sso, sleeve, resid, float(held_boost), sleeve_upro, 0.0, 0.0
    core = core_spy + core_sso
    if do_mix:
        tgt_core = core_weight * equity
        tgt_sleeve = sleeve_weight * equity
    else:
        tgt_core = core
        tgt_sleeve = sleeve
    tgt_resid = equity - tgt_core - tgt_sleeve
    tgt_sso = float(want_boost) * tgt_core
    tgt_spy = tgt_core - tgt_sso
    cost = abs(tgt_spy - core_spy) * spy_cost + abs(tgt_sso - core_sso) * sso_cost
    traded = abs(tgt_spy - core_spy) + abs(tgt_sso - core_sso)
    if want_trend != sleeve_upro:
        if sleeve_upro:
            cost += sleeve * upro_cost
            traded += sleeve
        if want_trend:
            cost += tgt_sleeve * upro_cost
            traded += tgt_sleeve
    elif want_trend:
        delta = abs(tgt_sleeve - sleeve)
        cost += delta * upro_cost
        traded += delta
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_spy * scale,
        tgt_sso * scale,
        tgt_sleeve * scale,
        tgt_resid * scale,
        float(want_boost),
        bool(want_trend),
        turnover,
        cost / equity,
    )


def _mark_gear(
    value: float,
    num_spy: float,
    den_spy: float,
    num_upro: float,
    den_upro: float,
    gear: int,
    cash_factor: float,
) -> float:
    """Mark a sleeve pile. Gear 2 is UPRO, gear 1 is SPY, gear 0 is cash."""
    if value == 0.0:
        return 0.0
    if gear == 2:
        return value * num_upro / den_upro
    if gear == 1:
        return value * num_spy / den_spy
    if gear == 0:
        return value * cash_factor
    raise ValueError(f"unknown sleeve gear {gear}")


def _allocate_sleeve_gear(
    equity: float,
    core: float,
    sleeve: float,
    resid: float,
    core_sso: bool,
    sleeve_gear: int,
    want_dip: bool,
    want_gear: int,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    """Same pile rules as the trend core, with the sleeve in cash, SPY, or UPRO.

    Cash pays no spread. Switching instruments pays the leaving leg and the
    arriving leg. A resize inside one instrument pays that instrument's cost
    on the dollars that move.
    """
    if equity <= 0:
        raise ValueError("equity is non-positive")
    want_gear = int(want_gear)
    sleeve_gear = int(sleeve_gear)
    if want_gear not in (0, 1, 2) or sleeve_gear not in (0, 1, 2):
        raise ValueError("sleeve gear must be 0, 1, or 2")
    if not do_mix and want_dip == core_sso and want_gear == sleeve_gear:
        return core, sleeve, resid, core_sso, sleeve_gear, 0.0, 0.0
    if do_mix:
        tgt_core = core_weight * equity
        tgt_sleeve = sleeve_weight * equity
    else:
        tgt_core = core
        tgt_sleeve = sleeve
    tgt_resid = equity - tgt_core - tgt_sleeve
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta

    def _leg(gear: int, dollars: float) -> tuple[float, float]:
        if gear == 2:
            return dollars * upro_cost, dollars
        if gear == 1:
            return dollars * spy_cost, dollars
        return 0.0, 0.0

    if want_gear != sleeve_gear:
        left_cost, left_traded = _leg(sleeve_gear, sleeve)
        right_cost, right_traded = _leg(want_gear, tgt_sleeve)
        cost += left_cost + right_cost
        traded += left_traded + right_traded
    else:
        moved_cost, moved = _leg(want_gear, abs(tgt_sleeve - sleeve))
        cost += moved_cost
        traded += moved
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_core * scale,
        tgt_sleeve * scale,
        tgt_resid * scale,
        bool(want_dip),
        want_gear,
        turnover,
        cost / equity,
    )


def _mark(value: float, num_sso: float, den_sso: float, num_spy: float, den_spy: float, in_sso: bool) -> float:
    if value == 0.0:
        return 0.0
    if in_sso:
        return value * num_sso / den_sso
    return value * num_spy / den_spy


def _allocate_trend(
    equity: float,
    core: float,
    sleeve: float,
    resid: float,
    core_sso: bool,
    sleeve_upro: bool,
    want_dip: bool,
    want_trend: bool,
    do_mix: bool,
    core_weight: float,
    sleeve_weight: float,
    spy_cost: float,
    sso_cost: float,
    upro_cost: float,
):
    if equity <= 0:
        raise ValueError("equity is non-positive")
    if not do_mix and want_dip == core_sso and want_trend == sleeve_upro:
        return core, sleeve, resid, core_sso, sleeve_upro, 0.0, 0.0
    if do_mix:
        tgt_core = core_weight * equity
        tgt_sleeve = sleeve_weight * equity
    else:
        tgt_core = core
        tgt_sleeve = sleeve
    tgt_resid = equity - tgt_core - tgt_sleeve
    cost = 0.0
    traded = 0.0
    if want_dip != core_sso:
        old = sso_cost if core_sso else spy_cost
        new = sso_cost if want_dip else spy_cost
        cost += core * old + tgt_core * new
        traded += core + tgt_core
    else:
        inst = sso_cost if want_dip else spy_cost
        delta = abs(tgt_core - core)
        cost += delta * inst
        traded += delta
    if want_trend != sleeve_upro:
        if sleeve_upro:
            cost += sleeve * upro_cost
            traded += sleeve
        if want_trend:
            cost += tgt_sleeve * upro_cost
            traded += tgt_sleeve
    elif want_trend:
        delta = abs(tgt_sleeve - sleeve)
        cost += delta * upro_cost
        traded += delta
    if cost >= equity:
        raise ValueError("transaction costs consume the whole book")
    scale = (equity - cost) / equity
    turnover = traded / 2.0 / equity
    return (
        tgt_core * scale,
        tgt_sleeve * scale,
        tgt_resid * scale,
        bool(want_dip),
        bool(want_trend),
        turnover,
        cost / equity,
    )
