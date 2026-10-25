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
