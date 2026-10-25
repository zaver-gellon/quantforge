"""Path statistics on a daily simple-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(
    returns: pd.Series,
    rf: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    exposure: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict:
    """CAGR, volatility, excess Sharpe, drawdown, and worst calendar year.

    CAGR and volatility use ``periods_per_year`` trading days per year.
    The Sharpe ratio is mean excess return over sample standard deviation,
    annualized. ``sharpe_per_period`` is the population (ddof=0) daily excess
    Sharpe, which is the unit the deflated-Sharpe ledger uses.
    """
    r = pd.Series(returns).dropna().astype(float).sort_index()
    if r.empty:
        return {"n_days": 0}
    n = int(len(r))
    years = n / periods_per_year
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    if rf is not None:
        excess = r - pd.Series(rf).reindex(r.index).astype(float).fillna(0.0)
    else:
        excess = r
    sample_sd = float(excess.std(ddof=1))
    pop_sd = float(excess.std(ddof=0))
    sharpe = float(excess.mean() / sample_sd * np.sqrt(periods_per_year)) if sample_sd > 0 else 0.0
    sharpe_pp = float(excess.mean() / pop_sd) if pop_sd > 0 else 0.0
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    annual = (1.0 + r).groupby(r.index.year).prod() - 1.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    underwater = (drawdown < -1e-12).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    out = {
        "n_days": n,
        "years": years,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sharpe_per_period": sharpe_pp,
        "max_drawdown": float(drawdown.min()),
        "maxdd_peak": str(pd.Timestamp(peak).date()),
        "maxdd_trough": str(pd.Timestamp(trough).date()),
        "longest_underwater_days": int(longest),
        "worst_year": int(annual.idxmin()) if len(annual) else None,
        "worst_year_return": float(annual.min()) if len(annual) else None,
        "total_return": float(equity.iloc[-1] - 1.0),
        "final_multiple": float(equity.iloc[-1]),
        "start": str(pd.Timestamp(r.index[0]).date()),
        "end": str(pd.Timestamp(r.index[-1]).date()),
    }
    if turnover is not None:
        traded = pd.Series(turnover).reindex(r.index).fillna(0.0)
        out["turnover_per_year"] = float(traded.sum() / years) if years else float("nan")
    if exposure is not None:
        out["mean_exposure"] = float(pd.Series(exposure).reindex(r.index).mean())
    if benchmark is not None:
        bench = pd.Series(benchmark).reindex(r.index).astype(float)
        both = r.index[bench.notna()]
        active = r.loc[both] - bench.loc[both]
        active_sd = float(active.std(ddof=1))
        ir = float(active.mean() / active_sd * np.sqrt(periods_per_year)) if active_sd > 0 else 0.0
        out["ir_vs_benchmark"] = ir
        out["ir_t"] = float(ir * np.sqrt(len(both) / periods_per_year))
        yearly_s = (1.0 + r.loc[both]).groupby(r.loc[both].index.year).prod()
        yearly_b = (1.0 + bench.loc[both]).groupby(bench.loc[both].index.year).prod()
        out["years_beating_benchmark"] = float((yearly_s > yearly_b).mean()) if len(yearly_s) else float("nan")
    return out


def path_cagr(returns, periods_per_year: int = 252) -> float:
    """Compounded growth of a simple-return path. Non-positive wealth is undefined."""
    series = pd.Series(returns).dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    equity = float((1.0 + series).prod())
    if equity <= 0:
        return float("nan")
    return equity ** (periods_per_year / len(series)) - 1.0


def path_max_drawdown(returns) -> float:
    series = pd.Series(returns).dropna().astype(float)
    if series.empty:
        return float("nan")
    equity = (1.0 + series).cumprod()
    if float(equity.min()) <= 0:
        return -1.0
    return float((equity / equity.cummax() - 1.0).min())


def slice_period(returns: pd.Series, start: str, end: str) -> pd.Series:
    series = pd.Series(returns).dropna().sort_index()
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]
