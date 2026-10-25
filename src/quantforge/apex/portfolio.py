"""Monthly target weights from a score panel.

The score at a month-end is turned into a unit book, then scaled by the
volatility target, the regime multiplier, and half-Kelly. Name, group, and
gross limits are applied after every scale. A turnover blend pulls the result
toward the previous target. Weights are forward-filled so a daily signal file
can read the latest target; the backtest still trades only on month-ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.apex.construct import hrp_score_weights, mean_variance_weights, project_weights, risk_budget_weights
from quantforge.apex.riskmodel import kelly_scale, ledoit_wolf, portfolio_vol
from quantforge.engine.schedule import month_ends


def group_index(columns, sector_map: dict[str, str]) -> list[np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for i, name in enumerate(columns):
        buckets.setdefault(sector_map.get(str(name), str(name)), []).append(i)
    return [np.asarray(idx, dtype=int) for idx in buckets.values()]


def _cov_window(returns: pd.DataFrame, as_of, window: int) -> tuple[np.ndarray, np.ndarray]:
    hist = returns.loc[:as_of].iloc[-window:]
    clean = hist.dropna()
    n = hist.shape[1]
    if len(clean) < max(30, n + 2):
        vol = hist.std(ddof=1).to_numpy(dtype=float)
        vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, 0.01)
        return np.diag(vol ** 2), vol
    cov, _shrink = ledoit_wolf(clean.to_numpy(dtype=float))
    vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return cov, vol


def monthly_targets(
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    method: str = "budget",
    long_only: bool = True,
    gross: float = 1.0,
    max_weight: float = 0.20,
    sector_map: dict[str, str] | None = None,
    sector_cap: float = 0.45,
    regime: pd.Series | None = None,
    vol_target: float | None = 0.10,
    kelly_fraction: float | None = 0.5,
    turnover_blend: float = 0.5,
    cov_window: int = 126,
    kelly_window: int = 252,
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """Target weights decided at each month-end close.

    ``turnover_blend`` is the weight on the new book. 1 replaces the old book.
    ``vol_target`` None leaves the unit book at ``gross`` before regime and Kelly.
    """
    if method not in ("budget", "hrp", "mv"):
        raise ValueError("method must be budget, hrp, or mv")
    assets = list(closes.columns)
    scores = scores.reindex(columns=assets)
    returns = closes.pct_change()
    groups = group_index(assets, sector_map or {})
    mask = month_ends(closes.index)
    decisions = list(closes.index[mask.to_numpy()])
    out = pd.DataFrame(0.0, index=closes.index, columns=assets)
    prev = np.zeros(len(assets))
    rate = None if rf is None else rf.reindex(closes.index).astype(float).fillna(0.0)
    for decision in decisions:
        row = scores.loc[decision] if decision in scores.index else None
        if row is None or row.isna().all():
            # Carry the previous target. Cash until the first real score.
            out.loc[decision] = prev
            continue
        vector = row.to_numpy(dtype=float)
        cov, vol = _cov_window(returns, decision, cov_window)
        if method == "budget":
            unit = risk_budget_weights(vector, vol, 1.0, long_only, max_weight, groups, sector_cap)
        elif method == "hrp":
            unit = hrp_score_weights(vector, cov, 1.0, long_only, max_weight, groups, sector_cap)
        else:
            unit = mean_variance_weights(vector, cov, 1.0, long_only, max_weight, groups, sector_cap)
        scale = float(gross)
        if vol_target is not None:
            forecast = portfolio_vol(unit, cov)
            if np.isfinite(forecast) and forecast > 0:
                scale = min(float(gross), float(vol_target) / forecast)
        if regime is not None and decision in regime.index and np.isfinite(regime.loc[decision]):
            scale *= float(regime.loc[decision])
        weights = unit * scale
        if kelly_fraction is not None and kelly_fraction > 0:
            hist = returns.loc[:decision].iloc[-kelly_window:]
            if len(hist.dropna()) >= 60:
                port = hist.fillna(0.0).to_numpy(dtype=float) @ weights
                if rate is not None:
                    port = port - rate.loc[hist.index].to_numpy(dtype=float)
                weights = weights * kelly_scale(port, 1.0, kelly_fraction)
        weights = project_weights(weights, gross, long_only, max_weight, groups, sector_cap)
        if turnover_blend < 1.0 and np.abs(prev).sum() > 0:
            weights = turnover_blend * weights + (1.0 - turnover_blend) * prev
            weights = project_weights(weights, gross, long_only, max_weight, groups, sector_cap)
        prev = weights
        out.loc[decision] = weights
    # Forward-fill the target so the latest row is the book a bot would hold.
    # Trading still happens only where the caller marks a rebalance.
    decided = out.copy()
    decided.loc[~mask.reindex(out.index).fillna(False)] = np.nan
    # The first rows are genuine zeros (cash), not missing decisions.
    first = mask.to_numpy().argmax() if mask.any() else 0
    filled = decided.ffill()
    if first > 0:
        filled.iloc[:first] = 0.0
    filled = filled.fillna(0.0)
    return filled
