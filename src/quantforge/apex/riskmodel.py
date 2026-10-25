"""Covariance, volatility targeting, drawdown throttle, and a Kelly cap.

Ledoit-Wolf here is the analytical shrinkage of the sample covariance toward
a scaled identity (Ledoit and Wolf, 2004). The target is ``(tr S / N) I``.
The intensity is clipped to [0, 1]. Eigenvalues are floored so a later
inverse is defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(covariance, shrinkage)`` for a ``T x N`` return matrix.

    Rows are dates. The sample covariance divides by ``T``, matching the
    estimator the shrinkage intensity was derived for.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a two-dimensional array")
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    t_obs, n_assets = x.shape
    if t_obs < 2 or n_assets < 1:
        raise ValueError("Ledoit-Wolf needs at least two rows")
    x = x - x.mean(axis=0)
    sample = (x.T @ x) / t_obs
    mu = float(np.trace(sample) / n_assets)
    target = mu * np.eye(n_assets)
    squared = x ** 2
    phi = float(np.sum((squared.T @ squared) / t_obs - sample ** 2))
    gamma = float(np.sum((sample - target) ** 2))
    if gamma <= 0:
        shrink = 1.0
    else:
        shrink = float(np.clip((phi / gamma) / t_obs, 0.0, 1.0))
    cov = (1.0 - shrink) * sample + shrink * target
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-10, None)
    cov = vectors @ np.diag(values) @ vectors.T
    return cov, shrink


def portfolio_vol(weights: np.ndarray, cov: np.ndarray, periods_per_year: int = 252) -> float:
    w = np.asarray(weights, dtype=float)
    variance = float(w @ cov @ w)
    if variance <= 0 or not np.isfinite(variance):
        return float("nan")
    return float(np.sqrt(variance * periods_per_year))


def vol_target_scale(weights: np.ndarray, cov: np.ndarray, target: float, cap: float) -> float:
    """Multiplier that moves forecast vol to ``target``, never above ``cap``."""
    if target <= 0 or cap <= 0:
        raise ValueError("target and cap must be positive")
    forecast = portfolio_vol(weights, cov)
    if not np.isfinite(forecast) or forecast <= 0:
        return 1.0
    return float(min(cap, target / forecast))


def kelly_scale(portfolio_excess: np.ndarray, gross: float, fraction: float = 0.5) -> float:
    """Fraction of the proposed gross to keep, capped at 1.

    ``0.5 * mean / variance`` is half-Kelly in units of the portfolio that
    ``portfolio_excess`` was computed on. Dividing by ``gross`` converts that
    into a scale on the weight vector. A non-positive edge scales to cash.
    """
    excess = np.asarray(portfolio_excess, dtype=float)
    excess = excess[np.isfinite(excess)]
    if len(excess) < 20 or gross <= 0:
        return 1.0
    mean = float(excess.mean())
    var = float(excess.var())
    if var <= 0 or mean <= 0:
        return 0.0
    full = fraction * mean / var
    return float(np.clip(full / gross, 0.0, 1.0))


def drawdown_exposure(
    returns: pd.Series,
    cuts: tuple[tuple[float, float], ...] = ((0.10, 0.70), (0.20, 0.40)),
) -> pd.Series:
    """Gross multiplier known at the close, from the path through that close.

    ``cuts`` are ``(drawdown depth, multiplier)``. The deepest breached cut
    wins. The series is the exposure to apply on the next session.
    """
    series = pd.Series(returns).astype(float)
    equity = 1.0
    peak = 1.0
    out = np.ones(len(series), dtype=float)
    ordered = tuple(sorted(cuts))
    for i, value in enumerate(series.to_numpy(dtype=float)):
        if not np.isfinite(value):
            out[i] = 1.0
            continue
        equity *= 1.0 + value
        if equity <= 0:
            out[i] = ordered[-1][1] if ordered else 0.0
            continue
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        scale = 1.0
        for depth, multiplier in ordered:
            if drawdown <= -depth:
                scale = multiplier
        out[i] = scale
    return pd.Series(out, index=series.index, name="dd_exposure")
