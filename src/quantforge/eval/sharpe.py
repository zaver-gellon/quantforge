"""Probabilistic and deflated Sharpe ratios.

Bailey and López de Prado, "The Deflated Sharpe Ratio" (2014). The per-period
Sharpe used inside the test is the population ratio ``mean / std(ddof=0)`` so
it matches the skew and kurtosis moments. ``N`` is the number of trials in the
ledger, not a guess made after looking at the winner.

The multiple-testing bar for an information-ratio t-stat is the expected
maximum of ``N`` independent standard normals (Euler-Mascheroni approximation).
Correlated trials make that bar conservative.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statistics import NormalDist

_EULER = 0.5772156649015329
_NORM = NormalDist()


def _excess(returns, rf) -> np.ndarray:
    series = pd.Series(returns).dropna().astype(float)
    if rf is not None:
        series = series - pd.Series(rf).reindex(series.index).astype(float).fillna(0.0)
    return series.to_numpy(dtype=float)


def _moments(x: np.ndarray) -> tuple[float, float, float, float]:
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd == 0.0 or len(x) < 3:
        return mu, sd, 0.0, 3.0
    z = (x - mu) / sd
    return mu, sd, float(np.mean(z**3)), float(np.mean(z**4))


def probabilistic_sharpe(returns, rf=None, sr_benchmark: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds ``sr_benchmark``."""
    x = _excess(returns, rf)
    n = len(x)
    if n < 3:
        return float("nan")
    mu, sd, skew, kurt = _moments(x)
    if sd == 0.0:
        return 1.0 if mu > sr_benchmark else 0.0
    sr = mu / sd
    variance = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(max(variance, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(_NORM.cdf(z))


def expected_max_z(n_trials: int) -> float:
    """Expected maximum of ``n_trials`` i.i.d. standard normals."""
    if n_trials < 2:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - _EULER) * z1 + _EULER * z2


def expected_maximum_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Sharpe ratio that the best of ``n_trials`` noise strategies is expected to post."""
    if n_trials < 2 or sharpe_std <= 0.0 or not math.isfinite(sharpe_std):
        return 0.0
    return float(sharpe_std * expected_max_z(n_trials))


def deflated_sharpe(returns, n_trials: int, sharpe_std: float, rf=None) -> float:
    """PSR of this path versus the multiple-testing benchmark Sharpe ``SR0``."""
    sr0 = expected_maximum_sharpe(n_trials, sharpe_std)
    return probabilistic_sharpe(returns, rf=rf, sr_benchmark=sr0)
