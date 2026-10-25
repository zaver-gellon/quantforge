"""Turn a score and a covariance into weights that respect the limits.

Risk budgeting is inverse-vol times the score. Hierarchical risk parity
follows Lopez de Prado's recursive bisection on a single-linkage order.
Mean-variance is a closed form that is then projected onto the same limits.
The projection is iterative clipping, not a quadratic program. It enforces
the gross, name, and group caps; it does not claim a unique optimum.
"""

from __future__ import annotations

import numpy as np


def project_weights(
    weights: np.ndarray,
    gross: float,
    long_only: bool,
    max_weight: float,
    groups: list[np.ndarray] | None = None,
    sector_cap: float = 1.0,
) -> np.ndarray:
    """Clip a weight vector onto the long/short, name, group, and gross limits."""
    w = np.asarray(weights, dtype=float).copy()
    w[~np.isfinite(w)] = 0.0
    lo = 0.0 if long_only else -abs(max_weight)
    hi = abs(max_weight)
    cap = abs(sector_cap)
    for _ in range(12):
        w = np.clip(w, lo, hi)
        if groups:
            for idx in groups:
                exposure = float(np.abs(w[idx]).sum())
                if cap > 0 and exposure > cap:
                    w[idx] *= cap / exposure
        gross_now = float(np.abs(w).sum())
        if gross > 0 and gross_now > gross:
            w *= gross / gross_now
    w = np.clip(w, lo, hi)
    gross_now = float(np.abs(w).sum())
    if gross > 0 and gross_now > gross + 1e-12:
        w *= gross / gross_now
    return w


def risk_budget_weights(
    scores: np.ndarray,
    vol: np.ndarray,
    gross: float,
    long_only: bool,
    max_weight: float,
    groups: list[np.ndarray] | None = None,
    sector_cap: float = 1.0,
) -> np.ndarray:
    """Score divided by vol. Long-only drops non-positive scores."""
    s = np.array(scores, dtype=float, copy=True)
    v = np.array(vol, dtype=float, copy=True)
    s[~np.isfinite(s)] = 0.0
    v = np.where(np.isfinite(v) & (v > 1e-8), v, np.nan)
    raw = s / v
    raw[~np.isfinite(raw)] = 0.0
    if long_only:
        raw = np.clip(raw, 0.0, None)
    else:
        active = s != 0
        raw[~active] = 0.0
        if int(active.sum()) >= 2:
            raw[active] = raw[active] - float(np.mean(raw[active]))
    denom = float(np.abs(raw).sum())
    if denom <= 0 or gross <= 0:
        return np.zeros_like(s)
    return project_weights(raw / denom * gross, gross, long_only, max_weight, groups, sector_cap)


def _corr_distance(cov: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(scale, scale)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))


def _single_linkage_order(distance: np.ndarray) -> list[int]:
    """Agglomerative order. Each merge concatenates the two cluster orders."""
    n = distance.shape[0]
    clusters = [[i] for i in range(n)]
    dist = distance.copy()
    np.fill_diagonal(dist, np.inf)
    while len(clusters) > 1:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if i > j:
            i, j = j, i
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        # Distance from the merged cluster to the others is the min link.
        keep = [k for k in range(dist.shape[0]) if k != j]
        reduced = np.array(dist[np.ix_(keep, keep)], copy=True)
        merged_pos = i if i < j else i - 1
        for k in range(len(keep)):
            if k == merged_pos:
                continue
            a = keep[k]
            # original indices before deletion
            left = i
            right = j
            reduced[merged_pos, k] = reduced[k, merged_pos] = min(dist[left, a], dist[right, a])
        dist = reduced
        np.fill_diagonal(dist, np.inf)
    return clusters[0]


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    inv = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    w = inv / inv.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only hierarchical risk parity. Weights are positive and sum to 1."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    order = _single_linkage_order(_corr_distance(cov))
    weights = np.ones(n)
    groups = [order]
    while groups:
        nxt = []
        for group in groups:
            if len(group) <= 1:
                continue
            split = len(group) // 2
            left, right = group[:split], group[split:]
            var_l = _cluster_variance(cov, left)
            var_r = _cluster_variance(cov, right)
            total = var_l + var_r
            alpha = 0.5 if total <= 0 else 1.0 - var_l / total
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            if len(left) > 1:
                nxt.append(left)
            if len(right) > 1:
                nxt.append(right)
        groups = nxt
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return np.full(n, 1.0 / n)
    return weights / total


def hrp_score_weights(
    scores: np.ndarray,
    cov: np.ndarray,
    gross: float,
    long_only: bool,
    max_weight: float,
    groups: list[np.ndarray] | None = None,
    sector_cap: float = 1.0,
) -> np.ndarray:
    """HRP magnitudes, with the score deciding the sign and a long-only mask."""
    base = hrp_weights(cov)
    s = np.array(scores, dtype=float, copy=True)
    s[~np.isfinite(s)] = 0.0
    if long_only:
        raw = base * np.clip(s, 0.0, None)
    else:
        raw = base * np.sign(s)
        raw[s == 0] = 0.0
    denom = float(np.abs(raw).sum())
    if denom <= 0 or gross <= 0:
        return np.zeros(len(base))
    return project_weights(raw / denom * gross, gross, long_only, max_weight, groups, sector_cap)


def mean_variance_weights(
    scores: np.ndarray,
    cov: np.ndarray,
    gross: float,
    long_only: bool,
    max_weight: float,
    groups: list[np.ndarray] | None = None,
    sector_cap: float = 1.0,
    risk_aversion: float = 3.0,
) -> np.ndarray:
    """Projected mean-variance. ``scores`` stand in for expected returns."""
    mu = np.array(scores, dtype=float, copy=True)
    mu[~np.isfinite(mu)] = 0.0
    try:
        raw = np.linalg.solve(cov, mu) / risk_aversion
    except np.linalg.LinAlgError:
        raw = mu.copy()
    raw[~np.isfinite(raw)] = 0.0
    if long_only:
        raw = np.clip(raw, 0.0, None)
    denom = float(np.abs(raw).sum())
    if denom <= 0 or gross <= 0:
        return np.zeros_like(mu)
    return project_weights(raw / denom * gross, gross, long_only, max_weight, groups, sector_cap)
