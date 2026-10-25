"""Combinatorial purged splits of an already-built return path.

This does not refit the book. It asks whether the path's Sharpe is an artifact
of a few blocks. Groups are contiguous. Test blocks are dropped from the
training side with an embargo of ``embargo`` sessions on each side, but only
the test-block Sharpes are returned. The selection rule for apex is still the
walk-forward, not the best of these splits.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def cpcv_sharpes(
    returns: pd.Series,
    n_groups: int = 6,
    k_test: int = 2,
    embargo: int = 21,
    periods_per_year: int = 252,
) -> dict:
    """Sharpe of every combination of ``k_test`` contiguous groups."""
    series = pd.Series(returns).dropna().astype(float).sort_index()
    if n_groups < 2 or k_test < 1 or k_test >= n_groups:
        raise ValueError("n_groups and k_test are inconsistent")
    if len(series) < n_groups * max(embargo, 5):
        return {"n_splits": 0, "sharpes": []}
    edges = np.linspace(0, len(series), n_groups + 1, dtype=int)
    groups = [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]
    values = series.to_numpy(dtype=float)
    sharpes = []
    for combo in itertools.combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[i] for i in combo])
        # Embargo: drop test observations that sit within ``embargo`` of a
        # group boundary with a non-test neighbor. Interior points stay.
        banned = np.zeros(len(values), dtype=bool)
        for i in combo:
            start = int(edges[i])
            end = int(edges[i + 1])
            if i - 1 not in combo:
                banned[start : min(len(values), start + embargo)] = True
            if i + 1 not in combo:
                banned[max(0, end - embargo) : end] = True
        keep = test_idx[~banned[test_idx]]
        if len(keep) < periods_per_year // 2:
            continue
        sample = values[keep]
        deviation = float(sample.std(ddof=1))
        if deviation <= 0:
            continue
        sharpes.append(float(sample.mean() / deviation * np.sqrt(periods_per_year)))
    if not sharpes:
        return {"n_splits": 0, "sharpes": []}
    arr = np.asarray(sharpes, dtype=float)
    return {
        "n_splits": int(len(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "fraction_positive": float(np.mean(arr > 0)),
        "sharpes": [float(v) for v in arr],
    }
