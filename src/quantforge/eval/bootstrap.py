"""Circular block bootstrap of terminal wealth.

Blocks are drawn with replacement and wrapped around the sample, jointly with
the benchmark so "beats the benchmark" uses the same calendar blocks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def block_bootstrap(
    returns,
    horizon: int,
    n_paths: int = 20_000,
    block: int = 63,
    seed: int = 7,
    start_value: float = 5_000.0,
    benchmark=None,
) -> dict:
    series = pd.Series(returns).astype(float)
    bench = None
    if benchmark is not None:
        bench = pd.Series(benchmark).reindex(series.index).astype(float)
        mask = series.notna() & bench.notna()
        series = series[mask]
        bench = bench[mask]
    else:
        series = series.dropna()
    values = series.to_numpy(dtype=float)
    length = len(values)
    if block < 1 or length < block:
        raise ValueError("block length must be at least 1 and no longer than the sample")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(horizon / block))
    starts = rng.integers(0, length, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]) % length
    index = index.reshape(n_paths, -1)[:, :horizon]
    growth = np.prod(1.0 + values[index], axis=1)
    terminal = start_value * growth
    out = {
        "horizon_days": horizon,
        "n_paths": n_paths,
        "block": block,
        "seed": seed,
        "start_value": start_value,
        "median": float(np.median(terminal)),
        "p10": float(np.percentile(terminal, 10)),
        "p90": float(np.percentile(terminal, 90)),
        "p5": float(np.percentile(terminal, 5)),
        "p_loss": float(np.mean(growth < 1.0)),
        "p_down_30": float(np.mean(growth < 0.7)),
    }
    if bench is not None:
        bench_growth = np.prod(1.0 + bench.to_numpy(dtype=float)[index], axis=1)
        out["p_beats_benchmark"] = float(np.mean(growth > bench_growth))
    return out
