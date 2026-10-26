"""The iteration bar. Synthetic paths only, no market data."""

import numpy as np
import pandas as pd

from quantforge.eval.materiality import (
    DRAWDOWN_IMPROVEMENT,
    IMPROVEMENT_DSR_MIN,
    SHARPE_IMPROVEMENT,
    adjacent_cells,
    criterion_a,
    evaluate_materiality,
)


def _path(seed: int, drift: float, periods: int = 900) -> pd.Series:
    idx = pd.bdate_range("2006-01-03", periods=periods)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(drift, 0.01, periods), index=idx)


def test_adjacent_cells_step_one_axis():
    axes = [(10, 21, 63), (0.10, 0.15, 0.20, 0.25)]
    neighbors = adjacent_cells((21, 0.15), axes)
    assert (10, 0.15) in neighbors
    assert (63, 0.15) in neighbors
    assert (21, 0.10) in neighbors
    assert (21, 0.20) in neighbors
    assert (10, 0.10) not in neighbors
    edge = adjacent_cells((0.10,), [(0.10, 0.20, 0.35, 0.50)])
    assert edge == [(0.20,)]


def test_sharpe_clause_needs_five_hundredths():
    baseline = {"sharpe": 0.60, "cagr": 0.15, "max_drawdown": -0.40}
    short = criterion_a({**baseline, "sharpe": 0.60 + SHARPE_IMPROVEMENT - 0.01}, baseline)
    exact = criterion_a({**baseline, "sharpe": 0.60 + SHARPE_IMPROVEMENT}, baseline)
    assert short["sharpe_pass"] is False
    assert exact["sharpe_pass"] is True
    assert exact["pass"] is True


def test_drawdown_clause_caps_the_cagr_giveup():
    baseline = {"sharpe": 0.60, "cagr": 0.15, "max_drawdown": -0.47}
    ok = criterion_a(
        {"sharpe": 0.60, "cagr": 0.15 - 0.005, "max_drawdown": -0.47 + DRAWDOWN_IMPROVEMENT},
        baseline,
    )
    giveup = criterion_a(
        {"sharpe": 0.60, "cagr": 0.15 - 0.006, "max_drawdown": -0.47 + DRAWDOWN_IMPROVEMENT},
        baseline,
    )
    assert ok["drawdown_pass"] is True
    assert giveup["drawdown_pass"] is False
    assert giveup["pass"] is False


def test_best_year_removal_and_neighbor_peak_are_enforced():
    idx = pd.bdate_range("2018-01-02", periods=800)
    rng = np.random.default_rng(4)
    baseline = pd.Series(rng.normal(0.0003, 0.008, len(idx)), index=idx)
    candidate = baseline.copy()
    # One year carries the whole improvement. The rest of the path matches.
    bump = candidate.index.year == 2019
    candidate.loc[bump] = candidate.loc[bump] + 0.004
    cell = candidate
    neighbor = baseline + 0.00005
    result = evaluate_materiality(
        candidate, baseline, {"neighbor": neighbor}, cell,
        n_trials=30, sharpe_dispersion=0.01, rf=None,
    )
    assert result["criterion_b"]["best_year"] == 2019
    assert result["criterion_b"]["still_helps"] is False
    assert result["pass"] is False

    flat = baseline + 0.0002
    close = flat + 0.00001
    plateau = evaluate_materiality(
        flat, baseline, {"neighbor": close}, flat,
        n_trials=2, sharpe_dispersion=0.0, rf=None,
    )
    assert plateau["criterion_c"]["pass"] is True
    assert plateau["checks"][-1]["name"] == "c neighboring parameters"

    peak = evaluate_materiality(
        flat, baseline, {"neighbor": baseline}, flat,
        n_trials=2, sharpe_dispersion=0.0, rf=None,
    )
    assert peak["criterion_c"]["pass"] is False
    assert IMPROVEMENT_DSR_MIN == 0.75
