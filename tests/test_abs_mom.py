"""Absolute-momentum ensemble. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.absmom import (
    LONG,
    MID,
    MIN_POSITIVE,
    MONTH,
    OFF_NAME,
    SHORTS,
    absolute_momentum_state,
    cells,
    horizons_for,
    kept_abs_momentum,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def test_grid_is_the_3_6_12_month_set_and_every_cell_has_a_neighbor():
    assert MONTH == 21
    assert SHORTS == (42, 63, 84)
    assert MID == 126 and LONG == 252
    assert MIN_POSITIVE == (2, 3)
    assert OFF_NAME == "mom_off"
    found = cells()
    assert found == [
        (42, 2), (42, 3),
        (63, 2), (63, 3),
        (84, 2), (84, 3),
    ]
    names = [variant_name(*cell) for cell in found]
    assert names == [
        "zmom_s042_k2", "zmom_s042_k3",
        "zmom_s063_k2", "zmom_s063_k3",
        "zmom_s084_k2", "zmom_s084_k3",
    ]
    assert OFF_NAME < min(names)
    assert horizons_for(63) == (63, 126, 252)
    with pytest.raises(ValueError):
        variant_name(21, 2)
    with pytest.raises(ValueError):
        variant_name(63, 1)
    for cell in found:
        neighbors = adjacent_cells(cell, [SHORTS, MIN_POSITIVE])
        assert neighbors
        assert cell not in neighbors
    assert set(adjacent_cells((63, 2), [SHORTS, MIN_POSITIVE])) == {(42, 2), (84, 2), (63, 3)}


def test_state_uses_return_signs_and_does_not_read_a_future_close():
    idx = pd.bdate_range("2020-01-02", periods=6)
    close = pd.Series([100.0, 110.0, 120.0, 130.0, 125.0, 140.0], index=idx)
    majority = absolute_momentum_state(close, (1, 2, 3), 2)
    unanimous = absolute_momentum_state(close, (1, 2, 3), 3)
    # Two of three returns are positive at the fifth close. The third is not.
    assert bool(majority.iloc[4]) is True
    assert bool(unanimous.iloc[4]) is False
    # All three are positive at the last close.
    assert bool(majority.iloc[5]) is True
    assert bool(unanimous.iloc[5]) is True
    # A horizon that does not exist yet leaves the state off.
    assert bool(majority.iloc[2]) is False
    # A zero return is not a positive vote.
    flat = close.copy()
    flat.iloc[5] = flat.iloc[4]
    tied = absolute_momentum_state(flat, (1, 2, 3), 3)
    assert bool(tied.iloc[5]) is False

    future = close.copy()
    future.iloc[5] = 50.0
    again = absolute_momentum_state(future, (1, 2, 3), 2)
    pd.testing.assert_series_equal(majority.iloc[:5], again.iloc[:5])


def test_override_replaces_the_sleeve_gate():
    idx = pd.bdate_range("2016-01-04", periods=40)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    off = pd.Series(False, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(window=10), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off,
        mom_info={"momentum_rule": "zmom_s063_k2", "momentum_on": False},
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    assert payload["diagnostics"]["trend_on"] is False
    assert payload["diagnostics"]["momentum_rule"] == "zmom_s063_k2"


def test_rejected_rule_does_not_change_the_signal(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=30)
    close = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": False,
            "status": "tested",
            "overlay": "abs_momentum",
            "walk_forward_picks": [["2018-01-02", "zmom_s063_k2"]],
            "variants": [{"name": "zmom_s063_k2", "short": 63, "min_positive": 2}],
        }
    ]))
    state, info = kept_abs_momentum(path, close, as_of=idx[-1])
    assert state is None and info is None


def test_kept_rule_reloads_only_closes_through_as_of(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=80)
    close = pd.Series(np.linspace(100, 180, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "abs_momentum",
            "walk_forward_picks": [["2018-01-02", "zmom_s063_k2"]],
            "variants": [{"name": "zmom_s063_k2", "short": 63, "min_positive": 2}],
        }
    ]))
    state, info = kept_abs_momentum(path, close, as_of=idx[70])
    assert info["momentum_rule"] == "zmom_s063_k2"
    assert info["momentum_horizons"] == [63, 126, 252]
    assert len(state) == 71
    # The 252-session horizon is not ready inside this short sample, so the sleeve stays off.
    assert bool(state.any()) is False

    off_path = tmp_path / "off.json"
    off_path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "abs_momentum",
            "walk_forward_picks": [["2018-01-02", "mom_off"]],
            "variants": [{"name": "mom_off"}],
        }
    ]))
    state, info = kept_abs_momentum(off_path, close, as_of=idx[-1])
    assert state is None and info is None
