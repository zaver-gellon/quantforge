"""Multi-lookback vote. No market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state
from quantforge.strategies.vote import (
    BANDS,
    MIN_VOTES,
    OFF_NAME,
    SHORT_WINDOWS,
    cells,
    kept_vote_trend,
    variant_name,
    vote_state,
    windows_for,
)


def _close(n=120, seed=5):
    idx = pd.bdate_range("2018-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n))), index=idx)
    return idx, close


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert len(found) == 8
    assert len({variant_name(*cell) for cell in found}) == 8
    assert windows_for(50) == (50, 100, 200)
    assert windows_for(100) == (100, 150, 200)
    for cell in found:
        neighbors = adjacent_cells(cell, [SHORT_WINDOWS, MIN_VOTES, BANDS])
        assert neighbors
        assert cell not in neighbors


def test_vote_does_not_read_a_future_close():
    idx, close = _close(90)
    base = vote_state(close, (10, 20, 40), 0.02, 2)
    future = close.copy()
    future.iloc[50:] = close.iloc[50] * 0.4
    again = vote_state(future, (10, 20, 40), 0.02, 2)
    pd.testing.assert_series_equal(base.iloc[:50], again.iloc[:50])


def test_unanimous_vote_matches_the_intersection_of_the_legs():
    _idx, close = _close(80, seed=9)
    windows = (8, 15, 30)
    vote = vote_state(close, windows, 0.0, 3)
    legs = [trend_state(close, window, 0.0, "daily") for window in windows]
    expected = legs[0] & legs[1] & legs[2]
    pd.testing.assert_series_equal(vote, expected, check_names=False)


def test_majority_can_turn_off_while_the_slow_average_is_still_on():
    idx = pd.bdate_range("2019-01-02", periods=80)
    # A long rise, then a break that pulls the short averages down first.
    rise = np.linspace(100, 160, 60)
    drop = np.linspace(160, 110, 20)
    close = pd.Series(np.concatenate([rise, drop]), index=idx)
    windows = (5, 10, 40)
    slow = trend_state(close, 40, 0.0, "daily")
    fast = trend_state(close, 5, 0.0, "daily")
    mid = trend_state(close, 10, 0.0, "daily")
    vote = vote_state(close, windows, 0.0, 2)
    fast_broken = ~fast & ~mid & slow
    assert bool(fast_broken.any())
    assert not bool(vote.loc[fast_broken].any())
    assert bool((slow & ~vote).any())


def test_override_replaces_the_sleeve_and_ignores_a_future_vote():
    idx = pd.bdate_range("2016-01-04", periods=80)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    off = pd.Series(False, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off, vote_info={"vote_rule": "vote_s50_k2_b0", "vote_on": False},
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    assert payload["diagnostics"]["trend_on"] is False
    assert payload["diagnostics"]["vote_rule"] == "vote_s50_k2_b0"

    future = off.copy()
    future.iloc[-1] = True
    earlier = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=future,
    )
    same = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[40], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=off,
    )
    assert earlier["diagnostics"]["trend_on"] is False
    assert earlier["weights"] == same["weights"]
    assert earlier["diagnostics"]["trend_on"] == same["diagnostics"]["trend_on"]


def test_rejected_vote_does_not_change_the_signal(tmp_path):
    idx, close = _close(60)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": False,
            "status": "tested",
            "overlay": "trend_vote",
            "walk_forward_picks": [["2018-01-02", "vote_s50_k3_b0"]],
            "variants": [{"name": "vote_s50_k3_b0", "short_window": 50, "min_votes": 3, "band": 0.0}],
        }
    ]))
    state, info = kept_vote_trend(path, close, as_of=idx[-1])
    assert state is None and info is None


def test_kept_vote_uses_only_closes_through_as_of(tmp_path):
    idx, close = _close(80, seed=4)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([
        {
            "kept": True,
            "status": "tested",
            "overlay": "trend_vote",
            "walk_forward_picks": [["2018-01-02", "vote_s50_k2_b0"]],
            "variants": [{"name": "vote_s50_k2_b0", "short_window": 50, "min_votes": 2, "band": 0.0}],
        }
    ]))
    # windows_for(50) needs 200 sessions, so this history stays off. The check
    # is that a later close cannot change the state at an earlier as_of.
    state, info = kept_vote_trend(path, close, as_of=idx[60])
    assert info["vote_rule"] == "vote_s50_k2_b0"
    assert info["vote_windows"] == [50, 100, 200]
    future = close.copy()
    future.iloc[61:] = close.iloc[61] * 3
    again, _info = kept_vote_trend(path, future, as_of=idx[60])
    pd.testing.assert_series_equal(state, again)
    assert OFF_NAME == "vote_off"
