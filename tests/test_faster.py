"""Faster books and the RSI boost. Synthetic prices only."""

import numpy as np
import pandas as pd
import pytest

from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.boost import (
    FRACTIONS,
    HOLDS,
    THRESHOLDS,
    book_frame,
    boost_episodes,
    boost_flag,
    cells as boost_cells,
    close_to_close_book,
    episode_summary,
    hold_flag,
    named_cell,
    parse_cell,
)
from quantforge.strategies.exposure import gross_exposure, mix_weights, rolling_total, trail_rates
from quantforge.strategies.fastsignal import book_weights, boost_signal
from quantforge.strategies.meanrev import cells as meanrev_cells, target_flag, wilder_rsi
from quantforge.strategies.nasdaqvol import cells as nasdaq_cells, tqqq_weight
from quantforge.engine.schedule import week_ends
from quantforge.strategies.rotation import weekly_pick, cell_name as rotation_name, cells as rotation_cells


def test_wilder_rsi_matches_a_hand_average_and_ignores_the_future():
    close = pd.Series([10.0, 9.0, 8.0, 9.0, 11.0], index=pd.bdate_range("2020-01-02", periods=5))
    rsi = wilder_rsi(close, 2)
    # Seed on the first two changes (-1, -1): RSI 0. Then +1 -> 50. Then +2 -> 83.333.
    assert rsi.iloc[2] == pytest.approx(0.0)
    assert rsi.iloc[3] == pytest.approx(50.0)
    assert rsi.iloc[4] == pytest.approx(100.0 - 100.0 / 6.0)
    cut = wilder_rsi(close.iloc[:4], 2)
    pd.testing.assert_series_equal(cut, rsi.iloc[:4])


def test_exposure_mix_is_the_gross_equity_beta():
    assert mix_weights(2.0) == pytest.approx((0.5, 0.5))
    assert mix_weights(1.5) == pytest.approx((0.75, 0.25))
    assert mix_weights(2.5) == pytest.approx((0.25, 0.75))
    assert mix_weights(1.0) == pytest.approx((1.0, 0.0))
    for exposure in (1.0, 1.5, 2.0, 2.5):
        core, sleeve = mix_weights(exposure)
        assert gross_exposure(core, sleeve) == pytest.approx(exposure)
        assert core + sleeve == pytest.approx(1.0)


def test_rolling_window_compounds_and_counts_trails():
    returns = pd.Series([0.10, -0.05, 0.02], index=pd.bdate_range("2020-01-02", periods=3))
    rolled = rolling_total(returns, 2)
    assert rolled.iloc[0] != rolled.iloc[0]
    assert rolled.iloc[1] == pytest.approx(1.10 * 0.95 - 1.0)
    assert rolled.iloc[2] == pytest.approx(0.95 * 1.02 - 1.0)
    strategy = pd.Series([0.01, -0.02, 0.03, -0.04], index=pd.bdate_range("2020-01-02", periods=4))
    rates = trail_rates(strategy, strategy * 0.0 + 0.05, {"1": 2})
    assert rates["1"]["trail_rate"] == pytest.approx(1.0)


def test_boost_hold_exits_on_the_cap_and_does_not_reenter_that_close():
    idx = pd.bdate_range("2020-01-02", periods=6)
    entry = pd.Series([True, True, True, True, True, False], index=idx)
    leave = pd.Series(False, index=idx)
    flag = hold_flag(entry, leave, 3)
    assert list(flag.astype(bool)) == [True, True, True, False, True, True]
    early = hold_flag(entry, pd.Series([False, True, False, False, False, False], index=idx), 10)
    assert list(early.astype(bool))[:3] == [True, False, True]


def test_boost_grid_neighbors_and_named_weights():
    assert len(boost_cells()) == 3 * 3 * 2
    assert named_cell() == "rsi10_f33_n10"
    threshold, fraction, hold = parse_cell(named_cell())
    assert (threshold, hold) == (10.0, 10)
    assert fraction == pytest.approx(1.0 / 3.0)
    for cell in boost_cells():
        neighbors = adjacent_cells(cell, [THRESHOLDS, FRACTIONS, HOLDS])
        assert neighbors
        assert cell not in neighbors
    idx = pd.bdate_range("2020-01-02", periods=3)
    weights = book_frame(pd.Series([False, True, False], index=idx), 1.0 / 3.0)
    assert weights["SPY"].iloc[0] == pytest.approx(1.0)
    assert weights["UPRO"].iloc[0] == pytest.approx(0.0)
    assert weights["SPY"].iloc[1] == pytest.approx(2.0 / 3.0)
    assert weights["UPRO"].iloc[1] == pytest.approx(1.0 / 3.0)


def test_close_fill_uses_the_previous_close_and_not_the_same_bar():
    idx = pd.bdate_range("2020-01-02", periods=4)
    spy = pd.Series([100.0, 100.0, 110.0, 110.0], index=idx)
    upro = pd.Series([100.0, 100.0, 130.0, 130.0], index=idx)
    weights = pd.DataFrame({"SPY": [1.0, 0.0, 0.0, 0.0], "UPRO": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    rf = pd.Series(0.0, index=idx)
    out = close_to_close_book(weights, pd.DataFrame({"SPY": spy, "UPRO": upro}), rf, cost=0.0)
    # The switch is decided at the second close. The third session's close-to-close move is UPRO's.
    assert out.loc[idx[2]] == pytest.approx(0.30)
    # A later price does not change the earlier return.
    upro2 = upro.copy()
    upro2.iloc[-1] = 200.0
    again = close_to_close_book(weights, pd.DataFrame({"SPY": spy, "UPRO": upro2}), rf, cost=0.0)
    assert again.loc[idx[2]] == pytest.approx(out.loc[idx[2]])


def test_boost_episodes_count_a_winning_hold_and_ignore_the_flat_days():
    idx = pd.bdate_range("2020-01-02", periods=5)
    flag = pd.Series([True, True, False, False, False], index=idx)
    port = pd.Series([0.0, 0.10, -0.05, 0.0, 0.0], index=idx)
    spy = pd.Series(0.0, index=idx)
    episodes = boost_episodes(flag, port, spy)
    assert len(episodes) == 1
    assert episodes[0]["days"] == 2
    assert episodes[0]["pnl"] == pytest.approx(1.10 * 0.95 - 1.0)
    assert episodes[0]["win"]
    summary = episode_summary(episodes, years=1.0)
    assert summary["per_year"] == pytest.approx(1.0)
    assert summary["win_rate"] == pytest.approx(1.0)


def test_boost_signal_does_not_read_past_as_of_and_names_the_rsi():
    idx = pd.bdate_range("2018-01-02", periods=260)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, len(idx))), index=idx)
    full = boost_signal(close, named_cell())
    cut = boost_signal(close.iloc[:-5], named_cell(), as_of=close.index[-6])
    assert cut["as_of"] == str(close.index[-6].date())
    assert cut["date"] > cut["as_of"]
    assert set(cut["weights"]) <= {"SPY", "UPRO", "BIL"}
    assert sum(cut["weights"].values()) == pytest.approx(1.0)
    assert "rsi2" in cut["levels"]
    assert cut["levels"]["stop"] == "none"
    again = boost_signal(close, named_cell(), as_of=close.index[-6])
    assert again["weights"] == cut["weights"]
    assert again["levels"]["boost_on"] == cut["levels"]["boost_on"]
    _ = full


def test_meanrev_flag_does_not_use_a_future_close_and_the_grids_have_neighbors():
    idx = pd.bdate_range("2019-01-02", periods=40)
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    entry = pd.Series(False, index=idx)
    entry.iloc[10] = True
    regime = pd.Series(True, index=idx)
    flag = target_flag(close, entry, regime, 4, exit_window=3)
    for end in range(12, len(close)):
        cut = target_flag(close.iloc[: end + 1], entry.iloc[: end + 1], regime.iloc[: end + 1], 4, exit_window=3)
        pd.testing.assert_series_equal(cut, flag.iloc[: end + 1])
    assert len(meanrev_cells()) == 5 * 3 * 3
    assert len(rotation_cells()) == 6
    assert len(nasdaq_cells()) == 6
    assert rotation_name("UPRO", 21) == "UPRO_L21"


def test_week_end_pick_carries_forward_and_loses_to_cash():
    idx = pd.bdate_range("2024-01-02", periods=12)
    ends = week_ends(idx)
    assert bool(ends.iloc[0]) is False or ends.sum() >= 1
    scores = {
        "UPRO": pd.Series(0.01, index=idx),
        "TMF": pd.Series(-0.02, index=idx),
        "UGL": pd.Series(0.0, index=idx),
    }
    # Cash earns more than every sleeve, so the pick stays in BIL.
    pick = weekly_pick(scores, pd.Series(0.02, index=idx), idx)
    assert set(pick.unique()) == {"BIL"}
    scores["UPRO"] = pd.Series(0.05, index=idx)
    picked = weekly_pick(scores, pd.Series(0.0, index=idx), idx)
    assert picked.iloc[-1] == "UPRO"
    # A future jump in TMF does not change the earlier week.
    jumped = scores["TMF"].copy()
    jumped.iloc[-1] = 0.50
    scores2 = dict(scores)
    scores2["TMF"] = jumped
    earlier = weekly_pick(scores2, pd.Series(0.0, index=idx), idx)
    assert list(earlier.iloc[:-1]) == list(picked.iloc[:-1])


def test_vol_target_is_capped_and_flat_when_the_trend_is_off():
    idx = pd.bdate_range("2018-01-02", periods=260)
    qqq = pd.Series(np.linspace(100, 180, len(idx)), index=idx)
    noisy = pd.Series(100.0 * np.cumprod(1.0 + np.resize([0.02, -0.02], len(idx))), index=idx)
    calm = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    hot = tqqq_weight(qqq, noisy, 0.0, 0.20)
    quiet = tqqq_weight(qqq, calm, 0.0, 0.40)
    assert hot.max() <= 1.0 + 1e-12
    assert quiet.max() == pytest.approx(1.0)
    falling = pd.Series(np.linspace(180, 80, len(idx)), index=idx)
    off = tqqq_weight(falling, noisy, 0.02, 0.30)
    assert off.iloc[-1] == pytest.approx(0.0)


def test_book_weights_put_the_residual_in_bil_and_reject_leverage():
    assert book_weights({"UPRO": 0.4}) == {"UPRO": 0.4, "BIL": 0.6}
    assert book_weights({}) == {"BIL": 1.0}
    with pytest.raises(ValueError):
        book_weights({"UPRO": 1.2})
