import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_backtest, run_core_satellite
from quantforge.engine.costs import CostModel


def _flat(n=6, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    px = pd.DataFrame({"ASSET": np.full(n, 100.0)}, index=idx)
    rf = pd.Series(0.0, index=idx)
    return idx, px, rf


def test_next_open_misses_the_gap_it_trades_into():
    idx = pd.bdate_range("2020-01-02", periods=4)
    close = pd.Series([100.0, 100.0, 150.0, 150.0], index=idx)
    open_ = pd.Series([100.0, 100.0, 150.0, 150.0], index=idx)
    # Gap is between close[1] and open[2]. A buy decided at close[1] fills at open[2].
    targets = pd.DataFrame({"ASSET": [0.0, 1.0, 1.0, 1.0]}, index=idx)
    reb = pd.Series([False, True, False, False], index=idx)
    px_o = open_.to_frame("ASSET")
    px_c = close.to_frame("ASSET")
    result = run_backtest(px_o, px_c, targets, reb, CostModel(), pd.Series(0.0, index=idx))
    assert result.returns.iloc[2] == pytest.approx(0.0)
    # Already holding through that gap captures it.
    targets_b = pd.DataFrame({"ASSET": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    reb_b = pd.Series([True, False, False, False], index=idx)
    held = run_backtest(px_o, px_c, targets_b, reb_b, CostModel(), pd.Series(0.0, index=idx))
    assert held.returns.iloc[2] == pytest.approx(0.5)


def test_same_day_close_does_not_change_same_day_fill():
    idx = pd.bdate_range("2020-01-02", periods=4)
    # Day index 2 rallies from 100 to 130. The book is already long from the prior close.
    close = pd.Series([100.0, 100.0, 130.0, 130.0], index=idx)
    open_ = pd.Series([100.0, 100.0, 100.0, 130.0], index=idx)
    rf = pd.Series(0.0, index=idx)
    base_tgt = pd.DataFrame({"ASSET": 1.0}, index=idx)
    alt_tgt = base_tgt.copy()
    alt_tgt.iloc[2] = 0.0  # decided at the day-2 close, so it can only fill on day 3
    reb = pd.Series(True, index=idx)
    args = dict(cost=CostModel(), cash_yield=rf)
    base = run_backtest(open_.to_frame("ASSET"), close.to_frame("ASSET"), base_tgt, reb, **args)
    alt = run_backtest(open_.to_frame("ASSET"), close.to_frame("ASSET"), alt_tgt, reb, **args)
    assert base.returns.iloc[2] == pytest.approx(0.30)
    assert alt.returns.iloc[2] == pytest.approx(base.returns.iloc[2])


def test_cost_is_charged_once_per_side_and_not_on_cash():
    idx, px, rf = _flat()
    cost = CostModel(commission_bps={"ASSET": 5.0})
    targets = pd.DataFrame(0.0, index=idx, columns=["ASSET"])
    targets.iloc[0] = 1.0
    reb = pd.Series(False, index=idx)
    reb.iloc[0] = True
    entered = run_backtest(px, px, targets, reb, cost, rf)
    assert entered.returns.iloc[1] == pytest.approx(-0.0005)
    assert entered.returns.iloc[2] == pytest.approx(0.0)

    targets.iloc[1] = 0.0
    reb.iloc[1] = True
    roundtrip = run_backtest(px, px, targets, reb, cost, rf)
    assert roundtrip.returns.iloc[1] == pytest.approx(-0.0005)
    assert roundtrip.returns.iloc[2] == pytest.approx(-0.0005)
    # Cash leg is not charged, so a 100% flip is 5 bps, not 10.
    assert cost.one_way_fraction("CASH") == 0.0


def test_slippage_adds_to_commission():
    cost = CostModel(commission_bps={"SPY": 5.0}, slippage_bps={"SPY": 3.0})
    assert cost.one_way_fraction("SPY") == pytest.approx(0.0008)
    assert cost.one_way_fraction("BIL") == 0.0


def test_cash_yield_compounds_and_flat_book_earns_it():
    idx, px, _rf = _flat(n=253)
    daily = 0.01 / 252
    rf = pd.Series(daily, index=idx)
    targets = pd.DataFrame(0.0, index=idx, columns=["ASSET"])
    reb = pd.Series(False, index=idx)
    result = run_backtest(px, px, targets, reb, CostModel(), rf)
    earned = result.returns.iloc[1:].to_numpy()
    assert earned == pytest.approx(np.full(len(earned), daily))
    assert np.prod(1 + earned) == pytest.approx((1 + daily) ** len(earned))


def test_weights_drift_until_the_scheduled_rebalance():
    idx = pd.bdate_range("2020-01-02", periods=4)
    # A doubles intraday on day 1 and then stays there. B is flat. Enter 50/50 at the first open.
    open_ = pd.DataFrame({"A": [100.0, 100.0, 200.0, 200.0], "B": 100.0}, index=idx)
    close = pd.DataFrame({"A": [100.0, 200.0, 200.0, 200.0], "B": 100.0}, index=idx)
    targets = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    reb = pd.Series([True, False, True, False], index=idx)
    result = run_backtest(open_, close, targets, reb, CostModel(), pd.Series(0.0, index=idx))
    assert result.returns.iloc[1] == pytest.approx(0.5)
    assert result.turnover.iloc[2] == pytest.approx(0.0)
    # Signal at the day-2 close resets 2/3 : 1/3 back to 50/50 on the next open.
    assert result.turnover.iloc[3] == pytest.approx(abs(2 / 3 - 0.5))


def test_next_close_fill_misses_the_entry_day_move():
    idx = pd.bdate_range("2020-01-02", periods=3)
    open_ = pd.Series([100.0, 100.0, 110.0], index=idx).to_frame("ASSET")
    close = pd.Series([100.0, 110.0, 110.0], index=idx).to_frame("ASSET")
    targets = pd.DataFrame({"ASSET": 1.0}, index=idx)
    reb = pd.Series([True, False, False], index=idx)
    rf = pd.Series(0.0, index=idx)
    at_open = run_backtest(open_, close, targets, reb, CostModel(), rf, execution="next_open")
    at_close = run_backtest(open_, close, targets, reb, CostModel(), rf, execution="next_close")
    assert at_open.returns.iloc[1] == pytest.approx(0.10)
    assert at_close.returns.iloc[1] == pytest.approx(0.0)
    assert at_close.returns.iloc[2] == pytest.approx(0.0)


def test_core_satellite_entry_cost_and_gap_timing():
    idx = pd.bdate_range("2020-01-02", periods=4)
    core_o = pd.Series(100.0, index=idx)
    core_c = pd.Series(100.0, index=idx)
    sleeve_o = pd.Series([100.0, 100.0, 150.0, 150.0], index=idx)
    sleeve_c = pd.Series(100.0, index=idx)
    sleeve_c.iloc[2:] = 150.0
    on_before_gap = pd.Series([True, True, True, True], index=idx)
    on_into_gap = pd.Series([False, True, True, True], index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    early = run_core_satellite(
        core_o, core_c, sleeve_o, sleeve_c, on_before_gap, mix, 0.0, rf, execution="next_open"
    )
    late = run_core_satellite(
        core_o, core_c, sleeve_o, sleeve_c, on_into_gap, mix, 0.0, rf, execution="next_open"
    )
    assert early.returns.iloc[2] == pytest.approx(0.5)
    assert late.returns.iloc[2] == pytest.approx(0.0)
    costly = run_core_satellite(
        core_o, core_c, core_o, core_c, on_before_gap, mix, 0.5, rf,
        core_cost=0.0005, sleeve_cost=0.0005,
    )
    assert costly.returns.iloc[1] == pytest.approx(-0.0005)


def test_borrowed_cash_pays_the_financing_spread():
    idx = pd.bdate_range("2020-01-02", periods=4)
    px = pd.DataFrame({"ASSET": 100.0}, index=idx)
    targets = pd.DataFrame({"ASSET": 2.0}, index=idx)
    reb = pd.Series([True, False, False, False], index=idx)
    result = run_backtest(
        px, px, targets, reb, CostModel(), pd.Series(0.0, index=idx),
        max_gross=2.0, borrow_spread=0.01,
    )
    assert result.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[2] == pytest.approx(-0.01)


def test_dip_flip_trades_the_core_and_leaves_a_flat_sleeve_untouched():
    from quantforge.engine.backtest import run_trend_core

    idx = pd.bdate_range("2020-01-02", periods=6)
    flat = pd.Series(100.0, index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series([False, False, True, True, True, True], index=idx)
    mix = pd.Series(False, index=idx)
    result = run_trend_core(
        flat, flat, flat, flat, flat, flat,
        trend, dip, mix, 0.5, 0.5, pd.Series(0.0, index=idx),
    )
    # Decision at index 2 fills on the next session. Only the 50% core swaps.
    assert result.turnover.iloc[3] == pytest.approx(0.5)
    assert result.returns.iloc[3] == pytest.approx(0.0)
    assert result.exposure.iloc[3] == pytest.approx(1.0)
