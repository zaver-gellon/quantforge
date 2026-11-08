"""Off-trend sleeve diversifier. Synthetic prices only."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_idle_diversifier, run_sleeve_diversifier, run_trend_core
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.sleevediv import (
    ASSETS,
    BANDS,
    OFF_NAME,
    WINDOWS,
    cells,
    sleeve_cash_signal_kwargs,
    sleeve_cash_state,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal, trend_state


def _prices(n=40, step=0.0):
    idx = pd.bdate_range("2016-01-04", periods=n)
    close = pd.Series(100.0 * np.cumprod(1.0 + np.full(n, step)), index=idx)
    open_ = close.shift(1)
    open_.iloc[0] = close.iloc[0]
    return idx, open_, close


def test_grid_is_fixed_and_every_cell_has_a_neighbor():
    found = cells()
    assert found[0] == ("GLD", 100, 0.0)
    assert found[-1] == ("TLT", 200, 0.02)
    assert len(found) == 12
    assert ASSETS == ("GLD", "IEF", "TLT")
    assert WINDOWS == (100, 200)
    assert BANDS == (0.0, 0.02)
    assert OFF_NAME == "sdiv_off"
    names = {variant_name(*cell) for cell in found}
    assert len(names) == 12
    assert variant_name("GLD", 100, 0.0) == "sdiv_gld_w100_b0"
    assert variant_name("TLT", 200, 0.02) == "sdiv_tlt_w200_b2"
    assert OFF_NAME not in names
    for cell in found:
        neighbors = adjacent_cells(cell, [ASSETS, WINDOWS, BANDS])
        assert neighbors
        assert cell not in neighbors
        for neighbor in neighbors:
            assert neighbor in found


def test_state_does_not_read_a_future_close():
    idx = pd.bdate_range("2014-01-02", periods=80)
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    state, ready = sleeve_cash_state(close, 20, 0.0, idx)
    assert not bool(ready.iloc[:19].any())
    assert not bool(state.iloc[:19].any())
    future = close.copy()
    future.iloc[40:] = 50.0
    again, again_ready = sleeve_cash_state(future, 20, 0.0, idx)
    pd.testing.assert_series_equal(state.iloc[:40], again.iloc[:40])
    pd.testing.assert_series_equal(ready.iloc[:40], again_ready.iloc[:40])


def _book(n=30, spy_step=0.0, div_step=0.0):
    idx, spy_open, spy_close = _prices(n, spy_step)
    _, div_open, div_close = _prices(n, div_step)
    # Leveraged legs are flat so a sleeve in UPRO does not add a price move.
    sso_open = pd.Series(110.0, index=idx)
    sso_close = pd.Series(110.0, index=idx)
    upro_open = pd.Series(120.0, index=idx)
    upro_close = pd.Series(120.0, index=idx)
    return idx, spy_open, spy_close, sso_open, sso_close, upro_open, upro_close, div_open, div_close


def test_unready_or_idle_filter_matches_the_trend_core_and_ignores_future_div_prices():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, div_o, div_c = _book(36, spy_step=0.001)
    trend = pd.Series(False, index=idx)
    trend.iloc[10:20] = True
    dip = pd.Series(False, index=idx)
    dip.iloc[4:8] = True
    mix = pd.Series(False, index=idx)
    mix.iloc[18] = True
    rf = pd.Series(0.0001, index=idx)
    ready = pd.Series(False, index=idx)
    div_on = pd.Series(True, index=idx)
    kwargs = dict(spy_cost=0.0005, sso_cost=0.0005, upro_cost=0.0005, div_cost=0.0005)
    plain = run_trend_core(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        trend, dip, mix, 0.5, 0.5, rf, **{k: kwargs[k] for k in ("spy_cost", "sso_cost", "upro_cost")},
    )
    parked = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o * 3, div_c * 3, trend, div_on, ready, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    pd.testing.assert_series_equal(parked.returns, plain.returns, check_names=False)
    # Filter ready but off is still the trend core, even if the diversifier rallies.
    ready_off = pd.Series(True, index=idx)
    div_off = pd.Series(False, index=idx)
    still = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o * 3, div_c * 3, trend, div_off, ready_off, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    pd.testing.assert_series_equal(still.returns, plain.returns, check_names=False)
    # A trend-on book does not hold the diversifier.
    always_on = pd.Series(True, index=idx)
    ignored = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o * 3, div_c * 3, always_on, div_on, ready_off, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    held = run_trend_core(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        always_on, dip, mix, 0.5, 0.5, rf, **{k: kwargs[k] for k in ("spy_cost", "sso_cost", "upro_cost")},
    )
    pd.testing.assert_series_equal(ignored.returns, held.returns, check_names=False)

    always_off = pd.Series(False, index=idx)
    ready_all = pd.Series(True, index=idx)
    div_held = pd.Series(True, index=idx)
    base_on = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o, div_c, always_off, div_held, ready_all, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    bumped = div_c.copy()
    bumped.iloc[12:] *= 2.0
    later = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o, bumped, always_off, div_held, ready_all, dip, mix, 0.5, 0.5, rf, **kwargs,
    )
    pd.testing.assert_series_equal(later.returns.iloc[:12], base_on.returns.iloc[:12], check_names=False)
    assert not np.isclose(later.returns.iloc[12], base_on.returns.iloc[12])


def test_off_trend_sleeve_earns_the_diversifier_and_the_core_stays_in_spy():
    idx, spy_o, spy_c, sso_o, sso_c, upro_o, upro_c, div_o, div_c = _book(8, spy_step=0.01, div_step=0.0)
    trend = pd.Series(False, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    ready = pd.Series(True, index=idx)
    div_on = pd.Series(True, index=idx)
    # Zero costs so the first open's 50/50 mix is a pure split of the open-to-close move.
    result = run_sleeve_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o, div_c, trend, div_on, ready, dip, mix, 0.5, 0.5, rf,
    )
    # Overnight is still cash. The open buys half SPY, which then rises 1%, and half flat diversifier.
    assert result.returns.iloc[1] == pytest.approx(0.005)
    whole = run_idle_diversifier(
        spy_o, spy_c, sso_o, sso_c, upro_o, upro_c,
        div_o, div_c, trend, div_on, ready, dip, mix, 0.5, 0.5, rf,
    )
    # The whole-book rule sells the core, so a flat diversifier earns nothing on that open-to-close.
    assert whole.returns.iloc[1] == pytest.approx(0.0)
    assert result.returns.iloc[1] != pytest.approx(whole.returns.iloc[1])


def test_rejected_rule_is_not_reloaded_and_a_kept_rule_changes_only_the_sleeve(tmp_path):
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(80.0, 120.0, len(idx)), index=idx)
    div = pd.Series(np.linspace(50.0, 90.0, len(idx)), index=idx)
    path = tmp_path / "iterations.json"
    path.write_text(json.dumps([{
        "kept": False,
        "status": "tested",
        "overlay": "sleeve_cash",
        "walk_forward_picks": [["2018-01-02", "sdiv_gld_w100_b0"]],
        "variants": [{"name": "sdiv_gld_w100_b0", "symbol": "GLD", "window": 100, "band": 0.0}],
    }]))
    assert sleeve_cash_signal_kwargs(path, close, idx[-1], {"GLD": div}) == {}

    path.write_text(json.dumps([{
        "kept": True,
        "status": "tested",
        "overlay": "sleeve_cash",
        "walk_forward_picks": [
            ["2018-01-02", "sdiv_gld_w100_b0"],
            ["2018-02-01", "sdiv_off"],
        ],
        "variants": [
            {"name": "sdiv_gld_w100_b0", "symbol": "GLD", "window": 100, "band": 0.0},
            {"name": "sdiv_off", "symbol": None, "window": None, "band": None},
        ],
    }]))
    assert sleeve_cash_signal_kwargs(path, close, idx[-1], {"GLD": div}) == {}
    earlier = sleeve_cash_signal_kwargs(path, close, pd.Timestamp("2018-01-31"), {"GLD": div})
    assert earlier["sleeve_cash_symbol"] == "GLD"
    assert earlier["sleeve_cash_info"]["sleeve_cash_window"] == 100
    assert earlier["sleeve_cash_on"].index[-1] <= pd.Timestamp("2018-01-31")
    bumped = div.copy()
    bumped.iloc[-1] = 10.0
    unchanged = sleeve_cash_signal_kwargs(path, close, pd.Timestamp("2018-01-31"), {"GLD": bumped})
    pd.testing.assert_series_equal(unchanged["sleeve_cash_on"], earlier["sleeve_cash_on"])
    assert unchanged["sleeve_cash_info"]["sleeve_cash_filter_on"] == earlier["sleeve_cash_info"]["sleeve_cash_filter_on"]

    trend = pd.Series(False, index=idx)
    ready = pd.Series(True, index=idx)
    on = pd.Series(True, index=idx)
    payload = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=trend,
        sleeve_cash_on=on, sleeve_cash_ready=ready, sleeve_cash_symbol="GLD",
        sleeve_cash_info=earlier["sleeve_cash_info"],
    )
    assert payload["weights"] == {"SPY": 0.5, "GLD": 0.5}
    assert payload["diagnostics"]["sleeve_cash_applied"] is True
    assert payload["diagnostics"]["trend_on"] is False
    assert "BIL" not in payload["weights"]
    assert "UPRO" not in payload["weights"]

    trend_on = pd.Series(True, index=idx)
    held = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=trend_on,
        sleeve_cash_on=on, sleeve_cash_ready=ready, sleeve_cash_symbol="GLD",
    )
    assert held["weights"]["SPY"] == pytest.approx(0.5)
    assert held["weights"]["UPRO"] == pytest.approx(0.5)
    assert "GLD" not in held["weights"]
    assert held["diagnostics"]["sleeve_cash_applied"] is False

    # Three down closes while the band is off: the core becomes SSO and the sleeve stays GLD.
    dip_close = close.copy()
    dip_close.iloc[-4:] = [100.0, 99.0, 98.0, 97.0]
    dipped = build_trend_signal(
        TrendCoreSpec(), dip_close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        trend_override=trend,
        sleeve_cash_on=on, sleeve_cash_ready=ready, sleeve_cash_symbol="GLD",
    )
    assert dipped["weights"] == {"SSO": 0.5, "GLD": 0.5}
    assert dipped["diagnostics"]["dip_on"] is True

    flipping = on.copy()
    flipping.iloc[-2] = False
    trade = build_trend_signal(
        TrendCoreSpec(), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, trend_override=trend,
        sleeve_cash_on=flipping, sleeve_cash_ready=ready, sleeve_cash_symbol="GLD",
    )
    assert trade["trade"] is True
    assert trade["trade_reason"] == "sleeve_diversifier"

    with pytest.raises(ValueError, match="whole-book"):
        build_trend_signal(
            TrendCoreSpec(), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, trend_override=trend,
            idle_symbol="TLT", idle_on=on, idle_ready=ready,
            sleeve_cash_on=on, sleeve_cash_ready=ready, sleeve_cash_symbol="GLD",
        )

    # A falling diversifier is not the same state as a rising equity average.
    equity = trend_state(close, 20, 0.0, "daily")
    falling = div.copy()
    falling.iloc[20:] = np.linspace(float(falling.iloc[20]), 10.0, len(idx) - 20)
    other, _ready = sleeve_cash_state(falling, 20, 0.0, idx)
    assert not equity.equals(other)
