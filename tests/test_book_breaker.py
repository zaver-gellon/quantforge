"""Book-level drawdown breaker. Synthetic prices only, no market data."""

import json

import numpy as np
import pandas as pd
import pytest

from quantforge.engine.backtest import run_book_breaker, run_trend_core
from quantforge.eval.materiality import adjacent_cells
from quantforge.strategies.bookbreaker import (
    CUTS,
    MODES,
    OFF_NAME,
    REENTRY_FRACTION,
    book_drawdown,
    cells,
    has_kept_book_breaker,
    kept_book_breaker,
    step_breaker,
    variant_name,
)
from quantforge.strategies.trend import TrendCoreSpec, build_trend_signal


def _paths(n=24, drop_on=12, drop_to=70.0, end_at=70.0):
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = np.full(n, 100.0)
    opened = np.full(n, 100.0)
    if drop_on is not None:
        opened[drop_on] = 100.0
        close[drop_on] = drop_to
        if drop_on + 1 < n:
            rest = np.linspace(drop_to, end_at, n - drop_on)
            close[drop_on + 1 :] = rest[1:]
            opened[drop_on + 1 :] = rest[:-1]
    spy_c = pd.Series(close, index=idx)
    spy_o = pd.Series(opened, index=idx)
    trend = pd.Series(True, index=idx)
    dip = pd.Series(False, index=idx)
    mix = pd.Series(False, index=idx)
    rf = pd.Series(0.0, index=idx)
    return idx, spy_o, spy_c, trend, dip, mix, rf


def _run(cut, mode, idx, spy_o, spy_c, trend, dip, mix, rf, cost=0.0):
    return run_book_breaker(
        spy_o, spy_c, spy_o, spy_c, spy_o, spy_c,
        trend, dip, mix, cut, mode, 0.5, 0.5, rf,
        spy_cost=cost, sso_cost=cost, upro_cost=cost,
        reentry_fraction=REENTRY_FRACTION,
    )


def test_grid_is_fixed_and_the_live_book_sorts_first():
    found = cells()
    assert CUTS == (0.15, 0.20, 0.25)
    assert MODES == ("confirm", "recover")
    assert REENTRY_FRACTION == 0.5
    assert OFF_NAME == "bk_off"
    assert found == [
        (0.15, "confirm"), (0.15, "recover"),
        (0.20, "confirm"), (0.20, "recover"),
        (0.25, "confirm"), (0.25, "recover"),
    ]
    names = [variant_name(*cell) for cell in found]
    assert names == [
        "zbk_c15_confirm", "zbk_c15_recover",
        "zbk_c20_confirm", "zbk_c20_recover",
        "zbk_c25_confirm", "zbk_c25_recover",
    ]
    assert OFF_NAME < min(names)
    with pytest.raises(ValueError):
        variant_name(0.10, "recover")
    with pytest.raises(ValueError):
        variant_name(0.15, "heal")
    for cut in CUTS:
        for mode in MODES:
            neighbors = adjacent_cells((cut, mode), [CUTS, MODES])
            assert neighbors
            assert (cut, mode) not in neighbors
    assert adjacent_cells((0.20, "recover"), [CUTS, MODES]) == [
        (0.15, "recover"), (0.25, "recover"), (0.20, "confirm"),
    ]


def test_state_machine_recover_and_confirm():
    tripped, seen = step_breaker(False, False, -0.15, True, 0.15, "recover", 0.075)
    assert tripped and not seen
    held, _seen = step_breaker(True, False, -0.10, True, 0.15, "recover", 0.075)
    assert held
    cleared, _seen = step_breaker(True, False, -0.075, True, 0.15, "recover", 0.075)
    assert not cleared
    still, _seen = step_breaker(True, False, -0.075, False, 0.15, "recover", 0.075)
    assert still

    tripped, seen = step_breaker(False, False, -0.20, True, 0.15, "confirm", 0.075)
    assert tripped and not seen
    same_spell, seen = step_breaker(tripped, seen, -0.01, True, 0.15, "confirm", 0.075)
    assert same_spell and not seen
    off, seen = step_breaker(same_spell, seen, -0.01, False, 0.15, "confirm", 0.075)
    assert off and seen
    fresh, seen = step_breaker(off, seen, -0.01, True, 0.15, "confirm", 0.075)
    assert not fresh and not seen
    with pytest.raises(ValueError):
        step_breaker(False, False, -0.2, True, 0.15, "pause", 0.075)
    with pytest.raises(ValueError):
        step_breaker(False, False, np.nan, True, 0.15, "recover", 0.075)


def test_a_cut_that_never_binds_matches_the_trend_core_book():
    idx, spy_o, spy_c, trend, dip, mix, rf = _paths(drop_on=None)
    cost = 0.0005
    plain = run_trend_core(
        spy_o, spy_c, spy_o, spy_c, spy_o, spy_c,
        trend, dip, mix, 0.5, 0.5, rf,
        spy_cost=cost, sso_cost=cost, upro_cost=cost, record_sleeve=True,
    )
    result, breaker, sleeve, drawdown = _run(0.99, "recover", idx, spy_o, spy_c, trend, dip, mix, rf, cost)
    assert not bool(breaker.any())
    pd.testing.assert_series_equal(result.returns, plain.returns, check_names=False)
    pd.testing.assert_series_equal(result.turnover, plain.turnover, check_names=False)
    live_sleeve = plain.sleeve_weight.notna().astype(bool)
    pd.testing.assert_series_equal(sleeve.astype(bool), live_sleeve, check_names=False)
    pd.testing.assert_series_equal(drawdown, book_drawdown(result.returns), check_names=False)


def test_the_trip_fills_next_session_and_not_the_same_close():
    idx, spy_o, spy_c, trend, dip, mix, rf = _paths(n=16, drop_on=8, drop_to=60.0, end_at=40.0)
    plain = run_trend_core(
        spy_o, spy_c, spy_o, spy_c, spy_o, spy_c,
        trend, dip, mix, 0.5, 0.5, rf,
    )
    result, breaker, sleeve, _drawdown = _run(0.15, "confirm", idx, spy_o, spy_c, trend, dip, mix, rf)
    assert result.returns.iloc[8] == pytest.approx(plain.returns.iloc[8])
    assert bool(breaker.iloc[7]) is False
    assert bool(breaker.iloc[8]) is True
    assert bool(sleeve.iloc[8]) is True
    assert bool(sleeve.iloc[9]) is False
    assert result.returns.iloc[9] != pytest.approx(plain.returns.iloc[9])


def test_confirm_stays_out_for_the_same_spell_and_recover_comes_back():
    idx, spy_o, spy_c, trend, dip, mix, rf = _paths(n=20, drop_on=6, drop_to=70.0, end_at=220.0)
    _confirm, confirm_on, confirm_sleeve, _dd = _run(0.15, "confirm", idx, spy_o, spy_c, trend, dip, mix, rf)
    _recover, recover_on, recover_sleeve, _dd2 = _run(0.15, "recover", idx, spy_o, spy_c, trend, dip, mix, rf)
    assert bool(confirm_on.iloc[-1]) is True
    assert bool(confirm_sleeve.iloc[-1]) is False
    assert bool(recover_on.iloc[-1]) is False
    assert bool(recover_sleeve.iloc[-1]) is True


def test_breaker_does_not_read_a_future_close():
    idx, spy_o, spy_c, trend, dip, mix, rf = _paths(n=18, drop_on=None)
    base, base_on, _sleeve, _dd = _run(0.15, "recover", idx, spy_o, spy_c, trend, dip, mix, rf)
    future_c = spy_c.copy()
    future_o = spy_o.copy()
    future_c.iloc[14:] = 40.0
    future_o.iloc[14:] = 40.0
    future_o.iloc[14] = 100.0
    alt, alt_on, _sleeve2, _dd2 = _run(0.15, "recover", idx, future_o, future_c, trend, dip, mix, rf)
    pd.testing.assert_series_equal(base.returns.iloc[:14], alt.returns.iloc[:14])
    pd.testing.assert_series_equal(base_on.iloc[:14].astype(bool), alt_on.iloc[:14].astype(bool))
    assert bool(alt_on.iloc[14]) is True


def _write_log(path, kept, pick, cut=0.15, mode="recover"):
    entry = {
        "id": "book",
        "kept": kept,
        "status": "tested",
        "overlay": "book_breaker",
        "walk_forward_picks": [["2018-01-02", pick]],
        "variants": [
            {"name": OFF_NAME},
            {"name": variant_name(cut, mode), "cut": cut, "reentry": mode},
        ],
    }
    path.write_text(json.dumps([entry]))


def test_a_rejected_breaker_does_not_reload(tmp_path):
    idx, spy_o, spy_c, _trend, _dip, _mix, rf = _paths(n=12, drop_on=8, drop_to=50.0, end_at=50.0)
    _write_log(tmp_path / "iterations.json", kept=False, pick="zbk_c15_recover")
    assert has_kept_book_breaker(tmp_path / "iterations.json") is False
    assert kept_book_breaker(
        tmp_path / "iterations.json", spy_o, spy_c, spy_o, spy_c, spy_o, spy_c, rf,
    ) == (None, None)
    _write_log(tmp_path / "iterations.json", kept=True, pick=OFF_NAME)
    assert has_kept_book_breaker(tmp_path / "iterations.json") is True
    assert kept_book_breaker(
        tmp_path / "iterations.json", spy_o, spy_c, spy_o, spy_c, spy_o, spy_c, rf, as_of=idx[-1],
    ) == (None, None)


def test_a_kept_breaker_uses_only_closes_through_as_of(tmp_path):
    idx, spy_o, spy_c, _trend, _dip, _mix, rf = _paths(n=12, drop_on=9, drop_to=50.0, end_at=50.0)
    folder = tmp_path
    _write_log(folder / "iterations.json", kept=True, pick="zbk_c15_recover")
    early, early_info = kept_book_breaker(
        folder / "iterations.json", spy_o, spy_c, spy_o, spy_c, spy_o, spy_c, rf, as_of=idx[8],
    )
    assert early_info["book_breaker_rule"] == "zbk_c15_recover"
    assert early_info["book_breaker_on"] is False
    assert float(early.iloc[-1]) == 1.0
    later, later_info = kept_book_breaker(
        folder / "iterations.json", spy_o, spy_c, spy_o, spy_c, spy_o, spy_c, rf, as_of=idx[-1],
    )
    assert later_info["book_breaker_on"] is True
    assert float(later.iloc[-1]) == 0.0
    crashed = spy_c.copy()
    crashed.iloc[-1] = 10.0
    again, _info = kept_book_breaker(
        folder / "iterations.json", spy_o, crashed, spy_o, crashed, spy_o, crashed, rf, as_of=idx[8],
    )
    pd.testing.assert_series_equal(early, again)


def test_signal_moves_only_the_sleeve_and_names_the_breaker():
    idx = pd.bdate_range("2018-01-02", periods=40)
    close = pd.Series(np.linspace(80, 140, len(idx)), index=idx)
    scale = pd.Series(1.0, index=idx)
    scale.iloc[-1] = 0.0
    info = {
        "book_breaker_rule": "zbk_c15_recover",
        "book_breaker_cut": 0.15,
        "book_breaker_reentry": "recover",
        "book_breaker_on": True,
        "book_drawdown": -0.16,
    }
    payload = build_trend_signal(
        TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
        apply_dip=False, sleeve_scale=scale, breaker_info=info,
    )
    assert payload["weights"]["SPY"] == pytest.approx(0.5)
    assert payload["weights"]["BIL"] == pytest.approx(0.5)
    assert "UPRO" not in payload["weights"]
    assert payload["diagnostics"]["book_breaker_on"] is True
    assert payload["trade_reason"] == "book_breaker"
    with pytest.raises(ValueError):
        build_trend_signal(
            TrendCoreSpec(window=20), close, as_of=idx[-1], generated_at="2026-01-01T00:00:00Z",
            apply_dip=False, sleeve_scale=scale, breaker_info=info, dd_info={"drawdown_rule": "dd_w63_c8"},
        )
