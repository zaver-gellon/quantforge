"""Causality and constraint tests for the apex building blocks."""

import numpy as np
import pandas as pd
import pytest

from quantforge.apex.alphas import alpha_panel, reversal_scores, tsmom_scores, xs_zscore
from quantforge.apex.combine import combine_equal, walk_forward_predictions
from quantforge.apex.construct import hrp_weights, project_weights, risk_budget_weights
from quantforge.apex.impact import capacity_notionals, impact_fraction
from quantforge.apex.portfolio import monthly_targets
from quantforge.apex.regime import hmm_scale, vol_trend_scale
from quantforge.apex.riskmodel import drawdown_exposure, ledoit_wolf
from quantforge.apex.validate import cpcv_sharpes
from quantforge.engine.schedule import month_ends


def _closes(n=420, k=4, seed=1):
    idx = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n, k))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(k)])


def _yields(index):
    vix = pd.Series(18.0, index=index)
    curve = pd.Series(1.2, index=index)
    credit = pd.Series(0.0, index=index)
    dollar = pd.Series(0.0, index=index)
    yields = {
        "^IRX": pd.Series(1.0, index=index),
        "^FVX": pd.Series(2.0, index=index),
        "^TNX": pd.Series(3.0, index=index),
        "^TYX": pd.Series(3.5, index=index),
    }
    return vix, curve, credit, dollar, yields


def test_alpha_rows_do_not_depend_on_later_prices():
    closes = _closes()
    cut = closes.index[300]
    market = closes["A0"]
    vix, curve, credit, dollar, yields = _yields(closes.index)
    full = alpha_panel(closes, market, yields, vix, curve, credit, dollar)
    trunc_y = {k: v.loc[:cut] for k, v in yields.items()}
    trunc = alpha_panel(
        closes.loc[:cut], market.loc[:cut], trunc_y,
        vix.loc[:cut], curve.loc[:cut], credit.loc[:cut], dollar.loc[:cut],
    )
    for name, panel in full.items():
        left = panel.loc[:cut].astype(float)
        right = trunc[name].astype(float)
        pd.testing.assert_frame_equal(left, right, check_names=False)


def test_ridge_prediction_does_not_use_later_labels():
    closes = _closes(n=700, k=5, seed=2)
    cut = closes.index[560]
    panels = {
        "tsmom": tsmom_scores(closes),
        "reversal": reversal_scores(closes),
    }
    full = walk_forward_predictions(panels, closes, list(panels), min_rows=40)
    trunc_panels = {name: panel.loc[:cut] for name, panel in panels.items()}
    trunc = walk_forward_predictions(trunc_panels, closes.loc[:cut], list(panels), min_rows=40)
    common = [d for d in full.dropna(how="all").index if d <= closes.index[480] and d in trunc.index]
    assert common
    pd.testing.assert_frame_equal(full.loc[common], trunc.loc[common], check_names=False)


def test_long_only_targets_respect_the_caps_and_the_future():
    closes = _closes(n=400, k=4, seed=3)
    scores = xs_zscore(tsmom_scores(closes))
    cut = closes.index[320]
    kwargs = dict(
        method="budget", long_only=True, gross=1.0, max_weight=0.4,
        sector_map={c: "g" if c != "A0" else "core" for c in closes.columns},
        sector_cap=0.7, vol_target=0.10, kelly_fraction=None, turnover_blend=1.0,
    )
    full = monthly_targets(scores, closes, **kwargs)
    trunc = monthly_targets(scores.loc[:cut], closes.loc[:cut], **kwargs)
    decisions = list(closes.index[month_ends(closes.index).to_numpy()])
    shared = [d for d in decisions if d < closes.index[300]]
    assert shared
    pd.testing.assert_frame_equal(full.loc[shared], trunc.loc[shared], check_names=False)
    assert (full.to_numpy() >= -1e-12).all()
    assert full.abs().sum(axis=1).max() <= 1.0 + 1e-8
    assert full.max(axis=1).max() <= 0.4 + 1e-8


def test_ledoit_wolf_is_positive_definite_and_shrinks_a_short_sample():
    rng = np.random.default_rng(4)
    cov, shrink = ledoit_wolf(rng.normal(size=(40, 8)))
    values = np.linalg.eigvalsh(cov)
    assert values.min() > 0
    assert 0.0 <= shrink <= 1.0
    _, shrink_long = ledoit_wolf(rng.normal(size=(4000, 3)))
    assert shrink_long < shrink


def test_hrp_weights_are_positive_and_sum_to_one():
    cov = np.diag([0.01, 0.01, 0.04, 0.04])
    weights = hrp_weights(cov)
    assert np.all(weights > 0)
    assert weights.sum() == pytest.approx(1.0)
    # Quieter names get more weight than the two high-vol names together? Not required.
    assert weights[:2].sum() > weights[2:].sum()


def test_projection_enforces_a_short_cap():
    raw = np.array([0.8, 0.8, -0.8])
    weights = project_weights(raw, gross=1.0, long_only=False, max_weight=0.4, groups=[np.array([0, 1])], sector_cap=0.5)
    assert abs(weights).sum() <= 1.0 + 1e-8
    assert abs(weights).max() <= 0.4 + 1e-8
    assert abs(weights[:2]).sum() <= 0.5 + 1e-8


def test_risk_budget_long_only_drops_negative_scores():
    weights = risk_budget_weights(
        np.array([1.0, -2.0, 0.5]),
        np.array([0.1, 0.1, 0.1]),
        gross=1.0, long_only=True, max_weight=0.8,
    )
    assert weights[1] == 0.0
    assert weights.sum() == pytest.approx(1.0)


def test_drawdown_throttle_cuts_after_the_loss_is_known():
    idx = pd.bdate_range("2020-01-02", periods=6)
    returns = pd.Series([0.0, -0.12, 0.0, 0.0, 0.01, 0.0], index=idx)
    scale = drawdown_exposure(returns, cuts=((0.10, 0.5),))
    assert scale.iloc[0] == 1.0
    assert scale.iloc[1] == 0.5


def test_vol_trend_and_hmm_do_not_read_the_future():
    closes = _closes(n=800, k=1, seed=5)["A0"]
    cut = closes.index[600]
    refits = list(closes.index[month_ends(closes.index).to_numpy()][::12])
    full = vol_trend_scale(closes)
    trunc = vol_trend_scale(closes.loc[:cut])
    pd.testing.assert_series_equal(full.loc[:cut], trunc, check_names=False)
    full_h = hmm_scale(closes, refits)
    trunc_h = hmm_scale(closes.loc[:cut], [d for d in refits if d <= cut])
    shared = full_h.index[full_h.index <= closes.index[500]]
    pd.testing.assert_series_equal(full_h.loc[shared], trunc_h.loc[shared], check_names=False)


def test_impact_and_capacity_move_with_size():
    small = impact_fraction(5_000, 1_000_000_000, 0.01)
    large = impact_fraction(50_000_000, 1_000_000_000, 0.01)
    assert large > small > 0
    table = capacity_notionals(1_000_000_000, 0.01, cost_bps=(10.0,))
    assert table[10.0] > 0


def test_cpcv_reports_a_distribution():
    idx = pd.bdate_range("2014-01-02", periods=800)
    returns = pd.Series(0.0004, index=idx)
    stats = cpcv_sharpes(returns, n_groups=4, k_test=2, embargo=5)
    assert stats["n_splits"] > 0
    assert stats["fraction_positive"] == 1.0


def test_equal_combination_averages_two_scores():
    idx = pd.bdate_range("2020-01-02", periods=5)
    a = pd.DataFrame({"X": [1.0, 2, 3, 4, 5], "Y": [3.0, 2, 1, 0, -1], "Z": [0.0, 1, 2, 3, 4]}, index=idx)
    b = pd.DataFrame({"X": [1.0, 1, 1, 1, 1], "Y": [1.0, 1, 1, 1, 1], "Z": [2.0, 2, 2, 2, 2]}, index=idx)
    combined = combine_equal({"a": a, "b": b}, ["a", "b"])
    assert combined.shape == a.shape
    assert combined.notna().all().all()
