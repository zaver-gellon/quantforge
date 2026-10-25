import numpy as np
import pandas as pd
import pytest

from quantforge.eval.bootstrap import block_bootstrap
from quantforge.eval.ledger import TrialLedger
from quantforge.eval.metrics import path_cagr, path_max_drawdown, summarize
from quantforge.eval.sharpe import deflated_sharpe, expected_max_z, probabilistic_sharpe
from quantforge.eval.walkforward import walk_forward_select


def _returns(values, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx)


def test_summarize_known_path():
    series = _returns(np.full(252, 0.01))
    stats = summarize(series, rf=pd.Series(0.0, index=series.index))
    assert stats["n_days"] == 252
    assert stats["cagr"] == pytest.approx(1.01 ** 252 - 1)
    assert stats["max_drawdown"] == pytest.approx(0.0)
    assert stats["vol"] == pytest.approx(0.0, abs=1e-12)


def test_probabilistic_sharpe_rises_with_drift():
    rng = np.random.default_rng(0)
    noise_x = rng.normal(0.0, 0.01, 2000)
    noise_x -= noise_x.mean()
    drift_x = rng.normal(0.001, 0.01, 2000)
    noise = _returns(noise_x)
    drift = _returns(drift_x)
    assert probabilistic_sharpe(noise) == pytest.approx(0.5, abs=0.05)
    assert probabilistic_sharpe(drift) > 0.9


def test_deflation_lowers_the_probability_and_the_bar_rises_with_n():
    rng = np.random.default_rng(1)
    series = _returns(rng.normal(0.0005, 0.01, 1500))
    raw = probabilistic_sharpe(series)
    haircut = deflated_sharpe(series, n_trials=50, sharpe_std=0.02)
    assert haircut < raw
    assert expected_max_z(100) > expected_max_z(10)


def test_walk_forward_does_not_see_the_future():
    idx = pd.bdate_range("2016-01-04", "2021-12-31")
    calm = pd.Series(0.0004, index=idx)
    # Amazing only from 2020 onward. It must not be selected for 2019.
    late = pd.Series(np.where(idx >= "2020-01-01", 0.01, -0.0002), index=idx)
    decisions = [pd.Timestamp(f"{year}-01-02") for year in range(2019, 2022)]
    # Snap decisions onto the index.
    decisions = [idx[idx >= day][0] for day in decisions]
    result = walk_forward_select(
        {"calm": calm, "late": late},
        decisions,
        score_fn=lambda s: float(s.mean()),
        min_obs=252,
    )
    assert result.picks[str(decisions[0].date())] == "calm"
    year_2020 = result.returns[result.returns.index.year == 2020]
    # 2020 is after the January 2020 decision, which can see 2019 and earlier only.
    assert result.picks[str(decisions[1].date())] == "calm"
    assert year_2020.mean() == pytest.approx(0.0004)
    assert result.picks[str(decisions[2].date())] == "late"


def test_drawdown_cap_rejects_a_past_crash_even_when_the_future_is_better():
    idx = pd.bdate_range("2016-01-04", "2021-12-31")
    calm = pd.Series(0.0003, index=idx)
    boom = pd.Series(0.001, index=idx)
    crash = (idx >= "2018-03-01") & (idx < "2018-04-01")
    boom.loc[crash] = -0.02
    boom.loc[idx >= "2020-01-01"] = 0.02
    decisions = [idx[idx >= pd.Timestamp(f"{year}-01-02")][0] for year in (2020, 2021)]
    past = boom[boom.index < decisions[0]]
    assert path_cagr(past) > path_cagr(calm[calm.index < decisions[0]])
    assert path_max_drawdown(past) < -0.30

    def accept(history: pd.Series) -> bool:
        return path_max_drawdown(history) >= -0.30

    result = walk_forward_select(
        {"calm": calm, "boom": boom},
        decisions,
        score_fn=path_cagr,
        accept_fn=accept,
        min_obs=252,
    )
    assert result.picks[str(decisions[0].date())] == "calm"
    assert result.returns[result.returns.index.year == 2020].mean() == pytest.approx(0.0003)


def test_bootstrap_preserves_a_constant_return_and_uses_the_seed():
    series = _returns(np.full(300, 0.001))
    bench = _returns(np.full(300, 0.0))
    out = block_bootstrap(series, horizon=20, n_paths=200, block=5, seed=7, start_value=5000, benchmark=bench)
    assert out["median"] == pytest.approx(5000 * (1.001 ** 20))
    assert out["p_loss"] == 0.0
    assert out["p_beats_benchmark"] == 1.0
    again = block_bootstrap(series, horizon=20, n_paths=200, block=5, seed=7, start_value=5000)
    assert again["median"] == out["median"]


def test_ledger_counts_a_hypothesis_once(tmp_path):
    path = tmp_path / "ledger.json"
    book = TrialLedger(path)
    params = {"window": 200, "band": 0.02}
    book.record(
        study="demo", family="trend", name="a", params=params,
        sample_start="2000-01-03", sample_end="2020-01-01",
        metrics={"sharpe_per_period": 0.02, "sharpe": 0.3},
    )
    book.record(
        study="demo", family="trend", name="a", params=params,
        sample_start="2000-01-03", sample_end="2024-01-01",
        metrics={"sharpe_per_period": 0.03, "sharpe": 0.4},
    )
    book.record(
        study="demo", family="trend", name="b", params={"window": 100},
        sample_start="2000-01-03", sample_end="2024-01-01",
        metrics={"sharpe_per_period": 0.01, "sharpe": 0.1},
    )
    book.record(
        study="demo", family="audit", name="a_cost", params=params,
        sample_start="2000-01-03", sample_end="2024-01-01",
        metrics={"sharpe_per_period": 0.02},
        counts_for_deflation=False,
    )
    book.save()
    reloaded = TrialLedger(path)
    assert reloaded.n_trials() == 2
    assert len(reloaded.trials) == 3
    kept = next(trial for trial in reloaded.counting() if trial["name"] == "a")
    assert kept["sample_end"] == "2024-01-01"
    assert kept["metrics"]["sharpe"] == 0.4
    assert reloaded.sharpe_dispersion() == pytest.approx(float(np.std([0.03, 0.01], ddof=1)))
