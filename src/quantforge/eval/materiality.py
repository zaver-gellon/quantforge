"""Out-of-sample bar for one iteration. Fixed before the candidate is scored.

A walk-forward candidate replaces the current book only when every clause below
is true on the same out-of-sample sessions:

(a) Excess Sharpe improves by at least 0.05, or maximum drawdown improves by at
    least 3 percentage points and CAGR is no more than 0.5 percentage points worse.
(b) The deflated Sharpe of the daily improvement (candidate minus current), using
    the ledger's trial count and daily-Sharpe dispersion, is at least 0.75.
    That is the precommitted meaning of "meaningfully positive": the probability
    that the improvement's true Sharpe clears the multiple-testing benchmark is
    at least three-quarters. The same clause of (a) that passed must still pass
    after the calendar year with the largest summed daily improvement is removed.
    Those sessions are dropped and the remaining sessions are concatenated.
(c) The parameter cell selected on the last decision date is not an isolated
    peak. Every grid neighbor (one step in one axis) has an out-of-sample Sharpe
    within 0.05 and a maximum drawdown within 3 percentage points of that cell.
    A cell with no neighbor fails this clause.

Sharpe ratios are the excess-Sharpe figures from ``summarize`` (T-bill subtracted
when a yield is supplied). Drawdown improvement is the candidate's maximum
drawdown minus the current book's, so a less severe drawdown is positive.
The improvement series is not excess of T-bills again: the yield cancels in the difference.
"""

from __future__ import annotations

import math
import statistics

import pandas as pd

from quantforge.eval.metrics import summarize
from quantforge.eval.sharpe import deflated_sharpe, expected_maximum_sharpe

SHARPE_IMPROVEMENT = 0.05
DRAWDOWN_IMPROVEMENT = 0.03
CAGR_SLACK = 0.005
IMPROVEMENT_DSR_MIN = 0.75
NEIGHBOR_SHARPE_TOL = 0.05
NEIGHBOR_DRAWDOWN_TOL = 0.03


def bar_constants() -> dict:
    return {
        "sharpe_improvement": SHARPE_IMPROVEMENT,
        "drawdown_improvement": DRAWDOWN_IMPROVEMENT,
        "cagr_slack": CAGR_SLACK,
        "improvement_dsr_min": IMPROVEMENT_DSR_MIN,
        "neighbor_sharpe_tol": NEIGHBOR_SHARPE_TOL,
        "neighbor_drawdown_tol": NEIGHBOR_DRAWDOWN_TOL,
    }


def adjacent_cells(cell: tuple, axes: list[tuple]) -> list[tuple]:
    """Cells one step away in exactly one axis. ``axes`` are ordered values."""
    if len(cell) != len(axes):
        raise ValueError("cell and axes must have the same length")
    found: list[tuple] = []
    for dim, values in enumerate(axes):
        ordered = tuple(values)
        if cell[dim] not in ordered:
            raise ValueError(f"{cell[dim]!r} is not on axis {dim}")
        index = ordered.index(cell[dim])
        for step in (index - 1, index + 1):
            if 0 <= step < len(ordered):
                nxt = list(cell)
                nxt[dim] = ordered[step]
                found.append(tuple(nxt))
    return found


def _finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("metric is not finite")
    return number


def _metrics(returns: pd.Series, rf: pd.Series | None) -> dict:
    window_rf = None if rf is None else pd.Series(rf).reindex(returns.index).astype(float).fillna(0.0)
    metrics = summarize(returns, rf=window_rf)
    return {
        "cagr": _finite(metrics["cagr"]),
        "vol": _finite(metrics["vol"]),
        "sharpe": _finite(metrics["sharpe"]),
        "sharpe_per_period": _finite(metrics["sharpe_per_period"]),
        "max_drawdown": _finite(metrics["max_drawdown"]),
        "n_days": int(metrics["n_days"]),
        "start": metrics["start"],
        "end": metrics["end"],
    }


def criterion_a(candidate: dict, baseline: dict) -> dict:
    """Clause (a). CAGR slack binds only the drawdown branch."""
    sharpe_delta = _finite(candidate["sharpe"]) - _finite(baseline["sharpe"])
    cagr_delta = _finite(candidate["cagr"]) - _finite(baseline["cagr"])
    drawdown_delta = _finite(candidate["max_drawdown"]) - _finite(baseline["max_drawdown"])
    sharpe_pass = sharpe_delta >= SHARPE_IMPROVEMENT - 1e-12
    drawdown_pass = (
        drawdown_delta >= DRAWDOWN_IMPROVEMENT - 1e-12 and cagr_delta >= -CAGR_SLACK - 1e-12
    )
    return {
        "sharpe_delta": sharpe_delta,
        "cagr_delta": cagr_delta,
        "max_drawdown_delta": drawdown_delta,
        "sharpe_pass": bool(sharpe_pass),
        "drawdown_pass": bool(drawdown_pass),
        "pass": bool(sharpe_pass or drawdown_pass),
    }


def _still_helps(original: dict, reduced: dict) -> bool:
    """The clause of (a) that passed must still pass with the best year removed."""
    if original["sharpe_pass"] and original["drawdown_pass"]:
        return bool(reduced["pass"])
    if original["sharpe_pass"]:
        return bool(reduced["sharpe_pass"])
    if original["drawdown_pass"]:
        return bool(reduced["drawdown_pass"])
    return False


def _align(series: pd.Series, index: pd.Index) -> pd.Series:
    aligned = pd.Series(series).reindex(index).astype(float)
    if aligned.isna().any():
        raise ValueError("a path is missing out-of-sample sessions")
    return aligned


def _similar(cell: dict, other: dict) -> bool:
    sharpe_gap = abs(_finite(cell["sharpe"]) - _finite(other["sharpe"]))
    drawdown_gap = abs(_finite(cell["max_drawdown"]) - _finite(other["max_drawdown"]))
    return sharpe_gap <= NEIGHBOR_SHARPE_TOL + 1e-12 and drawdown_gap <= NEIGHBOR_DRAWDOWN_TOL + 1e-12


def evaluate_materiality(
    candidate: pd.Series,
    baseline: pd.Series,
    neighbor_paths: dict[str, pd.Series] | None,
    cell_path: pd.Series | None,
    n_trials: int,
    sharpe_dispersion: float,
    rf: pd.Series | None = None,
) -> dict:
    """Score one walk-forward candidate. The return value is the decision record.

    ``candidate`` and ``baseline`` are already restricted to the out-of-sample
    window and share its index. ``cell_path`` is the frozen parameter cell
    chosen on the last decision date, or None when that pick is the current book.
    ``neighbor_paths`` are frozen neighbor paths; they are sliced to the window.
    """
    candidate = pd.Series(candidate).dropna().astype(float).sort_index()
    baseline = _align(baseline, candidate.index)
    if int(n_trials) < 1:
        raise ValueError("n_trials must be positive")
    dispersion = float(sharpe_dispersion)
    if not math.isfinite(dispersion) or dispersion < 0.0:
        raise ValueError("sharpe dispersion must be finite and non-negative")

    current = _metrics(baseline, rf)
    proposed = _metrics(candidate, rf)
    clause_a = criterion_a(proposed, current)

    improvement = candidate - baseline
    improvement_metrics = _metrics(improvement, None)
    improvement_dsr = float(deflated_sharpe(improvement, int(n_trials), dispersion, rf=None))
    sr0 = float(expected_maximum_sharpe(int(n_trials), dispersion))
    dsr_pass = bool(math.isfinite(improvement_dsr) and improvement_dsr >= IMPROVEMENT_DSR_MIN - 1e-12)

    yearly = improvement.groupby(improvement.index.year).sum()
    best_year = int(yearly.idxmax())
    kept = improvement.index.year != best_year
    reduced_a = criterion_a(_metrics(candidate.loc[kept], rf), _metrics(baseline.loc[kept], rf))
    still = _still_helps(clause_a, reduced_a)
    clause_b_pass = bool(dsr_pass and still)

    neighbors_out = []
    if cell_path is None:
        clause_c_pass = False
        clause_c_detail = "The last walk-forward pick is the current book, which has no parameter neighbors."
        cell_metrics = None
    elif not neighbor_paths:
        clause_c_pass = False
        clause_c_detail = "The selected cell has no grid neighbor."
        cell_metrics = _metrics(_align(cell_path, candidate.index), rf)
    else:
        cell_metrics = _metrics(_align(cell_path, candidate.index), rf)
        similar = True
        for name, path in sorted(neighbor_paths.items()):
            metrics = _metrics(_align(path, candidate.index), rf)
            ok = _similar(cell_metrics, metrics)
            similar = similar and ok
            neighbors_out.append({
                "name": name,
                "similar": bool(ok),
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_gap": abs(cell_metrics["sharpe"] - metrics["sharpe"]),
                "max_drawdown_gap": abs(cell_metrics["max_drawdown"] - metrics["max_drawdown"]),
            })
        clause_c_pass = bool(similar and len(neighbors_out) >= 1)
        if clause_c_pass:
            clause_c_detail = (
                f"All {len(neighbors_out)} neighbors are within {NEIGHBOR_SHARPE_TOL:.2f} Sharpe "
                f"and {NEIGHBOR_DRAWDOWN_TOL:.0%} drawdown of the selected cell."
            )
        else:
            clause_c_detail = (
                "At least one neighbor is more than "
                f"{NEIGHBOR_SHARPE_TOL:.2f} Sharpe or {NEIGHBOR_DRAWDOWN_TOL:.0%} drawdown "
                "away from the selected cell."
            )

    passed = bool(clause_a["pass"] and clause_b_pass and clause_c_pass)
    checks = [
        {
            "name": "a Sharpe +0.05",
            "pass": clause_a["sharpe_pass"],
            "detail": (
                f"Sharpe {current['sharpe']:.6f} -> {proposed['sharpe']:.6f} "
                f"(delta {clause_a['sharpe_delta']:.6f}; need >= {SHARPE_IMPROVEMENT:.2f})."
            ),
        },
        {
            "name": "a drawdown +3pp and CAGR within 0.5pp",
            "pass": clause_a["drawdown_pass"],
            "detail": (
                f"Max drawdown {current['max_drawdown']:.6f} -> {proposed['max_drawdown']:.6f} "
                f"(delta {clause_a['max_drawdown_delta']:.6f}; need >= {DRAWDOWN_IMPROVEMENT:.2f}). "
                f"CAGR {current['cagr']:.6f} -> {proposed['cagr']:.6f} "
                f"(delta {clause_a['cagr_delta']:.6f}; floor {-CAGR_SLACK:.4f})."
            ),
        },
        {
            "name": "a overall",
            "pass": clause_a["pass"],
            "detail": "Either the Sharpe clause or the drawdown clause is enough.",
        },
        {
            "name": "b deflated Sharpe of the improvement >= 0.75",
            "pass": dsr_pass,
            "detail": (
                f"Improvement Sharpe {improvement_metrics['sharpe']:.6f}, "
                f"deflated Sharpe {improvement_dsr:.6f} "
                f"(need >= {IMPROVEMENT_DSR_MIN:.2f}). "
                f"Trials {int(n_trials)}, daily-Sharpe dispersion {dispersion:.6f}, "
                f"daily multiple-testing benchmark {sr0:.6f}."
            ),
        },
        {
            "name": "b still helps with the best improvement year removed",
            "pass": still,
            "detail": (
                f"Removed {best_year} "
                f"(summed daily improvement {float(yearly.loc[best_year]):.6f}). "
                f"Remaining Sharpe delta {reduced_a['sharpe_delta']:.6f}, "
                f"CAGR delta {reduced_a['cagr_delta']:.6f}, "
                f"max-drawdown delta {reduced_a['max_drawdown_delta']:.6f}."
            ),
        },
        {
            "name": "b overall",
            "pass": clause_b_pass,
            "detail": "The deflated-Sharpe cutoff and the best-year check both have to pass.",
        },
        {
            "name": "c neighboring parameters",
            "pass": clause_c_pass,
            "detail": clause_c_detail,
        },
    ]
    return {
        "pass": passed,
        "bar": bar_constants(),
        "checks": checks,
        "oos": {"current": current, "candidate": proposed},
        "criterion_a": clause_a,
        "criterion_b": {
            "pass": clause_b_pass,
            "dsr_pass": dsr_pass,
            "still_helps": bool(still),
            "improvement_sharpe": improvement_metrics["sharpe"],
            "improvement_sharpe_per_period": improvement_metrics["sharpe_per_period"],
            "improvement_dsr": improvement_dsr,
            "sr0_daily": sr0,
            "n_trials": int(n_trials),
            "sharpe_dispersion_daily": dispersion,
            "best_year": best_year,
            "best_year_improvement_sum": float(yearly.loc[best_year]),
            "without_best_year": reduced_a,
        },
        "criterion_c": {
            "pass": clause_c_pass,
            "detail": clause_c_detail,
            "cell": cell_metrics,
            "neighbors": neighbors_out,
        },
    }
