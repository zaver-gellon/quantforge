"""Choose a variant using only returns strictly before each decision date."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardResult:
    returns: pd.Series
    picks: dict[str, str]
    scores: dict[str, dict[str, float]]


def walk_forward_select(
    variants: dict[str, pd.Series],
    decision_dates,
    score_fn,
    min_obs: int = 756,
    fallback: str | None = None,
    accept_fn=None,
) -> WalkForwardResult:
    """Hold, until the next decision date, the variant with the best trailing score.

    ``score_fn`` and ``accept_fn`` see ``returns[index < decision]`` only. A
    variant that fails ``accept_fn`` is not eligible at that date. Ties break
    alphabetically so the pick does not depend on dict order.
    """
    names = sorted(variants)
    decisions = list(pd.DatetimeIndex(pd.to_datetime(decision_dates)))
    picks: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    pieces: list[pd.Series] = []
    for i, decision in enumerate(decisions):
        nxt = decisions[i + 1] if i + 1 < len(decisions) else None
        scored: dict[str, float] = {}
        for name in names:
            history = variants[name].dropna()
            history = history[history.index < decision]
            if len(history) < min_obs:
                continue
            if accept_fn is not None and not accept_fn(history):
                continue
            scored[name] = float(score_fn(history))
        label = str(pd.Timestamp(decision).date())
        scores[label] = scored
        if not scored:
            if fallback is None:
                raise ValueError(f"no variant has {min_obs} observations before {label}")
            pick = fallback
        else:
            best = max(scored.values())
            pick = min(name for name, value in scored.items() if value == best)
        picks[label] = pick
        chosen = variants[pick].dropna()
        if nxt is None:
            piece = chosen[chosen.index >= decision]
        else:
            piece = chosen[(chosen.index >= decision) & (chosen.index < nxt)]
        pieces.append(piece)
    if not pieces:
        combined = pd.Series(dtype=float)
    else:
        combined = pd.concat(pieces)
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return WalkForwardResult(returns=combined, picks=picks, scores=scores)
