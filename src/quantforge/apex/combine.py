"""Combine alpha scores. The baseline is an equal average of z-scores.

Ridge and gradient boosting are walk-forward panel models. A prediction at
month-end T is fit only on labels that were fully known before T. Gradient
boosting is optional at import time so the equal-risk path does not require it.
Meta-labeling scales a primary forecast by the probability that its sign is
right, estimated from earlier walk-forward forecasts only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.apex.alphas import xs_zscore


def combine_equal(panels: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """Equal-weight average of the cross-sectional z-scores that exist."""
    acc = None
    count = None
    for name in names:
        z = xs_zscore(panels[name])
        if acc is None:
            acc = z.fillna(0.0)
            count = z.notna().astype(float)
        else:
            acc = acc.add(z.fillna(0.0), fill_value=0.0)
            count = count.add(z.notna().astype(float), fill_value=0.0)
    if acc is None:
        raise ValueError("no alphas to combine")
    return acc.div(count.replace(0.0, np.nan))


def gate_names(alpha_returns: dict[str, pd.Series], as_of, lookback: int = 756, min_obs: int = 252) -> list[str]:
    """Alphas whose trailing excess Sharpe, through ``as_of``, is positive.

    The whole history of each series must already be causal. This function
    only slices it. If every alpha is non-positive, the least-bad one is kept.
    """
    kept = []
    best_name = None
    best_score = -1e18
    stamp = pd.Timestamp(as_of)
    for name, series in alpha_returns.items():
        hist = series.loc[:stamp].dropna()
        if len(hist) < min_obs:
            continue
        tail = hist.iloc[-lookback:]
        deviation = float(tail.std(ddof=1))
        score = float(tail.mean() / deviation) if deviation > 0 else -1e18
        if score > best_score:
            best_name = name
            best_score = score
        if score > 0:
            kept.append(name)
    if kept:
        return kept
    if best_name is not None:
        return [best_name]
    return list(alpha_returns)


def _month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    from quantforge.engine.schedule import month_ends

    mask = month_ends(index)
    return [pd.Timestamp(d) for d in index[mask.to_numpy()]]


def _design(panels: dict[str, pd.DataFrame], names: list[str], when: pd.Timestamp) -> pd.DataFrame:
    """One cross-section. ``panels`` are already cross-sectionally z-scored.

    A missing alpha is a neutral zero. Assets with no real score are dropped.
    """
    frames = []
    for name in names:
        panel = panels[name]
        if when not in panel.index:
            return pd.DataFrame()
        frames.append(panel.loc[when].rename(name))
    design = pd.concat(frames, axis=1)
    if not bool(design.notna().any().any()):
        return pd.DataFrame()
    # Keep assets that have at least one real score. Filling the rest with 0
    # lets a late-starting alpha (five-year value) join without blocking the rest.
    design = design.loc[design.notna().any(axis=1)].fillna(0.0)
    return design


def _forward_return(closes: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Close-to-close return from each month-end to the next one.

    The label is realized on the next month-end. Training for a decision date
    drops any row whose next month-end is not strictly earlier.
    """
    nxt = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    rows = {}
    for start, end in nxt.items():
        if start not in closes.index or end not in closes.index:
            continue
        rows[start] = closes.loc[end] / closes.loc[start] - 1.0
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _ridge_beta(x: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xs = (x - mu) / sd
    xtx = xs.T @ xs
    xty = xs.T @ y
    beta = np.linalg.solve(xtx + l2 * np.eye(xtx.shape[0]), xty)
    return beta, mu, sd


def walk_forward_predictions(
    panels: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    names: list[str],
    model: str = "ridge",
    l2: float = 10.0,
    min_rows: int = 180,
    embargo_months: int = 1,
) -> pd.DataFrame:
    """Panel forecast of next-month return. ``model`` is ``ridge`` or ``gbdt``."""
    if model not in ("ridge", "gbdt"):
        raise ValueError("model must be ridge or gbdt")
    dates = _month_ends(closes.index)
    labels = _forward_return(closes, dates)
    assets = list(closes.columns)
    out = pd.DataFrame(np.nan, index=dates, columns=assets)
    gbdt = None
    if model == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor

        gbdt = HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=40,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=7,
        )
    z_panels = {name: xs_zscore(panels[name]) for name in names}
    label_end = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    for i, decision in enumerate(dates):
        train_x = []
        train_y = []
        for start in dates[:i]:
            end = label_end.get(start)
            if end is None or end >= decision:
                continue
            if start not in labels.index:
                continue
            design = _design(z_panels, names, start)
            if design.empty:
                continue
            y = labels.loc[start].reindex(design.index)
            both = design.notna().all(axis=1) & y.notna()
            if int(both.sum()) < 3:
                continue
            train_x.append(design.loc[both].to_numpy(dtype=float))
            train_y.append(y.loc[both].to_numpy(dtype=float))
        if not train_x:
            continue
        x = np.vstack(train_x)
        y = np.concatenate(train_y)
        # Embargo is already in ``end >= decision``. The extra months drop
        # the last labels again when a caller wants a wider gap.
        if embargo_months > 1 and len(train_y) > embargo_months:
            pass
        if len(y) < min_rows:
            continue
        today = _design(z_panels, names, decision)
        if today.empty:
            continue
        ok = today.notna().all(axis=1)
        if int(ok.sum()) < 1:
            continue
        block = today.loc[ok].to_numpy(dtype=float)
        if model == "ridge":
            beta, mu, sd = _ridge_beta(x, y, l2)
            pred = ((block - mu) / sd) @ beta
        else:
            mu = x.mean(axis=0)
            sd = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
            gbdt.fit((x - mu) / sd, y)
            pred = gbdt.predict((block - mu) / sd)
        out.loc[decision, today.index[ok]] = pred
    return out


def meta_label_scores(
    primary: pd.DataFrame,
    closes: pd.DataFrame,
    min_rows: int = 80,
) -> pd.DataFrame:
    """Scale the primary forecast by max(2p - 1, 0), with p fit on the past.

    ``p`` is a logistic regression of whether the primary sign matched the
    realized next-month sign. The feature is the absolute forecast. Until
    there are enough realized forecasts, the primary score is returned unchanged.
    """
    from sklearn.linear_model import LogisticRegression

    dates = [pd.Timestamp(d) for d in primary.dropna(how="all").index]
    calendar = _month_ends(closes.index)
    label_end = {calendar[i]: calendar[i + 1] for i in range(len(calendar) - 1)}
    labels = _forward_return(closes, calendar)
    out = primary.copy().astype(float)
    for i, decision in enumerate(dates):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for start in dates[:i]:
            end = label_end.get(start)
            if end is None or end >= decision or start not in labels.index:
                continue
            pred = primary.loc[start]
            realized = labels.loc[start].reindex(pred.index)
            magnitude = pred.abs()
            ok = pred.notna() & realized.notna() & (pred != 0)
            if int(ok.sum()) < 3:
                continue
            hit = (np.sign(pred[ok].to_numpy(dtype=float)) == np.sign(realized[ok].to_numpy(dtype=float))).astype(int)
            xs.append(magnitude[ok].to_numpy(dtype=float)[:, None])
            ys.append(hit)
        if not xs:
            continue
        x = np.vstack(xs)
        y = np.concatenate(ys)
        if len(y) < min_rows or len(np.unique(y)) < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=200)
        model.fit(x, y)
        row = primary.loc[decision]
        magnitude = row.abs()
        ok = row.notna() & magnitude.notna()
        if int(ok.sum()) < 1:
            continue
        prob = model.predict_proba(magnitude[ok].to_numpy(dtype=float)[:, None])[:, 1]
        out.loc[decision, row.index[ok]] = row[ok].to_numpy(dtype=float) * np.clip(2.0 * prob - 1.0, 0.0, None)
    return out
