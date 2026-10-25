"""Standalone alpha scores.

Every score at date t uses prices and yields through t only. The backtest
fills the resulting weights at the next open. Missing inputs stay missing.
They are not filled with a zero that would look like a view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.apex.universe import CARRY_YIELD, COMMODITY_ETFS, CREDIT_ETFS, DURATION_ETFS, EQUITY_ETFS, PAIRS


def xs_zscore(scores: pd.DataFrame, min_names: int = 3) -> pd.DataFrame:
    """Cross-sectional z-score. Rows with too few names are left missing."""
    frame = scores.astype(float)
    count = frame.notna().sum(axis=1)
    mean = frame.mean(axis=1)
    # Population scale: a row of identical scores has scale 0 and stays missing.
    centered = frame.sub(mean, axis=0)
    var = (centered ** 2).mean(axis=1)
    scale = np.sqrt(var.where(var > 0))
    out = centered.div(scale, axis=0)
    return out.where(count >= min_names)


def _panel_like(closes: pd.DataFrame, value: pd.Series | float | None = None) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return pd.DataFrame({col: value for col in closes.columns})
    return pd.DataFrame(np.nan if value is None else value, index=closes.index, columns=closes.columns)


def tsmom_scores(
    closes: pd.DataFrame,
    horizons: tuple[int, ...] = (21, 63, 126, 252),
    vol_window: int = 63,
) -> pd.DataFrame:
    """Average of horizon returns scaled by trailing volatility.

    Scaling by volatility makes a quiet bond and a noisy equity comparable
    before the cross-sectional step. The sign is the trend; the magnitude
    shrinks when the move is small relative to noise.
    """
    returns = closes.pct_change()
    vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    parts = []
    for horizon in horizons:
        move = closes / closes.shift(horizon) - 1.0
        scale = vol * np.sqrt(horizon)
        parts.append(move / scale.replace(0.0, np.nan))
    return sum(parts) / len(parts)


def xs_momentum_scores(closes: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 style total return. The last ``skip`` sessions are left out."""
    return closes.shift(skip) / closes.shift(lookback) - 1.0


def residual_momentum_scores(
    closes: pd.DataFrame,
    market: pd.Series,
    lookback: int = 252,
    skip: int = 21,
    beta_window: int = 126,
) -> pd.DataFrame:
    """Cumulative residual return versus the market, skipping the last month.

    Beta is the trailing covariance with ``market``. The residual on day t
    uses that beta and the same day's market return, both known at the close.
    """
    asset = closes.pct_change()
    bench = market.astype(float).pct_change()
    aligned = bench.reindex(asset.index)
    cov = asset.rolling(beta_window, min_periods=beta_window).cov(aligned)
    var = aligned.rolling(beta_window, min_periods=beta_window).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    residual = asset.sub(beta.mul(aligned, axis=0), axis=0)
    residual = residual.where(beta.notna())
    cumulative = residual.fillna(0.0).cumsum()
    cumulative = cumulative.where(residual.notna().cumsum() > 0)
    return cumulative.shift(skip) - cumulative.shift(lookback)


def reversal_scores(closes: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Short-horizon reversal: the negative of the trailing return."""
    return -(closes / closes.shift(window) - 1.0)


def low_vol_scores(closes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    vol = closes.pct_change().rolling(window, min_periods=window).std(ddof=1)
    return -vol


def bab_scores(closes: pd.DataFrame, market: pd.Series, window: int = 126) -> pd.DataFrame:
    """Betting against beta: minus the trailing beta to the market."""
    asset = closes.pct_change()
    bench = market.astype(float).pct_change().reindex(asset.index)
    cov = asset.rolling(window, min_periods=window).cov(bench)
    var = bench.rolling(window, min_periods=window).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    return -beta


def value_scores(closes: pd.DataFrame, window: int = 1260, skip: int = 21) -> pd.DataFrame:
    """Price-only value proxy: minus the skipped long-horizon return.

    Five years is the default. Names without that history stay missing until
    they have it. This is not a fundamental value score.
    """
    return -(closes.shift(skip) / closes.shift(window) - 1.0)


def carry_scores(closes: pd.DataFrame, yields: dict[str, pd.Series]) -> pd.DataFrame:
    """Map a lagged treasury yield onto the duration ETF that tracks it.

    The yield is lagged one session by the caller. Credit and equity names are
    left missing: the adjusted cache does not keep a point-in-time dividend
    yield, and commodity futures curves are not in the free daily ETF bars.
    """
    out = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for asset, symbol in CARRY_YIELD.items():
        if asset not in out.columns or symbol not in yields:
            continue
        series = yields[symbol].reindex(closes.index).astype(float)
        out[asset] = series
    if "TLT" in out.columns and out["TLT"].isna().all() and "^TNX" in yields:
        out["TLT"] = yields["^TNX"].reindex(closes.index).astype(float)
    return out


def pair_scores(closes: pd.DataFrame, pairs=PAIRS, window: int = 252) -> pd.DataFrame:
    """Mean-reversion score from a rolling log-price hedge, estimated through t-1.

    The hedge ratio and the spread's mean and scale use the window ending
    yesterday. Today's spread is the new observation. A rich leg (positive
    residual) gets a negative score.
    """
    scores = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    seen = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for left, right in pairs:
        if left not in closes.columns or right not in closes.columns:
            continue
        log_l = np.log(closes[left].astype(float))
        log_r = np.log(closes[right].astype(float))
        var = log_r.rolling(window, min_periods=window).var()
        cov = log_l.rolling(window, min_periods=window).cov(log_r)
        beta = (cov / var.replace(0.0, np.nan)).shift(1)
        alpha = (log_l.rolling(window, min_periods=window).mean() - (cov / var.replace(0.0, np.nan)) * log_r.rolling(window, min_periods=window).mean()).shift(1)
        spread = log_l - alpha - beta * log_r
        past = spread.shift(1)
        mean = past.rolling(window, min_periods=window).mean()
        scale = past.rolling(window, min_periods=window).std(ddof=1)
        z = (spread - mean) / scale.replace(0.0, np.nan)
        usable = z.notna()
        scores[left] = scores[left] - z.fillna(0.0)
        scores[right] = scores[right] + (z * beta).fillna(0.0)
        seen[left] = seen[left] + usable.astype(float)
        seen[right] = seen[right] + usable.astype(float)
    return scores.div(seen.replace(0.0, np.nan))


def vrp_scores(closes: pd.DataFrame, market: pd.Series, vix: pd.Series, window: int = 21) -> pd.DataFrame:
    """Volatility risk premium: implied vol minus realized vol of the market.

    A positive premium scores equities up and duration down. The premium is an
    expanding z-score so a permanent level difference is not a permanent bet.
    """
    realized = market.astype(float).pct_change().rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252.0)
    premium = vix.reindex(market.index).astype(float) / 100.0 - realized
    mean = premium.expanding(min_periods=252).mean()
    scale = premium.expanding(min_periods=252).std(ddof=1)
    z = (premium - mean) / scale.replace(0.0, np.nan)
    out = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    aligned = z.reindex(closes.index)
    for column in closes.columns:
        if column in EQUITY_ETFS:
            out[column] = aligned
        elif column in DURATION_ETFS:
            out[column] = -aligned
    return out


def macro_scores(
    closes: pd.DataFrame,
    curve: pd.Series,
    credit: pd.Series,
    dollar: pd.Series,
) -> pd.DataFrame:
    """Cross-asset tilt from the curve, credit relative strength, and the dollar.

    ``curve`` is a yield spread in percent (10-year minus bill), lagged by the
    caller. ``credit`` is the trailing relative return of high yield versus
    investment grade. ``dollar`` is the trailing return of the dollar proxy.
    Each leg is an expanding z-score. Equities and commodities take the
    risk-on side. Duration takes the other side. Credit ETFs take the credit leg.
    """
    def _z(series: pd.Series) -> pd.Series:
        mean = series.expanding(min_periods=252).mean()
        scale = series.expanding(min_periods=252).std(ddof=1)
        return (series - mean) / scale.replace(0.0, np.nan)

    curve_z = _z(curve.reindex(closes.index).astype(float))
    credit_z = _z(credit.reindex(closes.index).astype(float))
    dollar_z = _z(dollar.reindex(closes.index).astype(float))
    risk_on = curve_z.fillna(0.0) + credit_z.fillna(0.0) - dollar_z.fillna(0.0)
    risk_on = risk_on.where(curve_z.notna() | credit_z.notna() | dollar_z.notna())
    out = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for column in closes.columns:
        if column in EQUITY_ETFS or column in COMMODITY_ETFS:
            out[column] = risk_on
        elif column in DURATION_ETFS:
            out[column] = -risk_on
        elif column in CREDIT_ETFS:
            out[column] = credit_z if column == "HYG" else -credit_z
    return out


def alpha_panel(
    closes: pd.DataFrame,
    market: pd.Series,
    yields: dict[str, pd.Series],
    vix: pd.Series,
    curve: pd.Series,
    credit: pd.Series,
    dollar: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Every precommitted alpha, keyed by the name the study ablates."""
    return {
        "tsmom": tsmom_scores(closes),
        "xsmom": xs_momentum_scores(closes),
        "residual": residual_momentum_scores(closes, market),
        "reversal": reversal_scores(closes),
        "carry": carry_scores(closes, yields),
        "lowvol": low_vol_scores(closes),
        "bab": bab_scores(closes, market),
        "value": value_scores(closes),
        "pairs": pair_scores(closes),
        "vrp": vrp_scores(closes, market, vix),
        "macro": macro_scores(closes, curve, credit, dollar),
    }
