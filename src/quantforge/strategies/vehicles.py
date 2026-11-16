"""Stitch a listed leveraged ETF onto the daily-reset model used before it existed."""

from __future__ import annotations

import pandas as pd

from quantforge.engine.leverage import (
    FINANCING_SPREAD,
    LEVERAGED_ETFS,
    stitch_ohlc,
    synthesize_leveraged,
)


def stitched_letf(
    symbol: str,
    underlying_open: pd.Series,
    underlying_close: pd.Series,
    rf_daily: pd.Series,
    listed_open: pd.Series | None = None,
    listed_close: pd.Series | None = None,
    financing_spread: float = FINANCING_SPREAD,
) -> pd.DataFrame:
    """Synthetic daily-reset path, replaced by listed returns after the first real close-to-close bar.

    ``rf_daily`` is the lagged T-bill yield. Financing is ``(L - 1)`` times
    that yield plus ``financing_spread``. The expense ratio is the assumption
    stored on ``LEVERAGED_ETFS``.
    """
    meta = LEVERAGED_ETFS[symbol]
    synthetic = synthesize_leveraged(
        underlying_open,
        underlying_close,
        rf_daily,
        leverage=meta["leverage"],
        expense_ratio=meta["expense_ratio"],
        financing_spread=financing_spread,
    )
    if listed_open is None or listed_close is None:
        return synthetic
    return stitch_ohlc(synthetic["open"], synthetic["close"], listed_open, listed_close)
