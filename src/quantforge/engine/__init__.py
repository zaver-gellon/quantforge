"""Vectorized long-only backtest, costs, and daily-reset leverage."""

from quantforge.engine.backtest import BacktestResult, run_backtest, run_core_satellite, run_trend_core
from quantforge.engine.costs import CostModel, crypto_costs, equity_costs
from quantforge.engine.leverage import (
    FINANCING_SPREAD,
    LEVERAGED_ETFS,
    stitch_ohlc,
    synthesize_leveraged,
    tracking_stats,
)
from quantforge.engine.schedule import month_ends, quarter_ends, rebalance_mask

__all__ = [
    "FINANCING_SPREAD",
    "LEVERAGED_ETFS",
    "BacktestResult",
    "CostModel",
    "crypto_costs",
    "equity_costs",
    "month_ends",
    "quarter_ends",
    "rebalance_mask",
    "run_backtest",
    "run_core_satellite",
    "run_trend_core",
    "stitch_ohlc",
    "synthesize_leveraged",
    "tracking_stats",
]
