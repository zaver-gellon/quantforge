"""Per-asset one-way costs. Cash is free unless it is listed explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Proportional one-way cost by symbol, in basis points.

    ``commission_bps`` is commission plus the quoted half-spread.
    ``slippage_bps`` is extra adverse fill versus the reference price (the next
    open, or the next close in the robustness fill). Both are charged on the
    absolute traded notional of that symbol. Symbols that are omitted use the
    defaults, which are zero so unlisted cash does not pay a spread.

    This is the small-cost approximation: the drag does not depend on the path
    inside the bar, and it scales with the dollars traded.
    """

    commission_bps: dict[str, float] = field(default_factory=dict)
    slippage_bps: dict[str, float] = field(default_factory=dict)
    default_commission_bps: float = 0.0
    default_slippage_bps: float = 0.0

    def one_way_fraction(self, symbol: str) -> float:
        commission = self.commission_bps.get(symbol, self.default_commission_bps)
        slippage = self.slippage_bps.get(symbol, self.default_slippage_bps)
        return (commission + slippage) / 10_000.0


def equity_costs(
    symbols: list[str],
    commission_bps: float = 5.0,
    slippage_bps: float = 0.0,
) -> CostModel:
    """US equity / ETF preset. The desk's base case is 5 bps per side, all-in."""
    return CostModel(
        commission_bps={symbol: commission_bps for symbol in symbols},
        slippage_bps={symbol: slippage_bps for symbol in symbols},
    )


def crypto_costs(
    symbols: list[str],
    commission_bps: float = 25.0,
    slippage_bps: float = 3.0,
) -> CostModel:
    """Coinbase-like preset: 25 bps fee and 3 bps of slippage on market fills."""
    return CostModel(
        commission_bps={symbol: commission_bps for symbol in symbols},
        slippage_bps={symbol: slippage_bps for symbol in symbols},
    )
