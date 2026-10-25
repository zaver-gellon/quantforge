"""Square-root market impact.

The one-way cost, as a fraction of the traded notional, is

    y * daily_vol * sqrt(notional / ADV)

``y`` is a dimensionless coefficient. 0.5 is a conservative placeholder, not
a fit to our own fills. ADV is trailing dollar volume known at the decision
close. A small account on a liquid ETF pays almost nothing. The capacity
table inverts the formula for the notional that would cost a given number of
basis points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def impact_fraction(notional: float, adv: float, daily_vol: float, y: float = 0.5) -> float:
    if notional <= 0 or adv <= 0 or not np.isfinite(daily_vol) or daily_vol <= 0:
        return 0.0
    return float(y * daily_vol * np.sqrt(notional / adv))


def impact_matrix(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    volume: pd.DataFrame,
    aum: float,
    y: float = 0.5,
    adv_window: int = 21,
) -> pd.DataFrame:
    """Extra one-way cost fraction on each decision date.

    Notional is ``|Δw| * aum`` using the target-weight change, which ignores
    drift between rebalances. That is the order a monthly rebalance would send.
    """
    adv = (closes * volume).rolling(adv_window, min_periods=5).median().shift(1)
    vol = closes.pct_change().rolling(adv_window, min_periods=5).std(ddof=1).shift(1)
    change = weights.diff().abs()
    change.iloc[0] = weights.iloc[0].abs()
    out = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    for asset in weights.columns:
        if asset not in adv.columns:
            continue
        notional = change[asset] * float(aum)
        adv_a = adv[asset].reindex(weights.index)
        vol_a = vol[asset].reindex(weights.index)
        frac = y * vol_a * np.sqrt(notional / adv_a.replace(0.0, np.nan))
        out[asset] = frac.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return out


def capacity_notionals(
    adv: float,
    daily_vol: float,
    cost_bps: tuple[float, ...] = (5.0, 10.0, 25.0),
    y: float = 0.5,
) -> dict[float, float]:
    """Dollars of one order that would cost each bps target on this ADV."""
    out = {}
    if adv <= 0 or daily_vol <= 0:
        return {bps: 0.0 for bps in cost_bps}
    for bps in cost_bps:
        # bps/1e4 = y * vol * sqrt(q/adv)  => q = adv * (bps/1e4 / (y*vol))^2
        ratio = (bps / 1e4) / (y * daily_vol)
        out[bps] = float(adv * ratio ** 2)
    return out
