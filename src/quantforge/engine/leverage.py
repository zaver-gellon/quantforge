"""Daily-reset leveraged totals built from an underlying total-return series.

Close-to-close simple return:

    r_L = L * r - (L - 1) * (rf + spread / 252) - expense_ratio / 252

``rf`` is the lagged daily T-bill yield. ``spread`` is an annual financing
spread over that yield (swap / futures implied funding above bills). The
expense ratio is the ETF's annual expense ratio.

The synthetic open earns ``L`` times the underlying overnight gap from the
previous synthetic close. Financing and the expense ratio sit in the residual
between that open and the close, so the close-to-close return matches the
formula exactly. This is a model of a daily-reset fund, not a reprint of the
sponsor's NAV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ProShares-style expense ratios used by the desk and checked against the
# overlap with the listed funds. They are assumptions, not a live prospectus scrape.
LEVERAGED_ETFS = {
    "SSO": {"underlying": "SPY", "leverage": 2, "expense_ratio": 0.0089},
    "UPRO": {"underlying": "SPY", "leverage": 3, "expense_ratio": 0.0091},
    "QLD": {"underlying": "QQQ", "leverage": 2, "expense_ratio": 0.0095},
    "TQQQ": {"underlying": "QQQ", "leverage": 3, "expense_ratio": 0.0084},
}
FINANCING_SPREAD = 0.004


def synthesize_leveraged(
    open_: pd.Series,
    close: pd.Series,
    rf_daily: pd.Series,
    leverage: float,
    expense_ratio: float,
    financing_spread: float = FINANCING_SPREAD,
) -> pd.DataFrame:
    """Return synthetic ``open`` and ``close`` indexed like ``close``."""
    close = close.astype(float).copy()
    open_ = open_.reindex(close.index).astype(float)
    rf = rf_daily.reindex(close.index).astype(float).fillna(0.0)
    underlying = close.pct_change()
    overnight = open_ / close.shift(1) - 1.0
    borrow = rf + financing_spread / 252.0
    leveraged = leverage * underlying - (leverage - 1.0) * borrow - expense_ratio / 252.0
    leveraged = leveraged.clip(lower=-0.999)
    n = len(close)
    out_open = np.empty(n)
    out_close = np.empty(n)
    out_open[0] = 100.0
    out_close[0] = 100.0
    on = overnight.to_numpy(dtype=float)
    rl = leveraged.to_numpy(dtype=float)
    for i in range(1, n):
        gap = leverage * on[i] if np.isfinite(on[i]) else 0.0
        out_open[i] = out_close[i - 1] * max(1.0 + gap, 1e-8)
        ret = rl[i] if np.isfinite(rl[i]) else 0.0
        out_close[i] = max(out_close[i - 1] * (1.0 + ret), 1e-8)
    return pd.DataFrame({"open": out_open, "close": out_close}, index=close.index)


def stitch_ohlc(
    synth_open: pd.Series,
    synth_close: pd.Series,
    actual_open: pd.Series,
    actual_close: pd.Series,
) -> pd.DataFrame:
    """Synthetic path before the listing, actual total returns after.

    The first session that has an actual close but no previous actual close
    still follows the synthetic return. From the next session on, open and
    close follow the actual asset's returns, chained off the running level.
    """
    idx = synth_close.index
    so = synth_open.reindex(idx).to_numpy(dtype=float)
    sc = synth_close.reindex(idx).to_numpy(dtype=float)
    ao = actual_open.reindex(idx).to_numpy(dtype=float)
    ac = actual_close.reindex(idx).to_numpy(dtype=float)
    out_o = np.empty(len(idx))
    out_c = np.empty(len(idx))
    out_o[0] = so[0]
    out_c[0] = sc[0]
    for i in range(1, len(idx)):
        prev_actual = ac[i - 1]
        if (
            np.isfinite(ac[i])
            and np.isfinite(prev_actual)
            and prev_actual > 0
            and np.isfinite(ao[i])
            and ao[i] > 0
        ):
            out_o[i] = out_c[i - 1] * (ao[i] / prev_actual)
            out_c[i] = out_c[i - 1] * (ac[i] / prev_actual)
        elif np.isfinite(sc[i - 1]) and sc[i - 1] > 0 and np.isfinite(sc[i]) and np.isfinite(so[i]):
            out_o[i] = out_c[i - 1] * (so[i] / sc[i - 1])
            out_c[i] = out_c[i - 1] * (sc[i] / sc[i - 1])
        else:
            out_o[i] = out_c[i - 1]
            out_c[i] = out_c[i - 1]
    return pd.DataFrame({"open": out_o, "close": out_c}, index=idx)


def tracking_stats(
    synthetic_close: pd.Series,
    actual_close: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """Close-to-close overlap of a synthetic fund versus the listed ETF."""
    actual = actual_close.dropna().astype(float)
    synthetic = synthetic_close.reindex(actual.index).astype(float)
    a = actual.pct_change()
    s = synthetic.pct_change()
    mask = a.notna() & s.notna() & np.isfinite(a) & np.isfinite(s)
    aa = a[mask]
    ss = s[mask]
    n = int(mask.sum())
    if n < 30:
        return {"n_days": n}
    years = n / periods_per_year

    def _cagr(series: pd.Series) -> float:
        equity = float((1.0 + series).prod())
        if equity <= 0 or years <= 0:
            return float("nan")
        return equity ** (1.0 / years) - 1.0

    diff = aa - ss
    return {
        "n_days": n,
        "years": years,
        "start": str(aa.index[0].date()),
        "end": str(aa.index[-1].date()),
        "actual_cagr": _cagr(aa),
        "synthetic_cagr": _cagr(ss),
        "daily_corr": float(aa.corr(ss)),
        "tracking_error": float(diff.std(ddof=1) * np.sqrt(periods_per_year)),
        "cagr_gap": _cagr(ss) - _cagr(aa),
    }
