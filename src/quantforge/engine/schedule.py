"""Rebalance calendars. A True value is a decision at that close.

The trade, if any, happens on the next session. The final bar is a month-end
or quarter-end only when the next NYSE session falls in a later month, so a
mid-month history does not look like a scheduled rebalance.
"""

from __future__ import annotations

import pandas as pd

from quantforge.calendar import next_nyse_session


def month_ends(index) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    flag = pd.Series(False, index=idx)
    if len(idx) == 0:
        return flag
    months = idx.month.to_numpy()
    years = idx.year.to_numpy()
    changed = (months[1:] != months[:-1]) | (years[1:] != years[:-1])
    flag.iloc[:-1] = changed
    nxt = next_nyse_session(idx[-1])
    flag.iloc[-1] = (nxt.year, nxt.month) != (idx[-1].year, idx[-1].month)
    return flag


def quarter_ends(index) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return month_ends(idx) & pd.Series(idx.month.isin([3, 6, 9, 12]), index=idx)


def rebalance_mask(index, freq: str) -> pd.Series:
    if freq == "daily":
        idx = pd.DatetimeIndex(pd.to_datetime(index))
        return pd.Series(True, index=idx)
    if freq == "monthly":
        return month_ends(index)
    if freq == "quarterly":
        return quarter_ends(index)
    raise ValueError(f"unknown rebalance frequency {freq!r}")
