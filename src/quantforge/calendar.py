"""NYSE session dates. Weekends and full-day holidays are skipped.

Early-close sessions are regular trading days. Juneteenth is observed only from
2022, the year the NYSE adopted it.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    MO,
    TH,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

_HOLIDAYS: pd.DatetimeIndex | None = None


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYear", month=1, day=1, observance=nearest_workday),
        Holiday("MLK", month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday("Presidents", month=2, day=1, offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial", month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
        Holiday("Independence", month=7, day=4, observance=nearest_workday),
        Holiday("Labor", month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_holidays(start: str = "1990-01-01", end: str = "2040-12-31") -> pd.DatetimeIndex:
    named = NYSEHolidayCalendar().holidays(start, end, return_name=True)
    keep = [
        ts
        for ts, name in named.items()
        if not (name == "Juneteenth" and ts < pd.Timestamp("2022-06-01"))
    ]
    return pd.DatetimeIndex(keep).sort_values().normalize()


def _holidays() -> pd.DatetimeIndex:
    global _HOLIDAYS
    if _HOLIDAYS is None:
        _HOLIDAYS = nyse_holidays()
    return _HOLIDAYS


def next_nyse_session(after) -> pd.Timestamp:
    """First NYSE session strictly after ``after``.

    ``after`` is a calendar date (the signal close). The returned date is the
    session the resulting weights are for: the next open.
    """
    stamp = pd.Timestamp(after)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    stamp = stamp.normalize()
    nxt = stamp + CustomBusinessDay(holidays=_holidays())
    return pd.Timestamp(nxt).normalize()


def is_nyse_session(day) -> bool:
    stamp = pd.Timestamp(day).tz_localize(None).normalize()
    if stamp.day_of_week >= 5:
        return False
    return stamp not in set(_holidays())
