import pandas as pd

from quantforge.calendar import is_nyse_session, next_nyse_session


def test_next_session_skips_weekends_and_holidays():
    assert next_nyse_session("2026-10-23") == pd.Timestamp("2026-10-26")
    assert next_nyse_session("2024-12-24") == pd.Timestamp("2024-12-26")
    assert next_nyse_session("2024-11-27") == pd.Timestamp("2024-11-29")
    assert next_nyse_session("2025-12-31") == pd.Timestamp("2026-01-02")
    assert next_nyse_session("2026-01-16") == pd.Timestamp("2026-01-20")
    assert next_nyse_session("2026-04-02") == pd.Timestamp("2026-04-06")


def test_juneteenth_only_from_2022():
    assert is_nyse_session("2021-06-18")
    assert not is_nyse_session("2022-06-20")
    assert not is_nyse_session("2024-06-19")
