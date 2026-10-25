import pytest

from quantforge.signals.schema import ALLOWED_TREND_TICKERS, SchemaError, validate_signal


def _ok(**overrides):
    payload = {"date": "2026-10-26", "weights": {"SPY": 0.5, "UPRO": 0.5}}
    payload.update(overrides)
    return payload


def test_accepts_bot_contract_and_extra_keys():
    payload = validate_signal(
        _ok(strategy="trend_core", generated_at="2026-10-23T20:00:00Z", diagnostics={"trend_on": True}),
        allowed_tickers=ALLOWED_TREND_TICKERS,
    )
    assert payload["date"] == "2026-10-26"
    assert payload["weights"]["SPY"] == 0.5


def test_rejects_bad_date_zero_weight_unknown_ticker_and_sum():
    with pytest.raises(SchemaError):
        validate_signal({"weights": {"SPY": 1}})
    with pytest.raises(SchemaError):
        validate_signal(_ok(date="10/26/2026"))
    with pytest.raises(SchemaError):
        validate_signal(_ok(date="2026-02-31"))
    with pytest.raises(SchemaError):
        validate_signal(_ok(weights={"SPY": 0.0, "UPRO": 1.0}))
    with pytest.raises(SchemaError):
        validate_signal(_ok(weights={"SPY": 0.5, "QQQ": 0.5}), allowed_tickers=ALLOWED_TREND_TICKERS)
    with pytest.raises(SchemaError):
        validate_signal(_ok(weights={"SPY": 0.6, "UPRO": 0.6}))
    with pytest.raises(SchemaError):
        validate_signal(_ok(weights={"SPY": True}))


def test_remainder_cash_may_sum_below_one():
    validate_signal({"date": "2026-10-26", "weights": {"SPY": 0.5}})
    validate_signal({"date": "2026-10-26", "weights": {}})
