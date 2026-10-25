"""signals.json contract consumed by the paper-trading bot.

Required keys:

- ``date``: ``YYYY-MM-DD``, the US/Eastern NYSE session the weights are for
  (the next session after the signal close; fill at that session's open).
- ``weights``: ``{ticker: fraction}`` with each fraction ``> 0``. For the
  trend-core strategy the only tickers are ``SPY``, ``UPRO``, ``SSO``, and
  ``BIL``. The values sum to at most 1. The remainder is uninvested cash.

Zero-weight names are omitted, not sent as 0. Extra keys (strategy name,
``generated_at``, reasons, diagnostics) are allowed and ignored by the bot.
"""

from __future__ import annotations

import re
from datetime import datetime

ALLOWED_TREND_TICKERS = frozenset({"SPY", "UPRO", "SSO", "BIL"})
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SchemaError(ValueError):
    """The payload is not a signal the bot can read."""


def validate_signal(
    payload: dict,
    allowed_tickers: frozenset[str] | None = None,
    max_gross: float = 1.0,
    allow_negative: bool = False,
) -> dict:
    if not isinstance(payload, dict):
        raise SchemaError("signal must be a JSON object")
    date = payload.get("date")
    if not isinstance(date, str) or _DATE.fullmatch(date) is None:
        raise SchemaError('date must be "YYYY-MM-DD"')
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise SchemaError("date is not a real calendar date") from exc
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise SchemaError("weights must be an object of ticker to fraction")
    total = 0.0
    for ticker, value in weights.items():
        if not isinstance(ticker, str) or not ticker:
            raise SchemaError("weight tickers must be non-empty strings")
        if allowed_tickers is not None and ticker not in allowed_tickers:
            raise SchemaError(f"ticker {ticker} is not allowed ({sorted(allowed_tickers)})")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"weight for {ticker} must be a number")
        if allow_negative:
            if value == 0:
                raise SchemaError(f"weight for {ticker} must be non-zero")
        elif not value > 0:
            raise SchemaError(f"weight for {ticker} must be > 0")
        total += abs(float(value)) if allow_negative else float(value)
    if total > max_gross + 1e-6:
        raise SchemaError(f"weights sum to {total:.6f}, which is above {max_gross}")
    return payload
