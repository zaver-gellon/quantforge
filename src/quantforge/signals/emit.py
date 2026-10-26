"""Write a validated signal payload and a short companion note."""

from __future__ import annotations

import json
from pathlib import Path

from quantforge.signals.schema import ALLOWED_TREND_TICKERS, validate_signal


def write_signal(
    payload: dict,
    path,
    allowed_tickers=ALLOWED_TREND_TICKERS,
    max_gross: float = 1.0,
    allow_negative: bool = False,
) -> Path:
    validate_signal(
        payload,
        allowed_tickers=allowed_tickers,
        max_gross=max_gross,
        allow_negative=allow_negative,
    )
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest


def write_signals_note(path, sections: list[tuple[str, str, dict]]) -> Path:
    """``sections`` is ``(filename, title, payload)``."""
    lines = [
        "# Signals",
        "",
        "`date` is the NYSE session the weights are for. Fill at that session's open.",
        "Weights are strategic target fractions. Names with zero weight are omitted.",
        "The fractions sum to 1 or less; anything left over is uninvested cash.",
        "Trend-core tickers are only SPY, UPRO, SSO, and BIL.",
        "A later flagship (or the dip-buy overlay) uses the same object shape in its own file.",
        "",
        "`trade` is an extra key. `true` means the sleeve instrument changed or a scheduled",
        "mix rebalance is due. A bot that marks to `weights` on every session instead rebalances daily.",
        "",
    ]
    for filename, title, payload in sections:
        weights = ", ".join(f"{ticker} {value:.4f}" for ticker, value in sorted(payload["weights"].items()))
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- File: `{filename}`")
        lines.append(f"- Session: `{payload['date']}`")
        lines.append(f"- Weights: {weights or '(all cash)'}")
        if "signal_date" in payload:
            lines.append(f"- Signal close: `{payload['signal_date']}`")
        if "trade_reason" in payload:
            lines.append(f"- Trade reason: `{payload['trade_reason']}`")
        if payload.get("diagnostics"):
            diag = payload["diagnostics"]
            if "trend_on" in diag:
                lines.append(f"- Trend on: `{diag['trend_on']}`")
            if "hedge_on" in diag:
                lines.append(f"- Credit hedge on: `{diag['hedge_on']}`")
            if diag.get("hedge_rule"):
                lines.append(f"- Credit hedge rule: `{diag['hedge_rule']}`")
            if diag.get("vol_rule"):
                lines.append(f"- Vol sleeve rule: `{diag['vol_rule']}`")
            if "sleeve_scale" in diag:
                lines.append(f"- Sleeve scale: `{diag['sleeve_scale']}`")
            if "dip_on" in diag:
                lines.append(f"- Dip-buy on: `{diag['dip_on']}`")
            if "reference_close" in diag and diag.get("sma") is not None:
                lines.append(
                    f"- SPY close {diag['reference_close']:.2f} versus SMA {diag['sma']:.2f}"
                )
        lines.append("")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return dest
