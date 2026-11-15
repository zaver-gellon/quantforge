"""Render the faster-book write-up from the results object. No figures are typed here."""

from __future__ import annotations

import math


def pct(value, digits: int = 1) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def num(value, digits: int = 2) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _row(cells) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _worst(metrics: dict) -> str:
    year = metrics.get("worst_year")
    if year is None:
        return "n/a"
    return f"{year} {pct(metrics.get('worst_year_return'))}"


def _replication_lines(block: dict) -> list[str]:
    windows = block.get("windows") or {}
    if not windows:
        return []
    lines = [
        "## Bill's SPY + RSI(2) boost",
        "",
        "The named rule is RSI(2) below 10, one third of SPY moved into UPRO, back to SPY when RSI(2) is above 70, "
        "SPY closes above its 5-day average, or 10 sessions have passed. "
        "Close fill trades at the signal close. Next open trades the next morning. "
        f"Walk-forward pick `{block.get('walk_forward_pick')}`; the signal file keeps `{block.get('named_cell')}`. "
        f"Deflated Sharpe of the next-open gap over SPY is {num((block.get('vs_spy') or {}).get('dsr'), 3)}; "
        f"over the lead book, {num((block.get('vs_lead') or {}).get('dsr'), 3)}.",
        "",
        _row(["window", "fill", "CAGR", "SPY", "Sharpe", "max DD", "SPY DD", "worst year", "win rate", "avg P&L", "worst P&L", "per year", "days"]),
        _row(["---"] * 13),
    ]
    labels = (
        ("next_open", "next open"),
        ("close", "close"),
        ("standalone_next_open", "3x next open"),
        ("standalone_close", "3x close"),
        ("overlay_next_open", "lead overlay, next open"),
        ("overlay_close", "lead overlay, close"),
    )
    for window, body in windows.items():
        for key, fill in labels:
            side = body.get(key) or {}
            stats = side.get("stats") or {}
            spy = side.get("spy") or {}
            episodes = side.get("episodes") or {}
            lines.append(_row([
                window,
                fill,
                pct(stats.get("cagr")),
                pct(spy.get("cagr")),
                num(stats.get("sharpe"), 3),
                pct(stats.get("max_drawdown")),
                pct(spy.get("max_drawdown")),
                _worst(stats),
                pct(episodes.get("win_rate")),
                pct(episodes.get("avg_pnl")),
                pct(episodes.get("worst_pnl")),
                num(episodes.get("per_year"), 1),
                num(episodes.get("avg_days"), 1),
            ]))
    lines.append("")
    return lines


def render_report(results: dict) -> str:
    lines = [
        "# Faster books and the trend-account gap",
        "",
        results.get("verdict", "").strip(),
        "",
    ]
    lines.extend(_replication_lines(results.get("replication") or {}))
    lines += [
        "## Recent window",
        "",
        results.get("recent_note", "").strip(),
        "",
        "## Exposure",
        "",
        "Fixed rules on the study's out-of-sample window (first session of 2006 through the sample end). "
        "Gross exposure is the equity beta when the 200-day band is on and the dip is off: "
        "SPY weight plus three times the UPRO weight. The dip still moves the SPY weight into SSO. "
        "SPY/BIL is 100% SPY when the same band is on and cash when it is off, with no dip. "
        "A trailing window is 21 or 63 sessions where the book's compounded return is below close-to-close SPY.",
        "",
        _row(["book", "CAGR", "Sharpe", "max DD", "worst year", "trail 1m", "trail 3m"]),
        _row(["---"] * 7),
    ]
    for row in results.get("exposure") or []:
        oos = row.get("oos") or {}
        trail = row.get("trail") or {}
        lines.append(_row([
            row.get("name"),
            pct(oos.get("cagr")),
            num(oos.get("sharpe"), 3),
            pct(oos.get("max_drawdown")),
            _worst(oos),
            pct((trail.get("1m") or {}).get("trail_rate")),
            pct((trail.get("3m") or {}).get("trail_rate")),
        ]))
    lines += ["", "## Changes scored on the materiality bar", ""]
    for key, title in (("rebalance", "Rebalance frequency"), ("qqq", "QQQ/TQQQ instead of SPY/UPRO")):
        block = results.get(key) or {}
        lines += [f"### {title}", "", block.get("note", "").strip(), ""]
        lines.append(f"Decision: {block.get('decision', 'n/a')}.")
        lines.append("")
    lines += ["## Faster strategies", ""]
    lines.append(
        "Walk-forward picks use only returns before each January decision. "
        "Costs are 5 bps per side. The 10 bp column is the same picks at twice the cost, not a new selection. "
        "Deflated Sharpe uses the ledger count after every cell in this run was recorded. "
        "The blend is half the lead book and half the strategy, rebalanced every session. "
        "A strategy is validated only when that walk-forward path clears the materiality bar against the lead book."
    )
    lines.append("")
    lines.append(_row([
        "strategy", "hold", "CAGR", "Sharpe", "max DD", "worst year",
        "trades/yr", "hold days", "DSR", "10 bp CAGR", "corr", "blend CAGR", "blend max DD",
    ]))
    lines.append(_row(["---"] * 13))
    for row in results.get("strategies") or []:
        oos = row.get("oos") or {}
        stress = row.get("cost_10bps") or {}
        blend = row.get("blend") or {}
        lines.append(_row([
            row.get("id"),
            row.get("cell"),
            pct(oos.get("cagr")),
            num(oos.get("sharpe"), 3),
            pct(oos.get("max_drawdown")),
            _worst(oos),
            num(row.get("trades_per_year"), 1),
            num(row.get("avg_holding_days"), 1),
            num(row.get("dsr"), 3),
            pct(stress.get("cagr")),
            num(row.get("correlation"), 2),
            pct(blend.get("cagr")),
            pct(blend.get("max_drawdown")),
        ]))
        removed = row.get("best_year_removed") or {}
        lines.append("")
        lines.append(
            f"`{row.get('id')}` best year removed ({removed.get('year', 'n/a')}): "
            f"CAGR {pct(removed.get('cagr'))}, Sharpe {num(removed.get('sharpe'), 3)}, "
            f"max DD {pct(removed.get('max_drawdown'))}. "
            f"Validated: {row.get('validated')}. {row.get('signal_line', '')}"
        )
        lines.append("")
    if results.get("btc_note"):
        lines += ["## Bitcoin", "", results["btc_note"].strip(), ""]
    lines += ["## Caveats", ""]
    for item in results.get("caveats") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
