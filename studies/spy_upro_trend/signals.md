# Signals

`date` is the NYSE session the weights are for. Fill at that session's open.
Weights are strategic target fractions. Names with zero weight are omitted.
The fractions sum to 1 or less; anything left over is uninvested cash.
Trend-core tickers are only SPY, UPRO, SSO, and BIL.
A later flagship (or the dip-buy overlay) uses the same object shape in its own file.

`trade` is an extra key. `true` means the sleeve instrument changed or a scheduled
mix rebalance is due. A bot that marks to `weights` on every session instead rebalances daily.

## Trend-core (2% band, dip overlay on the SPY half)

- File: `signals.json`
- Session: `2026-10-26`
- Weights: SPY 0.5000, UPRO 0.5000
- Signal close: `2026-10-23`
- Trade reason: `none`
- Trend on: `True`
- SPY close 759.97 versus SMA 724.00


`signals_flagship.json` is written by `python studies/flagship/run.py` and is not this fixed book.
