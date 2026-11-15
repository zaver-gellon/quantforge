# Faster books and the trend-account gap

Bill's since-2016 numbers are not replicated on daily bars. He quotes 19.5% a year since 2016 against SPY at 15.4%, with a worst drop of -45% against SPY's -34%, and 17.0% against 11.5% since 2022. He also quotes 72% winning boosts, about 8 a year, about 3 days each, and a standalone 3x version around 11-12% a year. On these prices, since 2016, the next-open fill compounds at 16.2% against close-to-close SPY at 14.6%, max drawdown -35.9% against -33.7%. The close fill, the optimistic stand-in for a 12:50 PT trade, compounds at 17.1% with max drawdown -35.9%. Since 2022 the next-open fill is 12.9% against SPY 11.5%, and the close fill is 14.7% against SPY 11.5%. From 2006 the next-open fill is 12.4% (max drawdown -55.1%, worst year 2008 -36.8%) and the close fill is 12.7%. Next-open boosts since 2016: win rate 77.4% (75.3% beat SPY), 8.6 a year, 3.3 days on average, average P&L 0.8%, worst -7.4%. Close-fill boosts: win rate 74.2%, 8.6 a year, 3.3 days. Standalone UPRO-or-cash since 2016 is 10.4% next-open and 14.5% at the close. The deflated Sharpe of the next-open book's daily gap over SPY, on the 2006-onward window and the full ledger, is 0.311. Over the lead book it is 0.000. Removing 2012, the year the gap over SPY added the most, leaves a strategy compound growth of 11.8% against SPY 10.8%. The walk-forward pick on the next-open grid is rsi15_f50_n10; the published signal stays the named rule. The gap versus his quote is the 12:50 PT price versus a daily close, the fill (this close versus the next open), UPRO's financing and expense ratio in the stitched series, 5 bps per side, and the rule that an exit close does not re-enter. Through the 2026-11-13 close, the live 2x book (50% SPY + 50% UPRO, 200-day 2% band, dip on the SPY half) compounds at 15.4% out of sample with a Sharpe of 0.62, a worst drawdown of -46.7%, and a worst year of 2022 (-26.7%). It trails close-to-close SPY in 39.5% of rolling 21-session windows and 35.6% of rolling 63-session windows. The unlevered 1.0x mix, which is 100% SPY plus the same dip, compounds at 10.5%. SPY/BIL, the band with no leverage and no dip, compounds at 9.2% with a max drawdown of -22.0%. The shortfall versus SPY since late October is what a 2x book does in a soft or choppy stretch: the same down days are larger, and the dip can add SSO on the SPY half while the market is already weak. Moving the mix reset from quarterly to the walk-forward rebalance pick (daily) does not clear the materiality bar. Swapping the vehicles for QQQ/TQQQ (walk-forward pick qqq_band04) does not clear it. The lead book is unchanged. meanrev (ibs_0.2_n03_UPRO) has an out-of-sample compound growth of 8.3%, Sharpe 0.43, worst drawdown -37.2%, worst year 2023 (-13.7%). It is not validated. rotation (TQQQ_L63) has an out-of-sample compound growth of 17.5%, Sharpe 0.55, worst drawdown -71.0%, worst year 2022 (-60.7%). It is not validated. nasdaq-vol (band02_vol30) has an out-of-sample compound growth of 16.3%, Sharpe 0.62, worst drawdown -47.1%, worst year 2008 (-31.3%). It is not validated. btc-trend (sma10_030) has an out-of-sample compound growth of 35.8%, Sharpe 0.83, worst drawdown -66.5%, worst year 2022 (-31.7%). It is not validated. spy-rsi-boost (rsi10_f33_n10) has an out-of-sample compound growth of 12.4%, Sharpe 0.58, worst drawdown -55.1%, worst year 2008 (-36.8%). It is not validated.

## Bill's SPY + RSI(2) boost

The named rule is RSI(2) below 10, one third of SPY moved into UPRO, back to SPY when RSI(2) is above 70, SPY closes above its 5-day average, or 10 sessions have passed. Close fill trades at the signal close. Next open trades the next morning. Walk-forward pick `rsi15_f50_n10`; the signal file keeps `rsi10_f33_n10`. Deflated Sharpe of the next-open gap over SPY is 0.311; over the lead book, 0.000.

| window | fill | CAGR | SPY | Sharpe | max DD | SPY DD | worst year | win rate | avg P&L | worst P&L | per year | days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oos_2006 | next open | 12.4% | 11.0% | 0.580 | -55.1% | -55.2% | 2008 -36.8% | 75.1% | 0.6% | -23.1% | 8.1 | 3.4 |
| oos_2006 | close | 12.7% | 11.0% | 0.593 | -55.1% | -55.2% | 2008 -36.8% | 72.8% | 0.7% | -22.9% | 8.1 | 3.4 |
| oos_2006 | 3x next open | 9.1% | 11.0% | 0.468 | -42.5% | -55.2% | 2011 -33.0% | 73.4% | 0.8% | -39.0% | 8.1 | 3.4 |
| oos_2006 | 3x close | 10.5% | 11.0% | 0.538 | -44.2% | -55.2% | 2011 -36.2% | 72.8% | 1.3% | -39.1% | 8.1 | 3.4 |
| oos_2006 | lead overlay, next open | 15.5% | 11.0% | 0.608 | -48.0% | -55.2% | 2022 -27.4% | 71.6% | 0.9% | -22.1% | 8.1 | 3.4 |
| oos_2006 | lead overlay, close | 16.2% | 11.0% | 0.635 | -47.5% | -55.2% | 2022 -28.1% | 72.2% | 1.0% | -22.6% | 8.1 | 3.4 |
| since_2016 | next open | 16.2% | 14.6% | 0.752 | -35.9% | -33.7% | 2022 -20.0% | 77.4% | 0.8% | -7.4% | 8.6 | 3.3 |
| since_2016 | close | 17.1% | 14.6% | 0.790 | -35.9% | -33.7% | 2022 -20.0% | 74.2% | 0.9% | -7.2% | 8.6 | 3.3 |
| since_2016 | 3x next open | 10.4% | 14.6% | 0.515 | -33.4% | -33.7% | 2018 -20.2% | 75.3% | 0.9% | -14.7% | 8.6 | 3.3 |
| since_2016 | 3x close | 14.5% | 14.6% | 0.710 | -32.3% | -33.7% | 2018 -19.3% | 74.2% | 1.5% | -13.5% | 8.6 | 3.3 |
| since_2016 | lead overlay, next open | 19.0% | 14.6% | 0.692 | -43.6% | -33.7% | 2022 -27.4% | 74.2% | 1.1% | -15.7% | 8.6 | 3.3 |
| since_2016 | lead overlay, close | 19.7% | 14.6% | 0.716 | -44.8% | -33.7% | 2022 -28.1% | 75.3% | 1.2% | -15.9% | 8.6 | 3.3 |
| since_2022 | next open | 12.9% | 11.5% | 0.534 | -26.1% | -24.5% | 2022 -20.0% | 80.4% | 0.8% | -5.6% | 9.5 | 3.2 |
| since_2022 | close | 14.7% | 11.5% | 0.614 | -26.2% | -24.5% | 2022 -20.0% | 73.9% | 1.0% | -5.4% | 9.5 | 3.2 |
| since_2022 | 3x next open | 12.4% | 11.5% | 0.585 | -16.8% | -24.5% | 2022 -7.5% | 76.1% | 0.9% | -10.7% | 9.5 | 3.2 |
| since_2022 | 3x close | 20.6% | 11.5% | 1.038 | -16.0% | -24.5% | 2022 -7.4% | 73.9% | 1.7% | -9.8% | 9.5 | 3.2 |
| since_2022 | lead overlay, next open | 14.0% | 11.5% | 0.483 | -33.7% | -24.5% | 2022 -27.4% | 73.9% | 1.3% | -7.9% | 9.5 | 3.2 |
| since_2022 | lead overlay, close | 15.0% | 11.5% | 0.519 | -33.4% | -24.5% | 2022 -28.1% | 76.1% | 1.4% | -7.6% | 9.5 | 3.2 |

## Recent window

From 2026-10-26 through 2026-11-13 (15 sessions in the next-open backtest), the stitched lead book returned -1.7% and close-to-close SPY returned -0.7%. The 200-day band was on for 100.0% of those decision days and the three-down dip was on for 6.7%. If SPY's mark were $4,927 at the end of this window and both had started together, the book's mark would be about $4,879. That is the backtest's gap, not a replay of the paper account's fills.

## Exposure

Fixed rules on the study's out-of-sample window (first session of 2006 through the sample end). Gross exposure is the equity beta when the 200-day band is on and the dip is off: SPY weight plus three times the UPRO weight. The dip still moves the SPY weight into SSO. SPY/BIL is 100% SPY when the same band is on and cash when it is off, with no dip. A trailing window is 21 or 63 sessions where the book's compounded return is below close-to-close SPY.

| book | CAGR | Sharpe | max DD | worst year | trail 1m | trail 3m |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0x | 10.5% | 0.476 | -57.2% | 2008 -37.1% | 45.1% | 46.6% |
| 1.5x | 13.2% | 0.573 | -51.3% | 2008 -28.5% | 37.7% | 34.4% |
| 2.0x | 15.4% | 0.615 | -46.7% | 2022 -26.7% | 39.5% | 35.6% |
| 2.5x | 17.3% | 0.620 | -47.4% | 2022 -32.7% | 40.1% | 35.7% |
| SPY/BIL | 9.2% | 0.649 | -22.0% | 2022 -13.1% | 53.5% | 59.5% |

## Changes scored on the materiality bar

### Rebalance frequency

Monthly reset versus the live quarterly book. Daily is the other neighbor.

Decision: rejected.

### QQQ/TQQQ instead of SPY/UPRO

50/50 QQQ/TQQQ with the three-down dip on the QQQ half into QLD. Bands are 0%, 2%, and 4%.

Decision: rejected.

## Faster strategies

Walk-forward picks use only returns before each January decision. Costs are 5 bps per side. The 10 bp column is the same picks at twice the cost, not a new selection. Deflated Sharpe uses the ledger count after every cell in this run was recorded. The blend is half the lead book and half the strategy, rebalanced every session. A strategy is validated only when that walk-forward path clears the materiality bar against the lead book.

| strategy | hold | CAGR | Sharpe | max DD | worst year | trades/yr | hold days | DSR | 10 bp CAGR | corr | blend CAGR | blend max DD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| meanrev | ibs_0.2_n03_UPRO | 8.3% | 0.435 | -37.2% | 2023 -13.7% | 23.9 | 1.9 | 0.417 | 6.3% | 0.49 | 12.6% | -37.0% |

`meanrev` best year removed (2025): CAGR 6.9%, Sharpe 0.373, max DD -37.2%. Validated: False. For 2026-11-16, hold BIL 100%. Entry: IBS < 0.2. Exit: SPY close above its 5-day average. Stop: flat after 3 sessions, or when SPY closes at or below its 200-day average.

| rotation | TQQQ_L63 | 17.5% | 0.550 | -71.0% | 2022 -60.7% | 1.3 | 177.3 | 0.615 | 16.4% | 0.33 | 20.4% | -53.4% |

`rotation` best year removed (2025): CAGR 13.1%, Sharpe 0.473, max DD -71.0%. Validated: False. For 2026-11-16, hold BIL 100%. Entry: week-ending close, top trailing 63-session return among TQQQ, TMF, and UGL, and only if it beats cash. Exit: the next week-ending close, when the ranking is run again. Stop: cash when the best risky sleeve does not beat the T-bill return over the same window.

| nasdaq-vol | band02_vol30 | 16.3% | 0.618 | -47.1% | 2008 -31.3% | 1.1 | 180.9 | 0.731 | 15.9% | 0.76 | 16.4% | -41.0% |

`nasdaq-vol` best year removed (2017): CAGR 12.7%, Sharpe 0.508, max DD -47.1%. Validated: False. For 2026-11-16, hold BIL 100%. Entry: QQQ close above 694.8247 (200-day average times 1.02). Exit: QQQ close below 667.5767 (200-day average times 0.98). Stop: no price stop; the position is resized every session to the vol target and capped at 100% TQQQ.

| btc-trend | sma10_030 | 35.8% | 0.834 | -66.5% | 2022 -31.7% | 4.3 | 32.4 | 0.833 | 35.2% | 0.19 | 32.5% | -47.3% |

`btc-trend` best year removed (2021): CAGR 26.8%, Sharpe 0.690, max DD -66.5%. Validated: False. For 2026-11-16, hold BIL 100%. Entry: BTC 10-day average above its 30-day average. Exit: BTC 10-day average at or below its 30-day average. Stop: no price stop; the alternative is BIL.

| spy-rsi-boost | rsi10_f33_n10 | 12.4% | 0.580 | -55.1% | 2008 -36.8% | 8.1 | 3.4 | 0.311 | n/a | 0.86 | 14.1% | -50.7% |

`spy-rsi-boost` best year removed (n/a): CAGR 11.8%, Sharpe 0.634, max DD -55.1%. Validated: False. For 2026-11-16, hold SPY 100%. Entry: Wilder RSI(2) < 10 and SPY above its 200-day average. Exit: RSI(2) > 70, or SPY close above its 5-day average (754.7820), or 10 sessions. Stop: none.

## Caveats

- Leveraged ETFs before the listing are a daily-reset model: leverage times the underlying, minus financing at the lagged T-bill plus 0.40%, minus the assumed expense ratio. Listed returns replace the model after the first real close-to-close bar. UGL is 2x gold. TMF is 3x TLT. Expense ratios are assumptions (TMF 1.03%, UGL 0.95%), not a prospectus scrape.
- Cash is the lagged 13-week bill yield, not the BIL NAV. The signal names that residual BIL.
- The exposure rows are fixed rules on the 2006-onward window. They are not a walk-forward selection. The 2.0x row is the live book and is not a new trial. The other rows are.
- Mean-reversion entries sit on one ordered list so the materiality bar has neighbors. RSI 5/10/15 is a real step. IBS and the 5-day band are different rules on that list, not a smooth scale.
- Deflated Sharpe uses every counting trial in the ledger, including this run. A high full-sample Sharpe can still have a low deflated Sharpe.
- The 50/50 blend rebalances every session. That is a research combination, not a cheaper trading rule.
- Bitcoin at 5 bps per side is cheaper than a retail coin fill. The 28 bp audit is the repo's Coinbase-like preset and is not a selection trial.
- Bill's RSI boost is checked on daily bars. A fill at the signal close is an optimistic stand-in for a 12:50 PT trade. The next open is the conservative fill, and it is the one the signal file uses. Neither fill is the intraday price he sees at 12:50.
- The published RSI boost is the named rule (RSI below 10, one third into UPRO, 10-session cap), not the walk-forward pick. The grid is in the ledger so the deflated Sharpe counts those looks.
- This is research on public prices. It is not a recommendation to buy leveraged ETFs.
