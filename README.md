# quantforge

Walk-forward backtests for systematic strategies. Decisions are fixed at the close and filled at the next open. Parameters that were chosen after seeing a result stay in a ledger, and later significance tests use that count.

The trend-core book is half SPY and half UPRO while SPY is 2% above its 200-day average (enter above 1.02 times the average, exit below 0.98 times, hold the previous state in between). After three strictly down SPY closes, only the SPY half switches to SSO. The UPRO half is not resized by the dip. The kept credit-spread rule can move that UPRO half to BIL; the iteration log in [`studies/spy_upro_trend`](studies/spy_upro_trend) records the threshold the walk-forward rule is holding. The live signal reads the saved FRED file `studies/spy_upro_trend/BAA10Y.csv` and does not call FRED when that file is present.

[`studies/flagship`](studies/flagship) is the return-max track: out-of-sample CAGR under 30%, 45%, and 60% drawdown budgets. A full-sample ranking in that write-up is not a recommendation.

[`studies/apex`](studies/apex) is the multi-strategy book: a library of alphas, a regime scale, and a walk-forward choice between equal-risk and a purged machine-learning combiner. The tearsheet says whether that path beat SPY and the trend-core book after costs, and which components cleared a precommitted hurdle. A full-sample ranking there is not a recommendation.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer. Runtime dependencies are numpy, pandas, pyarrow, and yfinance. Tests use pytest.

```bash
python -m pytest
quantforge --help
```

## Layout

| Path | Role |
| --- | --- |
| `quantforge.data` | Yahoo (stocks, ETFs, `^IRX`) and Coinbase public candles, parquet cache, registry for a later Alpaca or Polygon adapter |
| `quantforge.engine` | Next-open backtest, costs, rebalance calendars, daily-reset leveraged ETF synthesis |
| `quantforge.eval` | CAGR, Sharpe, drawdown, walk-forward selection, block bootstrap, PSR/DSR, trial ledger |
| `quantforge.signals` | `signals.json` contract for the paper-trading bot |
| `quantforge.strategies` | Trend-core and dip-buy target weights |
| `studies/` | One command that fetches data, runs the study, and writes the tables |

## Data

Free sources only. `YahooSource` uses yfinance and falls back to the public chart API. Equity OHLC is total-return adjusted. `^IRX` is the 13-week T-bill yield in percent. `CoinbaseSource` reads public daily candles. Bars are cached as parquet under `.cache/quantforge` (or `QUANTFORGE_CACHE`).

Paid vendors are stubs. Subclass `AlpacaSource` or `PolygonSource`, implement `fetch_bars`, and `register_source`.

```bash
quantforge fetch yahoo SPY SSO UPRO ^IRX --start 1993-01-01
quantforge fetch coinbase BTC-USD ETH-USD --start 2017-01-01
```

## Backtest

`run_backtest` takes open, close, target weights, and a rebalance mask. `targets` on date t are the weights decided at that close. The default fill is the next open: yesterday's holdings earn the overnight gap, the trade happens at the open, and the new holdings earn the open-to-close move. Cash earns a yield that the caller has already lagged. Weights are long-only and sum to at most 1. The remainder is cash.

`run_core_satellite` is the trend book. The core drifts between mix dates. A trend flip moves only the sleeve.

Leveraged ETFs that did not exist for the whole sample are synthesized as a daily reset:

```text
r_L = L * r - (L - 1) * (rf + spread / 252) - expense_ratio / 252
```

The study uses expense ratios of 0.89% (SSO) and 0.91% (UPRO) and a financing spread of 0.40% over T-bills. Where the listed fund exists, the overlap statistics are the check, and a stitched path uses listed returns after inception.

## Evaluation

`walk_forward_select` scores each variant on returns strictly before the decision date. `summarize` reports CAGR, volatility, excess Sharpe, max drawdown, worst year, and the t-stat of active return versus a benchmark.

`probabilistic_sharpe` and `deflated_sharpe` follow Bailey and López de Prado. The benchmark Sharpe is the expected maximum of the trial Sharpes stored in `studies/trial_ledger.json`. Re-running the same study, family, name, parameters, and sample start updates the row and does not add a trial. Cost and fill checks are recorded as audits and do not increment N.

`block_bootstrap` resamples circular blocks jointly with the benchmark and reports the distribution of a $5,000 account.

## Signals

The paper-trading bot reads a JSON object:

```json
{
  "date": "2026-10-26",
  "weights": {"SPY": 0.5, "UPRO": 0.5}
}
```

`date` is the US/Eastern NYSE session the weights are for. Fill that session's open. `weights` values are greater than 0 and sum to at most 1. Anything left over is cash. Trend-core tickers are only `SPY`, `UPRO`, `SSO`, and `BIL`. Extra keys (`strategy`, `generated_at`, `diagnostics`, `trade`) are allowed.

`quantforge signals --strategy trend-core` writes `studies/spy_upro_trend/signals.json`: the 2% band book, with the dip overlay on the SPY half. `quantforge signals --strategy dip-buy` writes `signals_dipbuy.json` for that SPY-half overlay on its own. `quantforge signals --strategy flagship` replays the flagship study's `live_signal` into `signals_flagship.json`. `signals.md` is a short companion note.

`trade` is extra. It is true when the sleeve instrument changed or a scheduled mix reset is due. A bot that marks to `weights` every session rebalances daily instead of letting the mix drift.

## Study

```bash
python studies/spy_upro_trend/run.py
python studies/flagship/run.py
python studies/apex/run.py
```

or `quantforge study spy-upro-trend`, `quantforge study flagship`, and `quantforge study apex`.

The protocol is at the top of `studies/spy_upro_trend/run.py` and is copied into `RESULTS.md`. It specifies the grid, the walk-forward rule, and the bar a result has to clear. The desk cell is 50% SPY and 50% UPRO. The UPRO half turns on only when the close is above 1.02 times the 200-day average and turns off only when the close is below 0.98 times, with a quarterly mix reset. The live trend-core book adds the three-down-close overlay on the SPY half only. The 50/50 weight was chosen after the desk had seen results, so the study treats that mix as one cell among the trials and does not re-pick it on the full sample.

Outputs of the trend-core run:

- `studies/spy_upro_trend/RESULTS.md` — tables
- `studies/spy_upro_trend/results.json` — the numbers behind the tables
- `studies/spy_upro_trend/signals.json` — trend-core book, including the SPY-half dip
- `studies/spy_upro_trend/signals.md` — companion note
- `studies/trial_ledger.json` — every variant scored, shared with the flagship study

The flagship run writes `studies/flagship/RESULTS.md`, `results.json`, and `signals_flagship.json`. It also copies that signal onto `studies/spy_upro_trend/signals_flagship.json` so it does not collide with `signals.json`.

The apex run writes `studies/apex/RESULTS.md`, `results.json`, and `signals.json`. `signals_model.json` is the primary walk-forward book and `signals_long_only.json` is the cash-account book. Those files use the same `date` and `weights` object. Apex weights may name the core ETFs, and a short book may use a negative weight.

## Tests

`python -m pytest` covers the signal schema, a trend signal that does not read prices after `as_of`, next-open lookahead, costs, leverage synthesis, walk-forward causality, the ledger, and the report renderer. GitHub Actions runs the same suite on Ubuntu with Python 3.12.
