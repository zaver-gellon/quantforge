"""Tradable universe for the apex book.

The core is liquid US-listed ETFs that were trading by mid-2007, so the panel
can include 2008 without a survivorship screen of single stocks. Sector,
factor, bond, commodity, and international exposures are the ETF themselves.
Single-name momentum uses a fixed list of companies that are still listed.
That list is biased upward: delisted names are absent. It is a diagnostic
sleeve, not an input the endorsed book is allowed to require.

Crypto is BTC and ETH. They are the liquid majors that survived, which is its
own selection bias. A sleeve can use them only after they have history, and
the write-up keeps that caveat next to any crypto result.

Point-in-time: ETF constituent changes are inside the sponsor's NAV, which is
what the adjusted price reflects. This is not a reconstituted index. Yield
series (^IRX, ^FVX, ^TNX, ^TYX, ^VIX) are signals, not holdings. UUP is the
dollar proxy. Dividend yields are not recovered from the adjusted-price cache,
so equity carry is the long-horizon reversal proxy rather than a dividend yield.
"""

from __future__ import annotations

# Listed early enough to sit in one panel through 2008.
CORE_ETFS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "DBC",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
]

EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
]
DURATION_ETFS = ["TLT", "IEF", "SHY"]
CREDIT_ETFS = ["LQD", "HYG"]
COMMODITY_ETFS = ["GLD", "DBC"]

# Pre-specified pairs. Nothing here is chosen by scanning correlations.
PAIRS = (
    ("IEF", "TLT"),
    ("LQD", "HYG"),
    ("SPY", "IWM"),
    ("EFA", "EEM"),
    ("GLD", "DBC"),
    ("SPY", "EFA"),
)

# Group caps. SPY is its own group so the cap does not ban a core equity hold
# and a sector tilt at the same time.
SECTOR = {
    "SPY": "us_broad",
    "QQQ": "us_growth",
    "IWM": "us_small",
    "EFA": "intl",
    "EEM": "intl",
    "TLT": "duration",
    "IEF": "duration",
    "SHY": "duration",
    "LQD": "credit",
    "HYG": "credit",
    "GLD": "commodity",
    "DBC": "commodity",
    "XLK": "sector",
    "XLF": "sector",
    "XLE": "sector",
    "XLV": "sector",
    "XLY": "sector",
    "XLP": "sector",
    "XLI": "sector",
    "XLB": "sector",
    "XLU": "sector",
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
}

# Same survivor list the flagship study disclosed. Ineligible for the endorsement.
STOCKS = [
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL", "TXN", "QCOM", "AMZN",
    "JPM", "BAC", "WFC", "C", "AXP", "GS", "MS",
    "JNJ", "PFE", "MRK", "UNH", "ABT", "LLY", "BMY", "AMGN",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "DIS",
    "GE", "BA", "CAT", "MMM", "HON", "UPS",
]

YIELD_SYMBOLS = ["^IRX", "^FVX", "^TNX", "^TYX", "^VIX"]
DOLLAR_SYMBOLS = ["UUP"]

# Which treasury yield is the carry proxy for a bond ETF.
# Treasury yield used as a carry proxy. Credit ETFs have no clean free yield
# series in this cache, so they are left to the credit relative-strength signal.
CARRY_YIELD = {
    "SHY": "^IRX",
    "IEF": "^FVX",
    "TLT": "^TYX",
}
