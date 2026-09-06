"""Original dissertation parameters and ETF metadata."""

import pandas as pd

tickers = [
    "SPY",  # United States
    "EWA",  # Australia
    "EWC",  # Canada
    "EWG",  # Germany
    "EWH",  # Hong Kong
    "EWJ",  # Japan
    "EWL",  # Switzerland
    "EWU",  # United Kingdom
    "EWT",  # Taiwan
    "EWZ",  # Brazil
]

start_date = "2005-01-01"
end_date = "2026-01-01"
initial_value = 1_000_000
window = 252
confidence_levels = [0.95, 0.99]


etf_info = pd.DataFrame(
    {
        "ticker": [
            "SPY",
            "EWA",
            "EWC",
            "EWG",
            "EWH",
            "EWJ",
            "EWL",
            "EWU",
            "EWT",
            "EWZ",
        ],
        "market_exposure": [
            "United States",
            "Australia",
            "Canada",
            "Germany",
            "Hong Kong",
            "Japan",
            "Switzerland",
            "United Kingdom",
            "Taiwan",
            "Brazil",
        ],
        "local_market_currency": [
            "USD",
            "AUD",
            "CAD",
            "EUR",
            "HKD",
            "JPY",
            "CHF",
            "GBP",
            "TWD",
            "BRL",
        ],
        "listing_currency": ["USD"] * 10,
    }
)
