"""Adjusted Yahoo Finance prices and validated offline input."""

import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted Close prices and retain dates shared by the full universe."""
    raw_data = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=True, progress=False
    )

    clean_prices = raw_data["Close"].dropna()

    return clean_prices


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reject incomplete universes instead of silently changing portfolio weights."""
    if not set(tickers).issubset(prices.columns):
        raise ValueError("Price input must contain all configured ETF tickers.")
    prices = prices.loc[:, tickers].dropna()
    if (
        prices.empty
        or not prices.index.is_monotonic_increasing
        or prices.index.has_duplicates
    ):
        raise ValueError("Prices must have a nonempty, unique, ascending date index.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and positive.")
    return prices
