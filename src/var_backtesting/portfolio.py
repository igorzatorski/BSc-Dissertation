"""Daily rebalanced equal-weight portfolio; original return alignment retained."""

import numpy as np
import pandas as pd


def build_equal_weighted_portfolio(prices, initial_value):
    """Build the daily rebalanced portfolio, retaining original log-return alignment."""
    # Calculate daily simple returns for each ETF
    individual_returns = prices.pct_change().dropna()

    # Create equal weights for all ETFs
    n_assets = individual_returns.shape[1]
    weights = pd.Series(1 / n_assets, index=individual_returns.columns)

    # Daily rebalanced portfolio return
    portfolio_simple_return = individual_returns.dot(weights)

    # Convert daily portfolio returns into portfolio value
    portfolio_value = initial_value * (1 + portfolio_simple_return).cumprod()

    # Store portfolio value and returns in one DataFrame
    portfolio = pd.DataFrame(
        {"price": portfolio_value, "simple_ret": portfolio_simple_return}
    )

    # Calculate log returns for VaR modelling
    portfolio["log_ret"] = np.log(portfolio["price"] / portfolio["price"].shift(1))

    # Remove the first missing log return
    portfolio = portfolio.dropna()

    return portfolio
