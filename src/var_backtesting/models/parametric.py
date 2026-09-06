"""Parametric one-day VaR forecasts using past returns only."""

import pandas as pd
from scipy.stats import norm, t

from .base import make_var_output


def normal_var(portfolio, window, confidence_level):
    """Forecast Normal quantiles using rolling mean and sample standard deviation."""
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Calculate rolling mean using past returns only
    rolling_mean = portfolio["log_ret"].shift(1).rolling(window=window).mean()

    # Calculate rolling standard deviation using past returns only
    rolling_std = portfolio["log_ret"].shift(1).rolling(window=window).std()

    # Normal left-tail quantile
    z_score = norm.ppf(tail_prob)

    # Parametric Normal VaR threshold
    var_threshold = rolling_mean + rolling_std * z_score

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="Normal VaR",
        confidence_level=confidence_level,
        window=window,
    )


def student_t_var(portfolio, window, confidence_level):
    """Fit a location-scale Student-t distribution separately for each forecast."""
    # Convert confidence level into tail probability
    tail_prob = 1 - confidence_level

    # Prepare empty Series for VaR threshold
    var_threshold = pd.Series(index=portfolio.index, dtype=float)

    # Rolling estimation of Student-t parameters
    for i in range(window, len(portfolio)):
        # Use only past returns to avoid look-ahead bias
        sample = portfolio["log_ret"].iloc[i - window : i]

        # Fit Student-t distribution to the rolling window
        nu, loc, scale = t.fit(sample)

        # Calculate left-tail quantile of fitted Student-t distribution
        var_threshold.iloc[i] = t.ppf(tail_prob, df=nu, loc=loc, scale=scale)

    # Return results in the standard VaR output format
    return make_var_output(
        portfolio=portfolio,
        var_threshold=var_threshold,
        model_name="Student-t VaR",
        confidence_level=confidence_level,
        window=window,
    )
